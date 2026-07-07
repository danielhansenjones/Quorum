from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml
from jinja2 import DictLoader, Environment, select_autoescape
from markdown_it import MarkdownIt

DEFAULT_SRC = Path("eval/results/campaign-critic")
DEFAULT_GOLD = Path("eval/datasets/v1/gold.yaml")
DEFAULT_OUT = Path("site")

_md = MarkdownIt("commonmark")  # raw HTML disabled by default; model output is escaped
_STATUS_ORDER = {"ok": 0, "partial": 1, "refused": 2}


def _quality_mean(quality: dict | None) -> float | None:
    if not quality:
        return None
    dims = ("clarity", "comparison_quality", "evidence_coverage", "honesty_on_insufficient_data")
    vals = [quality[d] for d in dims if isinstance(quality.get(d), int)]
    return sum(vals) / len(vals) if vals else None


def _counter_summary(counter: Any) -> str | None:
    if isinstance(counter, dict):
        parts = [counter.get("ticker"), counter.get("section"), counter.get("quote")]
        return " - ".join(str(p) for p in parts if p) or None
    if isinstance(counter, str) and counter.strip():
        return counter.strip()
    return None


def _fmt_amount(value: Any, unit: Any) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value) if value else ""
    if unit == "USD":
        a = abs(v)
        if a >= 1e9:
            return f"${v / 1e9:.2f}B"
        if a >= 1e6:
            return f"${v / 1e6:.2f}M"
        if a >= 1e3:
            return f"${v / 1e3:.1f}K"
        return f"${v:,.0f}"
    return f"{v:,.4g}" + (f" {unit}" if unit else "")


def _citation(c: dict[str, Any]) -> dict[str, Any]:
    # Quant citations are code-built XBRL facts (concept/value/period, no quote);
    # qual citations carry a filing excerpt. The `claim` field is the whole axis
    # paragraph, identical across every citation, so it is not shown per-row - it
    # is the report text already rendered above.
    kind = c.get("kind", "")
    if kind == "quant":
        return {
            "kind": "quant",
            "ticker": c.get("ticker", ""),
            "concept": c.get("concept", ""),
            "amount": _fmt_amount(c.get("value"), c.get("unit")),
            "period": c.get("period", ""),
            "section": "",
            "quote": "",
        }
    return {
        "kind": kind,
        "ticker": c.get("ticker", ""),
        "concept": "",
        "amount": "",
        "period": "",
        "section": c.get("section", ""),
        "quote": c.get("quote", ""),
    }


def _load_questions(gold: Path) -> dict[str, str]:
    cases = yaml.safe_load(gold.read_text())["cases"]
    return {c["id"]: c["question"] for c in cases if c.get("id")}


def _build_record(path: Path, questions: dict[str, str]) -> dict[str, Any]:
    d = json.loads(path.read_text())
    cid = d.get("case_id") or path.stem
    scores = d.get("scores") or {}
    quality = scores.get("quality") or {}
    faith = scores.get("faithfulness") or {}
    critique = d.get("critique") or {}
    flagged = [
        {
            "flag": fc.get("flag", ""),
            "source_axis": fc.get("source_axis", ""),
            "claim": fc.get("claim", ""),
            "reason": fc.get("reason", ""),
            "counter": _counter_summary(fc.get("counter_citation")),
        }
        for fc in (critique.get("flagged_claims") or [])
    ]
    citations = [_citation(c) for c in (d.get("citations") or [])]
    per_axis = [
        {
            "axis": v.get("axis", k),
            "groundedness": v.get("groundedness", ""),
            "notes": v.get("notes", ""),
            "missed_evidence": v.get("missed_evidence", ""),
        }
        for k, v in (critique.get("per_axis") or {}).items()
        if isinstance(v, dict)
    ]
    return {
        "case_id": cid,
        "slug": cid,
        "question": questions.get(cid, cid),
        "status": d.get("final_status", ""),
        "elapsed_s": d.get("elapsed_s"),
        "report_html": _md.render(d.get("report") or ""),
        "citations": citations,
        "flagged": flagged,
        "cross_axis": critique.get("cross_axis") or [],
        "per_axis": per_axis,
        "critique_status": critique.get("status", ""),
        "turns_used": critique.get("turns_used"),
        "quality": quality,
        "quality_mean": _quality_mean(quality),
        "faithfulness": faith,
        "n_flags": len(flagged),
        "n_citations": len(citations),
    }


def _env() -> Environment:
    return Environment(
        loader=DictLoader({"index.html": _INDEX, "case.html": _CASE}),
        autoescape=select_autoescape(["html"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render the critic-ON campaign arm to a static gallery (index + per-case pages)."
    )
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    questions = _load_questions(args.gold)
    paths = [
        p
        for p in sorted(args.src.glob("*.json"))
        if p.name not in ("summary.json", "cost_report.json")
    ]
    records = [_build_record(p, questions) for p in paths]
    records.sort(key=lambda r: (_STATUS_ORDER.get(r["status"], 9), r["case_id"]))

    total_flags = sum(r["n_flags"] for r in records)
    stats = {
        "n_cases": len(records),
        "n_ok": sum(1 for r in records if r["status"] == "ok"),
        "n_partial": sum(1 for r in records if r["status"] == "partial"),
        "n_refused": sum(1 for r in records if r["status"] == "refused"),
        "total_flags": total_flags,
    }

    env = _env()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "style.css").write_text(_STYLE)
    (args.out / "index.html").write_text(
        env.get_template("index.html").render(records=records, stats=stats)
    )
    case_tmpl = env.get_template("case.html")
    for r in records:
        (args.out / f"{r['slug']}.html").write_text(case_tmpl.render(r=r))

    print(f"wrote {len(records)} case pages + index to {args.out}")
    return 0


_STYLE = """
:root {
  --bg: #0f1115; --panel: #171a21; --border: #262b36; --text: #d7dbe0;
  --muted: #8b93a1; --accent: #6ea8fe; --ok: #3fb950; --partial: #d29922; --refused: #6e7681;
  --flag: #f0883e; --mono: ui-monospace, SFMono-Regular, Menlo, monospace;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
  font: 15px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
.wrap { max-width: 860px; margin: 0 auto; padding: 2rem 1.25rem 4rem; }
h1 { font-size: 1.6rem; margin: 0 0 .35rem; }
.lede { color: var(--muted); margin: 0 0 1.5rem; }
.statbar { display: flex; flex-wrap: wrap; gap: .5rem 1.25rem; color: var(--muted);
  font-size: .85rem; margin-bottom: 1.75rem; }
.statbar b { color: var(--text); }
.cards { display: grid; grid-template-columns: 1fr; gap: .75rem; }
.card { display: block; background: var(--panel); border: 1px solid var(--border);
  border-radius: 10px; padding: 1rem 1.1rem; color: var(--text); }
.card:hover { border-color: var(--accent); text-decoration: none; }
.card .q { font-weight: 600; margin-bottom: .5rem; }
.card .meta { display: flex; flex-wrap: wrap; gap: .4rem .9rem; font-size: .8rem; color: var(--muted); }
.badge { display: inline-block; padding: .05rem .5rem; border-radius: 999px; font-size: .72rem;
  font-weight: 600; text-transform: uppercase; letter-spacing: .03em; }
.badge.ok { background: rgba(63,185,80,.15); color: var(--ok); }
.badge.partial { background: rgba(210,153,34,.15); color: var(--partial); }
.badge.refused { background: rgba(110,118,129,.2); color: var(--refused); }
.badge.flag { background: rgba(240,136,62,.15); color: var(--flag); }
.badge.quant { background: rgba(110,168,254,.15); color: var(--accent); }
.badge.qual { background: rgba(139,147,161,.18); color: var(--muted); }
.back { font-size: .85rem; }
.metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: .6rem; margin: 1.25rem 0; }
.metric { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: .6rem .75rem; }
.metric .k { font-size: .72rem; color: var(--muted); text-transform: uppercase; letter-spacing: .03em; }
.metric .v { font-size: 1.15rem; font-weight: 600; }
section { margin: 2rem 0; }
section > h2 { font-size: 1.05rem; border-bottom: 1px solid var(--border); padding-bottom: .35rem; }
.report h2 { font-size: 1.05rem; margin-top: 1.4rem; }
.report h3 { font-size: .95rem; }
.report { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: .5rem 1.25rem; }
.flag-item, .cite, .axis { background: var(--panel); border: 1px solid var(--border);
  border-radius: 8px; padding: .75rem .9rem; margin-bottom: .6rem; }
.flag-item .claim, .cite .quote { font-style: italic; color: var(--text); }
.flag-item .reason, .cite .claim { color: var(--muted); font-size: .9rem; margin-top: .35rem; }
.cite .head, .flag-item .head, .axis .head { display: flex; flex-wrap: wrap; gap: .5rem;
  align-items: center; font-size: .82rem; color: var(--muted); margin-bottom: .4rem; }
.cite .fact { font-size: .9rem; }
.cite .fact .mono { color: var(--text); }
.mono { font-family: var(--mono); }
.notes { color: var(--muted); font-size: .9rem; }
.empty { color: var(--muted); font-style: italic; }
"""

_INDEX = """<div class="wrap">
<h1>Quorum - report gallery</h1>
<p class="lede">Real outputs from the critic-ON campaign arm. Each page shows the question,
the final report, the critic's flags, and the LLM-judge scores - no clone, no API key.</p>
<div class="statbar">
  <span><b>{{ stats.n_cases }}</b> cases</span>
  <span><b>{{ stats.n_ok }}</b> ok</span>
  <span><b>{{ stats.n_partial }}</b> partial</span>
  <span><b>{{ stats.n_refused }}</b> refused</span>
  <span><b>{{ stats.total_flags }}</b> critic flags</span>
</div>
<div class="cards">
{% for r in records %}
  <a class="card" href="{{ r.slug }}.html">
    <div class="q">{{ r.question }}</div>
    <div class="meta">
      <span class="badge {{ r.status }}">{{ r.status }}</span>
      {% if r.quality_mean is not none %}<span>quality {{ "%.1f"|format(r.quality_mean) }}/5</span>{% endif %}
      {% if r.faithfulness.mean_score is not none %}<span>faithfulness {{ "%.1f"|format(r.faithfulness.mean_score) }}/5</span>{% endif %}
      <span>{{ r.n_flags }} flag{{ '' if r.n_flags == 1 else 's' }}</span>
      <span>{{ r.n_citations }} citation{{ '' if r.n_citations == 1 else 's' }}</span>
    </div>
  </a>
{% endfor %}
</div>
</div>
"""

_CASE = """<div class="wrap">
<a class="back" href="index.html">&larr; all cases</a>
<h1>{{ r.question }}</h1>
<div><span class="badge {{ r.status }}">{{ r.status }}</span></div>
<div class="metrics">
  {% if r.quality.clarity is defined %}
  <div class="metric"><div class="k">clarity</div><div class="v">{{ r.quality.clarity }}/5</div></div>
  <div class="metric"><div class="k">comparison</div><div class="v">{{ r.quality.comparison_quality }}/5</div></div>
  <div class="metric"><div class="k">evidence</div><div class="v">{{ r.quality.evidence_coverage }}/5</div></div>
  <div class="metric"><div class="k">honesty</div><div class="v">{{ r.quality.honesty_on_insufficient_data }}/5</div></div>
  {% endif %}
  {% if r.faithfulness.mean_score is not none %}
  <div class="metric"><div class="k">faithfulness</div><div class="v">{{ "%.1f"|format(r.faithfulness.mean_score) }}/5</div></div>
  {% endif %}
  {% if r.elapsed_s is not none %}
  <div class="metric"><div class="k">elapsed</div><div class="v">{{ "%.1f"|format(r.elapsed_s) }}s</div></div>
  {% endif %}
</div>

<section>
  <h2>Report</h2>
  <div class="report">{{ r.report_html|safe }}</div>
</section>

<section>
  <h2>Critic flags ({{ r.n_flags }})</h2>
  {% for f in r.flagged %}
  <div class="flag-item">
    <div class="head"><span class="badge flag">{{ f.flag }}</span><span class="mono">{{ f.source_axis }}</span></div>
    <div class="claim">{{ f.claim }}</div>
    <div class="reason">{{ f.reason }}</div>
    {% if f.counter %}<div class="reason">counter: {{ f.counter }}</div>{% endif %}
  </div>
  {% else %}
  <p class="empty">No claims flagged.</p>
  {% endfor %}
  {% if r.cross_axis %}
  <h2>Cross-axis notes</h2>
  {% for note in r.cross_axis %}<div class="axis"><div class="notes">{{ note }}</div></div>{% endfor %}
  {% endif %}
</section>

<section>
  <h2>Citations ({{ r.n_citations }})</h2>
  {% for c in r.citations %}
  <div class="cite">
    <div class="head">
      <span class="badge {{ c.kind }}">{{ c.kind }}</span>
      <span class="mono">{{ c.ticker }}</span>
      {% if c.period %}<span class="mono">{{ c.period }}</span>{% endif %}
      {% if c.section %}<span class="mono">{{ c.section }}</span>{% endif %}
    </div>
    {% if c.concept %}<div class="fact"><span class="mono">{{ c.concept }}</span> = <b>{{ c.amount }}</b></div>{% endif %}
    {% if c.quote %}<div class="quote">"{{ c.quote }}"</div>{% endif %}
  </div>
  {% else %}
  <p class="empty">No citations.</p>
  {% endfor %}
</section>
</div>
"""


if __name__ == "__main__":
    raise SystemExit(main())
