# /// script
# requires-python = ">=3.12"
# dependencies = ["pillow"]
# ///
"""Render the agent-graph demo GIF from a recorded replay fixture.

Draws the graph topology (fan-out, critic tool loop, replan back-edge) as an
animated GIF, one scene per recorded event. Same input as scripts/demo.py
--replay, so the recording stays reproducible with no key, server, or Docker.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

DEFAULT_FIXTURE = Path("eval/fixtures/demo_replay.jsonl")
DEFAULT_OUT = Path("assets/demo.gif")

W, H = 1280, 680
FRAME_MS = 90

BG = "#0f1317"
CARD = "#1c242d"
CARD_DIM = "#161c23"
LINE = "#2b3642"
INK = "#d7e0e8"
MUTED = "#7e8b97"
ACCENT = "#4cc3e0"
OK = "#58b981"
WARN = "#d9a545"
CRITIC = "#c78bd4"
TERM_BG = "#0b0e11"
TERM_DIM = "#66737f"

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")


def _font(name: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = FONT_DIR / name
    if path.is_file():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size)


F_NAME = _font("DejaVuSansMono-Bold.ttf", 14)
F_ROLE = _font("DejaVuSansMono.ttf", 9)
F_DETAIL = _font("DejaVuSansMono.ttf", 11)
F_HEADER = _font("DejaVuSansMono-Bold.ttf", 16)
F_HEADER_Q = _font("DejaVuSansMono.ttf", 13)
F_CHIP = _font("DejaVuSansMono.ttf", 11)
F_RAIL = _font("DejaVuSansMono-Bold.ttf", 10)
F_TICK = _font("DejaVuSansMono.ttf", 12)
F_TICK_B = _font("DejaVuSansMono-Bold.ttf", 12)
F_EDGE = _font("DejaVuSansMono.ttf", 10)

CARD_W, CARD_H, COL_GAP = 130, 74, 27
AXIS_Y = 185
RAIL_TOP, RAIL_BOT = 312, 476
TERM_TOP, TERM_BOT = 486, 656


def _hex(c: str) -> tuple[int, int, int]:
    return int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)


def _blend(a: str, b: str, t: float) -> tuple[int, int, int]:
    ra, ga, ba = _hex(a)
    rb, gb, bb = _hex(b)
    return (int(ra + (rb - ra) * t), int(ga + (gb - ga) * t), int(ba + (bb - ba) * t))


def _truncate(s: str, n: int) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 3] + "..."


def _cubic(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    n: int = 48,
) -> list[tuple[float, float]]:
    pts = []
    for i in range(n + 1):
        t = i / n
        u = 1 - t
        x = u**3 * p0[0] + 3 * u**2 * t * p1[0] + 3 * u * t**2 * p2[0] + t**3 * p3[0]
        y = u**3 * p0[1] + 3 * u**2 * t * p1[1] + 3 * u * t**2 * p2[1] + t**3 * p3[1]
        pts.append((x, y))
    return pts


@dataclass
class Node:
    name: str
    role: str
    col: int
    row: int = 0
    rows: int = 1
    state: str = "pending"  # pending | running | done
    ground: str | None = None  # ok | weak
    detail: list[str] = field(default_factory=list)
    border: str = ""  # override for the critic's magenta

    def rect(self) -> tuple[int, int, int, int]:
        x = 25 + self.col * (CARD_W + COL_GAP)
        total = self.rows * CARD_H + (self.rows - 1) * 16
        y = AXIS_Y - total // 2 + self.row * (CARD_H + 16)
        return x, y, x + CARD_W, y + CARD_H


@dataclass
class Edge:
    src: str
    dst: str
    kind: str = "h"  # h | back | down
    state: str = "pending"  # pending | active | done | firing | dormant


@dataclass
class State:
    question: str = ""
    tag: str = "replay"
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: dict[str, Edge] = field(default_factory=dict)
    chips: list[tuple[str, str]] = field(default_factory=list)  # (kind, text)
    rail_live: bool = False
    rail_note: str = ""
    ticker: list[list[tuple[str, str, bool]]] = field(default_factory=list)

    def tick(self, label: str, body: str, color: str = INK, bold: bool = False) -> None:
        segs: list[tuple[str, str, bool]] = []
        if label:
            segs.append((f"{label:<11} ", ACCENT if not bold else INK, bold))
        segs.append((body, color, bold))
        self.ticker.append(segs)


def _edge_points(st: State, e: Edge) -> list[tuple[float, float]]:
    a = st.nodes[e.src].rect()
    if e.kind == "down":
        x = (a[0] + a[2]) / 2
        return [(x, a[3]), (x, RAIL_TOP)]
    b = st.nodes[e.dst].rect()
    if e.kind == "back":
        s = ((a[0] + a[2]) / 2, a[1])
        t = ((b[0] + b[2]) / 2, b[1])
        return _cubic(s, (s[0], s[1] - 58), (t[0], t[1] - 58), t)
    s = (a[2], (a[1] + a[3]) / 2)
    t = (b[0], (b[1] + b[3]) / 2)
    dx = max(18.0, (t[0] - s[0]) / 2)
    return _cubic(s, (s[0] + dx, s[1]), (t[0] - dx, t[1]), t)


def _draw_dashed(
    d: ImageDraw.ImageDraw,
    pts: list[tuple[float, float]],
    color: Any,
    width: int,
    offset: float,
) -> None:
    on, period = 7.0, 12.0
    dist = 0.0
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        seg = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        if (dist + seg / 2 + offset) % period < on:
            d.line([(x0, y0), (x1, y1)], fill=color, width=width)
        dist += seg


def _draw_arrow(d: ImageDraw.ImageDraw, pts: list[tuple[float, float]], color: Any) -> None:
    (x0, y0), (x1, y1) = pts[-2], pts[-1]
    dx, dy = x1 - x0, y1 - y0
    n = (dx**2 + dy**2) ** 0.5 or 1.0
    ux, uy = dx / n, dy / n
    px, py = -uy, ux
    d.polygon(
        [
            (x1, y1),
            (x1 - 7 * ux + 3.5 * px, y1 - 7 * uy + 3.5 * py),
            (x1 - 7 * ux - 3.5 * px, y1 - 7 * uy - 3.5 * py),
        ],
        fill=color,
    )


def _draw_edge(d: ImageDraw.ImageDraw, st: State, e: Edge, phase: int) -> None:
    pts = _edge_points(st, e)
    if e.state == "pending":
        color: Any = LINE
        if e.kind != "back":
            d.line(pts, fill=color, width=1)
        else:
            _draw_dashed(d, pts, MUTED, 1, 0)
    elif e.state == "dormant":
        _draw_dashed(d, pts, MUTED, 1, 0)
    elif e.state == "active":
        _draw_dashed(d, pts, _hex(ACCENT), 2, -phase * 3.0)
        _draw_arrow(d, pts, _hex(ACCENT))
    elif e.state == "firing":
        _draw_dashed(d, pts, _hex(WARN), 2, phase * 3.0)
        _draw_arrow(d, pts, _hex(WARN))
    else:  # done
        d.line(pts, fill=_blend(ACCENT, BG, 0.45), width=2)
        _draw_arrow(d, pts, _blend(ACCENT, BG, 0.45))
    if e.kind == "back":
        top_y = min(p[1] for p in pts)
        mid_x = (pts[0][0] + pts[-1][0]) / 2
        label = "replan (on weak)"
        lw = d.textlength(label, font=F_EDGE)
        color = WARN if e.state == "firing" else MUTED
        d.text((mid_x - lw / 2, top_y - 13), label, font=F_EDGE, fill=color)


def _draw_node(d: ImageDraw.ImageDraw, node: Node, phase: int) -> None:
    x0, y0, x1, y1 = node.rect()
    dim = node.state == "pending"
    fill = CARD_DIM if dim else CARD
    if node.state == "running":
        border: Any = node.border or ACCENT
        bw = 2
    elif node.state == "done":
        border = node.border or {"ok": OK, "weak": WARN}.get(
            node.ground or "", _blend(LINE, INK, 0.25)
        )
        bw = 1 if node.ground != "weak" else 2
    else:
        border = LINE
        bw = 1
    d.rounded_rectangle((x0, y0, x1, y1), radius=6, fill=fill, outline=border, width=bw)
    role_c = _blend(MUTED, BG, 0.4) if dim else MUTED
    name_c = _blend(INK, BG, 0.55) if dim else INK
    d.text((x0 + 10, y0 + 7), node.role.upper(), font=F_ROLE, fill=role_c)
    d.text((x0 + 10, y0 + 19), node.name, font=F_NAME, fill=name_c)
    if node.state == "running":
        t = [1.0, 0.6, 0.25, 0.6][phase % 4]
        nw = d.textlength(node.name, font=F_NAME)
        cx, cy = x0 + 10 + nw + 10, y0 + 27
        d.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=_blend(CARD, node.border or ACCENT, t))
    for i, ln in enumerate(node.detail[:2]):
        color = MUTED
        if ln == "grounding ok":
            color = OK
        elif ln == "grounding weak":
            color = WARN
        elif node.state == "done":
            color = _blend(INK, BG, 0.2)
        d.text((x0 + 10, y0 + 40 + i * 14), _truncate(ln, 17), font=F_DETAIL, fill=color)


def _draw_rail(d: ImageDraw.ImageDraw, st: State, phase: int) -> None:
    color = CRITIC if st.rail_live else LINE
    step = 4
    for x in range(25, W - 25, step * 2):
        d.line([(x, RAIL_TOP), (min(x + step, W - 25), RAIL_TOP)], fill=color, width=1)
        d.line([(x, RAIL_BOT), (min(x + step, W - 25), RAIL_BOT)], fill=color, width=1)
    for y in range(RAIL_TOP, RAIL_BOT, step * 2):
        d.line([(25, y), (25, min(y + step, RAIL_BOT))], fill=color, width=1)
        d.line([(W - 25, y), (W - 25, min(y + step, RAIL_BOT))], fill=color, width=1)
    head = "CRITIC RUNS ITS OWN TOOL LOOP"
    head_c = CRITIC if st.rail_live else MUTED
    d.text((41, RAIL_TOP + 10), head, font=F_RAIL, fill=head_c)
    if st.rail_note:
        hw = d.textlength(head, font=F_RAIL)
        d.text((41 + hw + 14, RAIL_TOP + 10), st.rail_note, font=F_CHIP, fill=MUTED)
    x, y = 41, RAIL_TOP + 32
    for kind, text in st.chips:
        tw = d.textlength(text, font=F_CHIP)
        cw = tw + 18
        if x + cw > W - 41:
            x = 41
            y += 30
        border = WARN if kind == "flag" else LINE
        fill = None if kind == "flag" else CARD_DIM
        ink = WARN if kind == "flag" else INK
        d.rounded_rectangle((x, y, x + cw, y + 22), radius=4, fill=fill, outline=border, width=1)
        d.text((x + 9, y + 4), text, font=F_CHIP, fill=ink)
        x += cw + 8


def _draw_ticker(d: ImageDraw.ImageDraw, st: State) -> None:
    d.rounded_rectangle((25, TERM_TOP, W - 25, TERM_BOT), radius=6, fill=TERM_BG)
    lines = st.ticker[-9:]
    y = TERM_TOP + 10
    for segs in lines:
        x = 41.0
        for text, color, bold in segs:
            font = F_TICK_B if bold else F_TICK
            d.text((x, y), text, font=font, fill=color)
            x += d.textlength(text, font=font)
        y += 17


def render_frame(st: State, phase: int) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    x = 28.0
    d.text((x, 22), "quorum", font=F_HEADER, fill=INK)
    x += d.textlength("quorum", font=F_HEADER) + 8
    d.text((x, 22), ">", font=F_HEADER, fill=ACCENT)
    x += d.textlength("> ", font=F_HEADER) + 4
    d.text((x, 25), _truncate(st.question, 80), font=F_HEADER_Q, fill=_blend(INK, BG, 0.2))
    x += d.textlength(_truncate(st.question, 80), font=F_HEADER_Q) + 12
    d.text((x, 25), f"({st.tag})", font=F_HEADER_Q, fill=MUTED)
    for e in st.edges.values():
        _draw_edge(d, st, e, phase)
    for node in st.nodes.values():
        _draw_node(d, node, phase)
    _draw_rail(d, st, phase)
    _draw_ticker(d, st)
    return img


def _animated(st: State) -> bool:
    if any(n.state == "running" for n in st.nodes.values()):
        return True
    return any(e.state in ("active", "firing") for e in st.edges.values())


class Recording:
    def __init__(self) -> None:
        self.frames: list[Image.Image] = []
        self.durations: list[int] = []

    def hold(self, st: State, seconds: float) -> None:
        if not _animated(st):
            self.frames.append(render_frame(st, 0))
            self.durations.append(int(seconds * 1000))
            return
        n = max(1, round(seconds * 1000 / FRAME_MS))
        for i in range(n):
            self.frames.append(render_frame(st, i))
            self.durations.append(FRAME_MS)


def _fmt_value(result: str) -> str:
    m = re.search(r'"value":\s*(-?[\d.]+)', result)
    if not m:
        return _truncate(result, 14)
    v = float(m.group(1))
    if abs(v) >= 1e12:
        return f"${v / 1e12:.1f}T"
    if abs(v) >= 1e9:
        return f"${v / 1e9:.1f}B"
    if abs(v) >= 1e6:
        return f"${v / 1e6:.1f}M"
    return f"${v:,.0f}"


def _short_key(key: str) -> str:
    return key.split(":", 1)[-1]


def build(events: list[dict[str, Any]], rec: Recording) -> None:
    st = State()
    st.nodes = {
        "classify": Node("classify", "gate", 0),
        "resolve": Node("resolve", "gate", 1),
        "plan": Node("plan", "supervisor", 2),
        "assess": Node("assess", "judge", 4),
        "critic": Node("critic", "audit agent", 5, border=CRITIC),
        "synthesize": Node("synthesize", "writer", 6),
        "report": Node("report", "output", 7),
    }
    st.edges = {
        "c-r": Edge("classify", "resolve"),
        "r-p": Edge("resolve", "plan"),
        "a-cr": Edge("assess", "critic"),
        "cr-s": Edge("critic", "synthesize"),
        "s-rep": Edge("synthesize", "report"),
        "fb": Edge("assess", "plan", kind="back", state="dormant"),
        "cb": Edge("critic", "critic", kind="down"),
    }
    analysts: dict[str, str] = {}  # axis -> node id
    plan_seen = False

    def analyst_slots(tasks: list[str]) -> None:
        for i, axis in enumerate(tasks):
            nid = f"a{i}"
            analysts[axis] = nid
            st.nodes[nid] = Node(_truncate(axis, 15), "analyst agent", 3, row=i, rows=len(tasks))
            st.edges[f"p-{nid}"] = Edge("plan", nid)
            st.edges[f"{nid}-a"] = Edge(nid, "assess")

    for entry in events:
        event, data = entry["event"], entry["data"]
        if event == "meta":
            st.question = data.get("question", "")
            rec.hold(st, 0.5)
            continue
        if event != "node":
            if event == "final":
                for eid in ("s-rep",):
                    st.edges[eid].state = "done"
                cits = len(data.get("citations") or [])
                st.nodes["report"].state = "done"
                st.nodes["report"].ground = "ok" if data.get("status") == "ok" else "weak"
                st.nodes["report"].detail = [str(data.get("status")), f"{cits} citations"]
                st.tick("REPORT", f"status={data.get('status')}  {cits} citations", bold=True)
                rec.hold(st, 0.4)
            elif event == "cost":
                totals = data.get("totals", {})
                spent = totals.get("cost_effective")
                body = f"total ${totals.get('cost', 0.0):.4f}"
                if spent is not None:
                    body += f"  spent ${spent:.4f}"
                body += f"  cache_read={totals.get('cache_read_fraction', 0.0):.0%}"
                st.tick("COST", body, color=TERM_DIM)
                rec.hold(st, 3.0)
            continue

        node, detail = data["node"], data.get("detail", {})
        if node == "entry":
            continue
        if node == "classify":
            st.nodes["classify"].state = "running"
            rec.hold(st, 0.65)
            st.nodes["classify"].state = "done"
            if detail.get("out_of_scope"):
                st.nodes["classify"].ground = "weak"
                st.nodes["classify"].detail = ["out of scope"]
                st.tick("classify", "out of scope", color=WARN)
                continue
            axes = detail.get("axes", [])
            st.nodes["classify"].ground = "ok"
            st.nodes["classify"].detail = ["in-scope", f"{len(axes)} axes"]
            st.tick(
                "classify",
                f"in-scope  axes=[{', '.join(axes)}]"
                f"  mentions=[{', '.join(detail.get('companies_raw', []))}]",
            )
            st.edges["c-r"].state = "active"
            st.nodes["resolve"].state = "running"
        elif node == "resolve":
            rec.hold(st, 0.65)
            tickers = detail.get("tickers", [])
            st.edges["c-r"].state = "done"
            st.nodes["resolve"].state = "done"
            st.nodes["resolve"].ground = "ok"
            st.nodes["resolve"].detail = [", ".join(tickers)]
            st.tick("resolve", f"tickers=[{', '.join(tickers)}]")
            st.edges["r-p"].state = "active"
            st.nodes["plan"].state = "running"
        elif node == "plan":
            tasks = detail.get("tasks", [])
            replan = detail.get("replan_count") or 0
            if replan:
                st.edges["fb"].state = "firing"
                st.nodes["plan"].state = "running"
                st.nodes["plan"].detail = [f"re-plan #{replan}"]
                st.tick(
                    "plan",
                    f"re-plan #{replan} -> re-dispatch: [{', '.join(tasks)}]"
                    f"   budget={detail.get('remaining_steps')}",
                    color=WARN,
                )
                rec.hold(st, 1.0)
                st.edges["fb"].state = "dormant"
            else:
                rec.hold(st, 0.7)
                if not plan_seen:
                    analyst_slots(tasks)
                    plan_seen = True
                st.tick(
                    "plan",
                    f"{len(tasks)} analyst task(s) -> fan-out: [{', '.join(tasks)}]"
                    f"   budget={detail.get('remaining_steps')}",
                )
            st.edges["r-p"].state = "done"
            st.nodes["plan"].state = "done"
            st.nodes["plan"].ground = "ok"
            st.nodes["plan"].detail = [
                f"re-plan #{replan}" if replan else f"fan-out: {len(tasks)}",
                f"budget: {detail.get('remaining_steps')}",
            ]
            for axis in tasks:
                nid = analysts.get(axis)
                if nid is None:
                    continue
                st.edges[f"p-{nid}"].state = "active"
                st.nodes[nid].state = "running"
                st.nodes[nid].ground = None
                st.nodes[nid].detail = ["re-running..." if replan else "running..."]
            rec.hold(st, 1.05)
        elif node == "analyze_axis":
            for r in detail.get("results", []):
                nid = analysts.get(r["axis"])
                if nid is None:
                    continue
                st.edges[f"p-{nid}"].state = "done"
                st.nodes[nid].state = "done"
                st.nodes[nid].ground = (
                    r["grounding"] if r["grounding"] in ("ok", "weak") else "weak"
                )
                st.nodes[nid].detail = [f"{r['citations']} cites", f"grounding {r['grounding']}"]
                color = OK if r["grounding"] == "ok" else WARN
                st.tick(
                    "analyze",
                    f"{r['axis']:<16} done  grounding={r['grounding']}  {r['citations']} cites",
                    color=color if r["grounding"] != "ok" else INK,
                )
            rec.hold(st, 0.8)
        elif node == "assess":
            for nid in analysts.values():
                if st.nodes[nid].state == "done":
                    st.edges[f"{nid}-a"].state = "active"
            st.nodes["assess"].state = "running"
            rec.hold(st, 0.7)
            weak = detail.get("weak") or 0
            for nid in analysts.values():
                if st.edges[f"{nid}-a"].state == "active":
                    st.edges[f"{nid}-a"].state = "done"
            st.nodes["assess"].state = "done"
            st.nodes["assess"].ground = "weak" if weak else "ok"
            st.nodes["assess"].detail = [f"{detail.get('axes')} axes", f"{weak} weak"]
            st.tick(
                "assess", f"{detail.get('axes')} axes  {weak} weak", color=WARN if weak else INK
            )
            rec.hold(st, 0.65)
        elif node == "critic":
            st.edges["a-cr"].state = "active"
            if "critique" in detail:
                st.nodes["critic"].state = "done"
                st.nodes["critic"].detail = ["unavailable"]
                st.tick("critic", "unavailable (bypassed / timeout / failed)", color=TERM_DIM)
                rec.hold(st, 0.6)
                st.edges["a-cr"].state = "done"
                continue
            st.edges["cb"].state = "active"
            st.nodes["critic"].state = "running"
            st.nodes["critic"].detail = ["auditing..."]
            st.rail_live = True
            st.tick(
                "critic",
                f"agent loop  turns={detail.get('turns_used')}  status={detail.get('status')}",
            )
            rec.hold(st, 0.6)
            for i, tc in enumerate(detail.get("tool_calls", []), 1):
                key = _short_key(str(tc["args"].get("key", tc["tool"])))
                ticker = tc["args"].get("ticker", "")
                value = _fmt_value(tc["result"])
                ok = "ok" if tc["ok"] else "err"
                st.chips.append(("tool", f"[{i}] {key} {ticker} -> {value} {ok}"))
                st.tick(
                    "", f"  [{i}] {tc['tool']}({key} {ticker}) -> {value}  [{ok}]", color=TERM_DIM
                )
                rec.hold(st, 0.38)
            for f in detail.get("flags", []):
                st.chips.append(("flag", f"FLAG {f['flag']}: {_truncate(f['claim'], 58)}"))
                st.tick(
                    "",
                    f"  FLAG [{f['axis']}] {f['flag']}: {_truncate(f['claim'], 66)}",
                    color=WARN,
                )
                rec.hold(st, 0.6)
            n_flags = len(detail.get("flags", []))
            st.nodes["critic"].state = "done"
            st.nodes["critic"].detail = [
                f"turns: {detail.get('turns_used')}",
                f"{n_flags} flags raised",
            ]
            st.rail_note = f"{len(detail.get('tool_calls', []))} tool calls, {n_flags} flags"
            st.edges["a-cr"].state = "done"
            st.edges["cb"].state = "done"
            st.edges["cr-s"].state = "active"
            st.nodes["synthesize"].state = "running"
            rec.hold(st, 0.8)
        elif node == "synthesize":
            st.edges["cr-s"].state = "done"
            st.nodes["synthesize"].state = "done"
            st.nodes["synthesize"].ground = "ok" if detail.get("status") == "ok" else "weak"
            st.nodes["synthesize"].detail = [
                f"status {detail.get('status')}",
                f"{detail.get('citations')} citations",
            ]
            st.tick(
                "synthesize",
                f"status={detail.get('status')}  {detail.get('citations')} citations",
            )
            st.edges["s-rep"].state = "active"
            rec.hold(st, 0.5)
        elif node == "refuse":
            st.nodes["report"].state = "done"
            st.nodes["report"].ground = "weak"
            st.nodes["report"].detail = [str(detail.get("status"))]
            st.tick("refuse", str(detail.get("status")), color=WARN)
            rec.hold(st, 2.0)


def save(rec: Recording, out: Path) -> None:
    mid = rec.frames[len(rec.frames) // 2]
    composite = Image.new("RGB", (W, H * 2), BG)
    composite.paste(mid, (0, 0))
    composite.paste(rec.frames[-1], (0, H))
    palette = composite.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    frames = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in rec.frames]
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=rec.durations,
        loop=0,
        optimize=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render the agent-graph demo GIF from a replay fixture."
    )
    parser.add_argument("fixture", nargs="?", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    events = [
        json.loads(line)
        for line in args.fixture.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rec = Recording()
    build(events, rec)
    save(rec, args.out)
    total = sum(rec.durations) / 1000
    print(f"wrote {args.out} ({len(rec.frames)} frames, {total:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
