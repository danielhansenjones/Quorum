from __future__ import annotations

from typing import Any

from psycopg_pool import ConnectionPool


def _percentile(values: list[float], pct: float) -> float:
    # Nearest-rank percentile; coarse but adequate for eval-scale request counts.
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[k]


def summarize_trace_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    # Pure aggregation over trace_events rows. Each row: request_id, node_name,
    # tokens_in, tokens_out, cache_read_tokens, cost_dollars_billed.
    if not rows:
        return {"requests": 0, "per_request": {}, "per_node": {}, "totals": {}}

    by_request: dict[str, dict[str, float]] = {}
    by_node: dict[str, dict[str, float]] = {}
    tot_in = tot_out = tot_cache = 0
    tot_cost = 0.0
    tot_effective = 0.0

    for r in rows:
        rid = str(r["request_id"])
        node = str(r["node_name"])
        t_in = int(r.get("tokens_in") or 0)
        t_out = int(r.get("tokens_out") or 0)
        t_cache = int(r.get("cache_read_tokens") or 0)
        cost = float(r.get("cost_dollars_billed") or 0.0)
        # Rows written before the split carry effective == billed; disk-cache
        # replays after it carry effective == 0.
        effective = float(r.get("cost_dollars_effective", cost) or 0.0)

        req = by_request.setdefault(rid, {"cost": 0.0, "tokens_in": 0, "tokens_out": 0})
        req["cost"] += cost
        req["tokens_in"] += t_in
        req["tokens_out"] += t_out

        nd = by_node.setdefault(node, {"cost": 0.0, "rows": 0, "tokens_in": 0, "tokens_out": 0})
        nd["cost"] += cost
        nd["rows"] += 1
        nd["tokens_in"] += t_in
        nd["tokens_out"] += t_out

        tot_in += t_in
        tot_out += t_out
        tot_cache += t_cache
        tot_cost += cost
        tot_effective += effective

    req_costs = [v["cost"] for v in by_request.values()]
    n_req = len(by_request)
    return {
        "requests": n_req,
        "per_request": {
            "cost_mean": tot_cost / n_req,
            "cost_p50": _percentile(req_costs, 50),
            "cost_p95": _percentile(req_costs, 95),
            "tokens_in_mean": tot_in / n_req,
            "tokens_out_mean": tot_out / n_req,
        },
        "per_node": {
            node: {
                "cost_total": v["cost"],
                "cost_mean": v["cost"] / v["rows"],
                "rows": int(v["rows"]),
                "tokens_in": int(v["tokens_in"]),
                "tokens_out": int(v["tokens_out"]),
            }
            for node, v in sorted(by_node.items(), key=lambda kv: kv[1]["cost"], reverse=True)
        },
        "totals": {
            "cost": tot_cost,
            "cost_effective": tot_effective,
            "tokens_in": tot_in,
            "tokens_out": tot_out,
            "cache_read_tokens": tot_cache,
            "cache_read_fraction": (tot_cache / tot_in) if tot_in else 0.0,
        },
    }


def per_request_cost(rows: list[dict[str, Any]]) -> dict[str, float]:
    # request_id -> summed cost_dollars_billed. The A/B comparison pairs cost by
    # case via each case's recorded request_id (Phase 12c/12e).
    out: dict[str, float] = {}
    for r in rows:
        rid = str(r["request_id"])
        out[rid] = out.get(rid, 0.0) + float(r.get("cost_dollars_billed") or 0.0)
    return out


def per_request_cost_from_db(
    pool: ConnectionPool, *, request_ids: list[str] | None = None
) -> dict[str, float]:
    sql = "SELECT request_id, cost_dollars_billed FROM trace_events"
    params: list[Any] = []
    if request_ids:
        sql += " WHERE request_id = ANY(%s)"
        params.append(request_ids)
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [d.name for d in cur.description or []]
        rows = [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]
    return per_request_cost(rows)


def cost_report_from_db(
    pool: ConnectionPool, *, request_ids: list[str] | None = None
) -> dict[str, Any]:
    sql = (
        "SELECT request_id, node_name, tokens_in, tokens_out, cache_read_tokens, "
        "cost_dollars_billed, cost_dollars_effective FROM trace_events"
    )
    params: list[Any] = []
    if request_ids:
        sql += " WHERE request_id = ANY(%s)"
        params.append(request_ids)
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [d.name for d in cur.description or []]
        rows = [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]
    return summarize_trace_rows(rows)
