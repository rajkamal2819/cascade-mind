"""
server/playground.py
--------------------
Lightweight Gradio 6.x interactive playground for cascade-mind.

Mounted at /playground by server/app.py via gr.mount_gradio_app().
Theme and CSS are passed to mount_gradio_app() (Gradio 6 API).
Uses ServiceImpactEnvironment in-process -- no HTTP round-trip.
"""
from __future__ import annotations

import re
from typing import List, Optional

import gradio as gr

try:
    from ..graph.graph_builder import (
        SERVICES, SERVICE_METADATA, BASE_EDGES, CANDIDATE_EDGES,
        build_service_graph, get_affected_services,
    )
    from ..env.service_impact_environment import ServiceImpactEnvironment
    from ..domain import DOMAINS
except ImportError:
    from cascade_mind.server.graph.graph_builder import (  # type: ignore
        SERVICES, SERVICE_METADATA, BASE_EDGES, CANDIDATE_EDGES,
        build_service_graph, get_affected_services,
    )
    from cascade_mind.server.env.service_impact_environment import ServiceImpactEnvironment  # type: ignore
    from cascade_mind.server.domain import DOMAINS  # type: ignore

try:
    from ...models import ServiceImpactAction
except ImportError:
    from cascade_mind.models import ServiceImpactAction  # type: ignore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIFFICULTY_SEED = {"easy": 0, "medium": 1, "hard": 2}

ACTION_CHOICES = [
    ("query_dependents   (-1 budget)", "query_dependents"),
    ("query_dependencies (-1 budget)", "query_dependencies"),
    ("query_runbook       (free)",     "query_runbook"),
    ("query_changelog     (free)",     "query_changelog"),
    ("query_monitoring    (free)",     "query_monitoring"),
    ("query_service_health (free)",    "query_service_health"),
    ("query_topology_diff  (free)",    "query_topology_diff"),
    ("submit_hypothesis   (-1 budget)","submit_hypothesis"),
    ("submit  (terminal)",             "submit"),
]

ALL_SERVICES = sorted(SERVICES)

DOMAIN_CHOICES = [
    ("🖥️  SRE / Microservices", "sre"),
    ("🚢  Supply-Chain Disruption", "supply_chain"),
]


def _services_for_domain(domain_key: str) -> list[str]:
    """Return sorted node list for the given domain key."""
    d = DOMAINS.get(domain_key)
    return sorted(d.nodes) if d else ALL_SERVICES


def _empty_state() -> dict:
    return dict(
        env=None, changed="", max_q=15, remaining=15, done=False,
        discovered={},   # {svc: "how found"}
        edges_seen=[],    # [(from_svc, to_svc)] directed edges discovered
        steps=[],         # [{n, action, target, cost, budget_left}]
        scores=[],        # [{seed, diff, domain, score, steps_used, budget_left}]
        difficulty="easy", seed=0,
        domain="sre",           # active domain key
        # v3: world modeling
        belief_state={},        # {service: confidence}
        ig_history=[],          # [float] per step
        world_version=0,
        contradiction_count=0,
    )


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def _budget_html(rem: int, mx: int) -> str:
    pct = int(100 * rem / mx) if mx > 0 else 0
    c = "#059669" if pct > 50 else ("#d97706" if pct > 20 else "#dc2626")
    label = f"{rem} / {mx} queries remaining"
    return (
        '<div style="background:#fff;border-radius:10px;padding:14px 18px;margin-bottom:8px;'
        'box-shadow:0 1px 3px rgba(0,0,0,0.06),0 1px 2px rgba(0,0,0,0.04)">'
        '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">'
        f'<span style="font-size:11px;color:#6b7280;font-weight:700;letter-spacing:0.8px">QUERY BUDGET</span>'
        f'<span style="font-size:13px;color:{c};font-weight:700">{label}</span></div>'
        f'<div style="background:#f3f4f6;border-radius:99px;height:10px;overflow:hidden">'
        f'<div style="width:{pct}%;height:100%;background:{c};border-radius:99px;'
        f'transition:width 0.4s ease"></div>'
        '</div></div>'
    )


def _banner_html(svc: str, diff: str, seed: int) -> str:
    bc = {"easy": "#059669", "medium": "#d97706", "hard": "#dc2626"}.get(diff, "#6b7280")
    meta = SERVICE_METADATA.get(svc, {})
    team = meta.get("team", "?").upper()
    tier = meta.get("tier", "?")
    return (
        '<div style="background:#fff;border-left:4px solid #dc2626;border-radius:10px;'
        'padding:16px 20px;margin-bottom:10px;'
        'box-shadow:0 1px 3px rgba(0,0,0,0.06),0 1px 2px rgba(0,0,0,0.04)">'
        '<div style="display:flex;align-items:center;gap:10px;margin-bottom:12px">'
        '<span style="font-size:20px">&#128680;</span>'
        '<span style="color:#dc2626;font-weight:800;font-size:14px;letter-spacing:0.3px">'
        'PagerDuty &middot; P1 &middot; TRIGGERED</span>'
        f'<span style="background:{bc};color:#fff;font-size:10px;font-weight:700;'
        f'padding:3px 10px;border-radius:99px;margin-left:auto">{diff.upper()}</span></div>'
        f'<div style="background:#fafafa;border:1px solid #e5e7eb;border-radius:8px;'
        f'padding:10px 14px;margin-bottom:12px">'
        f'<div style="color:#9ca3af;font-size:10px;font-weight:700;margin-bottom:3px;'
        f'letter-spacing:0.8px">SERVICE WITH BREAKING CHANGE</div>'
        f'<code style="color:#4f46e5;font-size:18px;font-weight:800">{svc}</code></div>'
        f'<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px">'
        f'<div style="background:#fafafa;border:1px solid #e5e7eb;border-radius:8px;'
        f'padding:8px 10px;text-align:center">'
        f'<div style="font-size:9px;color:#9ca3af;font-weight:700;letter-spacing:0.5px">TEAM</div>'
        f'<div style="font-size:13px;color:#111827;font-weight:700">{team}</div></div>'
        f'<div style="background:#fafafa;border:1px solid #e5e7eb;border-radius:8px;'
        f'padding:8px 10px;text-align:center">'
        f'<div style="font-size:9px;color:#9ca3af;font-weight:700;letter-spacing:0.5px">TIER</div>'
        f'<div style="font-size:13px;color:#111827;font-weight:700">{tier}</div></div>'
        f'<div style="background:#fafafa;border:1px solid #e5e7eb;border-radius:8px;'
        f'padding:8px 10px;text-align:center">'
        f'<div style="font-size:9px;color:#9ca3af;font-weight:700;letter-spacing:0.5px">SEED</div>'
        f'<div style="font-size:13px;color:#111827;font-weight:700">#{seed}</div></div>'
        '</div></div>'
    )


def _score_html(msg: str, reward: float) -> str:
    f_m = re.search(r"F-beta\(.*?\)=([\d.]+)", msg)
    p_m = re.search(r"Precision=([\d.]+)", msg)
    r_m = re.search(r"Recall=([\d.]+)", msg)
    b_m = re.search(r"Brier\)=([\d.]+)", msg)
    fbeta = float(f_m.group(1)) if f_m else reward
    prec  = float(p_m.group(1)) if p_m else 0.0
    rec   = float(r_m.group(1)) if r_m else 0.0
    brier = float(b_m.group(1)) if b_m else None
    pct   = int(fbeta * 100)
    c     = "#059669" if pct >= 70 else ("#d97706" if pct >= 40 else "#dc2626")
    grade = "EXCELLENT" if pct >= 80 else ("GOOD" if pct >= 60 else ("FAIR" if pct >= 40 else "POOR"))
    brier_tile = ""
    if brier is not None:
        bc = "#6366f1" if brier >= 0.7 else ("#f59e0b" if brier >= 0.5 else "#9ca3af")
        brier_tile = (
            f'<div style="background:#fafafa;border:1px solid #e5e7eb;border-radius:8px;padding:10px">'
            f'<div style="font-size:9px;color:#9ca3af;font-weight:700;letter-spacing:0.5px">CALIBRATION</div>'
            f'<div style="font-size:22px;font-weight:800;color:{bc}">{brier:.0%}</div></div>'
        )
    grid_cols = "1fr 1fr 1fr" if brier is not None else "1fr 1fr"
    grid_width = "320px" if brier is not None else "220px"
    return (
        f'<div style="background:#fff;border-top:4px solid {c};border-radius:10px;'
        f'padding:24px 20px;margin-top:10px;text-align:center;'
        f'box-shadow:0 1px 3px rgba(0,0,0,0.06),0 1px 2px rgba(0,0,0,0.04)">'
        f'<div style="font-size:10px;color:#9ca3af;font-weight:700;letter-spacing:1.2px;'
        f'margin-bottom:6px">EPISODE SCORE &middot; F-beta + Calibration</div>'
        f'<div style="font-size:56px;font-weight:900;color:{c};line-height:1">{pct}%</div>'
        f'<div style="font-size:12px;color:{c};font-weight:700;margin:4px 0 14px;'
        f'letter-spacing:0.5px">{grade}</div>'
        f'<div style="background:#f3f4f6;border-radius:99px;height:8px;overflow:hidden;'
        f'margin:0 auto 14px;max-width:180px">'
        f'<div style="width:{pct}%;height:100%;background:{c};border-radius:99px"></div></div>'
        f'<div style="display:grid;grid-template-columns:{grid_cols};gap:10px;max-width:{grid_width};margin:0 auto">'
        f'<div style="background:#fafafa;border:1px solid #e5e7eb;border-radius:8px;padding:10px">'
        f'<div style="font-size:9px;color:#9ca3af;font-weight:700;letter-spacing:0.5px">PRECISION</div>'
        f'<div style="font-size:22px;font-weight:800;color:#111827">{prec:.0%}</div></div>'
        f'<div style="background:#fafafa;border:1px solid #e5e7eb;border-radius:8px;padding:10px">'
        f'<div style="font-size:9px;color:#9ca3af;font-weight:700;letter-spacing:0.5px">RECALL</div>'
        f'<div style="font-size:22px;font-weight:800;color:#111827">{rec:.0%}</div></div>'
        f'{brier_tile}</div></div>'
    )


def _idle_banner() -> str:
    return (
        '<div style="background:#fff;border:1.5px dashed #d1d5db;border-radius:10px;'
        'padding:28px;text-align:center;margin-bottom:10px">'
        '<div style="font-size:32px;margin-bottom:10px">&#127919;</div>'
        '<div style="color:#374151;font-size:15px;font-weight:700">No active episode</div>'
        '<div style="color:#9ca3af;font-size:12px;margin-top:6px">'
        'Choose a difficulty and click <strong style="color:#4f46e5">New Episode</strong> '
        'to start investigating'
        '</div></div>'
    )


_ALL_SERVICES_SET = set(ALL_SERVICES)


def _extract_services(text: str) -> set:
    """Find all known service names mentioned in a text blob."""
    found = set()
    for svc in _ALL_SERVICES_SET:
        if svc in text:
            found.add(svc)
    return found


def _discovered_html(discovered: dict) -> str:
    """Render the discovered-services tracker panel."""
    if not discovered:
        return (
            '<div style="background:#fff;border:1px dashed #d1d5db;border-radius:8px;'
            'padding:16px;text-align:center;color:#9ca3af;font-size:12px">'
            'No services discovered yet. Start querying to explore the graph.'
            '</div>'
        )
    rows = ""
    for svc in sorted(discovered):
        how = discovered[svc]
        rows += (
            f'<div style="display:flex;align-items:center;justify-content:space-between;'
            f'padding:6px 10px;border-bottom:1px solid #f3f4f6">'
            f'<code style="color:#4f46e5;font-weight:600;font-size:12px">{svc}</code>'
            f'<span style="color:#9ca3af;font-size:10px;font-weight:600">{how}</span>'
            f'</div>'
        )
    return (
        f'<div style="background:#fff;border-radius:8px;border:1px solid #e5e7eb;'
        f'box-shadow:0 1px 2px rgba(0,0,0,0.04);overflow:hidden">'
        f'<div style="background:#fafafa;padding:8px 12px;border-bottom:1px solid #e5e7eb;'
        f'display:flex;justify-content:space-between;align-items:center">'
        f'<span style="font-size:11px;color:#6b7280;font-weight:700;letter-spacing:0.5px">'
        f'DISCOVERED SERVICES</span>'
        f'<span style="background:#eef2ff;color:#4f46e5;font-size:10px;font-weight:700;'
        f'padding:2px 8px;border-radius:99px">{len(discovered)} / {len(ALL_SERVICES)}</span>'
        f'</div>{rows}</div>'
    )


def _timeline_html(steps: list) -> str:
    """Render a compact step timeline."""
    if not steps:
        return ""
    icons = {
        "query_dependents": "🔵", "query_dependencies": "🔵",
        "query_runbook": "📗", "query_changelog": "📗",
        "query_monitoring": "📗", "query_service_health": "📗",
        "query_topology_diff": "📗",
        "submit_hypothesis": "🟡", "submit": "🔴",
    }
    rows = ""
    for s in steps:
        icon = icons.get(s["action"], "⚪")
        cost_label = f'-{s["cost"]}' if s["cost"] > 0 else "free"
        cost_color = "#dc2626" if s["cost"] > 0 else "#059669"
        target = f'<code style="color:#4f46e5;font-size:10px">{s["target"]}</code>' if s.get("target") else ""
        rows += (
            f'<div style="display:flex;align-items:center;gap:8px;padding:5px 10px;'
            f'border-bottom:1px solid #f3f4f6;font-size:12px">'
            f'<span style="color:#9ca3af;font-weight:700;min-width:20px">#{s["n"]}</span>'
            f'<span>{icon}</span>'
            f'<span style="color:#374151;font-weight:600;flex:1">{s["action"]}</span>'
            f'{target}'
            f'<span style="color:{cost_color};font-size:10px;font-weight:700">{cost_label}</span>'
            f'<span style="color:#9ca3af;font-size:10px">{s["budget_left"]} left</span>'
            f'</div>'
        )
    return (
        f'<div style="background:#fff;border-radius:8px;border:1px solid #e5e7eb;'
        f'box-shadow:0 1px 2px rgba(0,0,0,0.04);overflow:hidden;max-height:200px;'
        f'overflow-y:auto">'
        f'<div style="background:#fafafa;padding:8px 12px;border-bottom:1px solid #e5e7eb;'
        f'position:sticky;top:0">'
        f'<span style="font-size:11px;color:#6b7280;font-weight:700;letter-spacing:0.5px">'
        f'ACTION TIMELINE &middot; {len(steps)} steps</span></div>'
        f'{rows}</div>'
    )


def _diff_table_html(predicted: list, correct: list) -> str:
    """Render a visual diff of predicted vs ground-truth affected services."""
    predicted_set = set(predicted)
    correct_set = set(correct)
    all_relevant = sorted(predicted_set | correct_set)
    if not all_relevant:
        return ""
    rows = ""
    for svc in all_relevant:
        in_pred = svc in predicted_set
        in_corr = svc in correct_set
        if in_pred and in_corr:
            icon, label, bg, color = "✔", "Correct", "#f0fdf4", "#059669"
        elif in_corr and not in_pred:
            icon, label, bg, color = "⚠", "Missed", "#fef9c3", "#d97706"
        else:
            icon, label, bg, color = "✖", "Wrong", "#fef2f2", "#dc2626"
        pred_cell = '<span style="color:#059669">✅</span>' if in_pred else '<span style="color:#d1d5db">—</span>'
        corr_cell = '<span style="color:#059669">✅</span>' if in_corr else '<span style="color:#d1d5db">—</span>'
        rows += (
            f'<tr style="background:{bg}">'
            f'<td style="padding:6px 10px;border-bottom:1px solid #e5e7eb">'
            f'<code style="font-size:11px;color:#374151">{svc}</code></td>'
            f'<td style="padding:6px 10px;border-bottom:1px solid #e5e7eb;text-align:center">{pred_cell}</td>'
            f'<td style="padding:6px 10px;border-bottom:1px solid #e5e7eb;text-align:center">{corr_cell}</td>'
            f'<td style="padding:6px 10px;border-bottom:1px solid #e5e7eb;text-align:center">'
            f'<span style="color:{color};font-weight:700;font-size:12px">{icon} {label}</span></td>'
            f'</tr>'
        )
    tp = len(predicted_set & correct_set)
    fp = len(predicted_set - correct_set)
    fn = len(correct_set - predicted_set)
    return (
        f'<div style="background:#fff;border-radius:8px;border:1px solid #e5e7eb;'
        f'box-shadow:0 1px 2px rgba(0,0,0,0.04);overflow:hidden;margin-top:10px">'
        f'<div style="background:#fafafa;padding:8px 12px;border-bottom:1px solid #e5e7eb;'
        f'display:flex;justify-content:space-between;align-items:center">'
        f'<span style="font-size:11px;color:#6b7280;font-weight:700;letter-spacing:0.5px">'
        f'SUBMISSION DIFF</span>'
        f'<span style="font-size:10px;color:#6b7280">'
        f'<span style="color:#059669;font-weight:700">{tp} correct</span> &middot; '
        f'<span style="color:#d97706;font-weight:700">{fn} missed</span> &middot; '
        f'<span style="color:#dc2626;font-weight:700">{fp} wrong</span></span>'
        f'</div>'
        f'<table style="width:100%;border-collapse:collapse;font-size:12px">'
        f'<tr style="background:#fafafa">'
        f'<th style="padding:6px 10px;text-align:left;color:#9ca3af;font-size:10px;'
        f'font-weight:700;border-bottom:1px solid #e5e7eb">SERVICE</th>'
        f'<th style="padding:6px 10px;text-align:center;color:#9ca3af;font-size:10px;'
        f'font-weight:700;border-bottom:1px solid #e5e7eb">YOURS</th>'
        f'<th style="padding:6px 10px;text-align:center;color:#9ca3af;font-size:10px;'
        f'font-weight:700;border-bottom:1px solid #e5e7eb">TRUTH</th>'
        f'<th style="padding:6px 10px;text-align:center;color:#9ca3af;font-size:10px;'
        f'font-weight:700;border-bottom:1px solid #e5e7eb">RESULT</th>'
        f'</tr>{rows}</table></div>'
    )


def _scoreboard_html(scores: list) -> str:
    """Render multi-episode scoreboard."""
    if not scores:
        return (
            '<div style="color:#9ca3af;font-size:12px;text-align:center;padding:12px">'
            'No episodes completed yet.</div>'
        )
    avg = sum(s["score"] for s in scores) / len(scores)
    avg_c = "#059669" if avg >= 0.7 else ("#d97706" if avg >= 0.4 else "#dc2626")
    rows = ""
    for i, s in enumerate(reversed(scores), 1):
        sc = int(s["score"] * 100)
        sc_c = "#059669" if sc >= 70 else ("#d97706" if sc >= 40 else "#dc2626")
        diff_c = {"easy": "#059669", "medium": "#d97706", "hard": "#dc2626"}.get(s["diff"], "#6b7280")
        rows += (
            f'<tr><td style="padding:5px 10px;border-bottom:1px solid #f3f4f6;'
            f'color:#9ca3af;font-size:11px">#{len(scores)-i+1}</td>'
            f'<td style="padding:5px 10px;border-bottom:1px solid #f3f4f6">'
            f'<span style="color:{diff_c};font-weight:600;font-size:11px">{s["diff"]}</span></td>'
            f'<td style="padding:5px 10px;border-bottom:1px solid #f3f4f6;color:#6b7280;font-size:11px">'
            f'#{s["seed"]}</td>'
            f'<td style="padding:5px 10px;border-bottom:1px solid #f3f4f6;text-align:center">'
            f'<span style="color:{sc_c};font-weight:800;font-size:13px">{sc}%</span></td>'
            f'<td style="padding:5px 10px;border-bottom:1px solid #f3f4f6;color:#6b7280;font-size:11px;'
            f'text-align:center">{s["steps_used"]}</td>'
            f'<td style="padding:5px 10px;border-bottom:1px solid #f3f4f6;color:#6b7280;font-size:11px;'
            f'text-align:center">{s["budget_left"]}</td></tr>'
        )
    return (
        f'<div style="background:#fff;border-radius:8px;border:1px solid #e5e7eb;'
        f'box-shadow:0 1px 2px rgba(0,0,0,0.04);overflow:hidden">'
        f'<div style="background:#fafafa;padding:8px 12px;border-bottom:1px solid #e5e7eb;'
        f'display:flex;justify-content:space-between;align-items:center">'
        f'<span style="font-size:11px;color:#6b7280;font-weight:700;letter-spacing:0.5px">'
        f'SCOREBOARD &middot; {len(scores)} episodes</span>'
        f'<span style="font-size:12px;color:{avg_c};font-weight:800">'
        f'Avg: {int(avg*100)}%</span></div>'
        f'<table style="width:100%;border-collapse:collapse">'
        f'<tr style="background:#fafafa">'
        f'<th style="padding:5px 10px;text-align:left;color:#9ca3af;font-size:9px;font-weight:700">EP</th>'
        f'<th style="padding:5px 10px;text-align:left;color:#9ca3af;font-size:9px;font-weight:700">DIFF</th>'
        f'<th style="padding:5px 10px;text-align:left;color:#9ca3af;font-size:9px;font-weight:700">SEED</th>'
        f'<th style="padding:5px 10px;text-align:center;color:#9ca3af;font-size:9px;font-weight:700">SCORE</th>'
        f'<th style="padding:5px 10px;text-align:center;color:#9ca3af;font-size:9px;font-weight:700">STEPS</th>'
        f'<th style="padding:5px 10px;text-align:center;color:#9ca3af;font-size:9px;font-weight:700">BUDGET</th>'
        f'</tr>{rows}</table></div>'
    )


# ---------------------------------------------------------------------------
# Dynamic vis.js knowledge graph
# ---------------------------------------------------------------------------

def _extract_edges(action_type: str, target_service: str, response_text: str) -> list:
    """Infer directed edges from a query response.

    query_dependents(X) returns predecessors of X → edges: each → X
    query_dependencies(X) returns successors of X → edges: X → each
    Other actions: no edges inferred (nodes only).
    """
    if action_type not in ("query_dependents", "query_dependencies") or not target_service:
        return []
    mentioned = _extract_services(response_text) - {target_service}
    edges = []
    for svc in mentioned:
        if action_type == "query_dependents":
            edges.append((svc, target_service))  # svc depends on target
        else:
            edges.append((target_service, svc))  # target depends on svc
    return edges


def _vis_js_graph_html(
    discovered: dict,
    edges_seen: list,
    changed: str,
    done: bool = False,
) -> str:
    """Render an interactive vis.js graph of discovered services and edges."""
    if not discovered:
        return (
            '<div style="background:#fff;border:1px dashed #d1d5db;border-radius:8px;'
            'padding:40px;text-align:center;color:#9ca3af;font-size:13px">'
            '<div style="font-size:28px;margin-bottom:8px">&#128566;</div>'
            'Start an episode to see the knowledge graph grow as you investigate.'
            '</div>'
        )

    tier_colors = {
        1: "#818cf8",  # indigo — gateway
        2: "#60a5fa",  # blue — business
        3: "#a78bfa",  # violet — support
        4: "#22d3ee",  # cyan — data
        5: "#94a3b8",  # slate — infra
    }

    import json as _json

    nodes = []
    for svc in sorted(discovered):
        meta = SERVICE_METADATA.get(svc, {})
        tier = meta.get("tier", 3)
        team = meta.get("team", "?")
        color = "#ef4444" if svc == changed else tier_colors.get(tier, "#6b7280")
        border_width = 3 if svc == changed else 2
        font_color = "#fff" if svc == changed else "#e2e8f0"
        bg = "#dc2626" if svc == changed else (color + "33")
        nodes.append({
            "id": svc,
            "label": svc.replace("_", "\n"),
            "color": {
                "background": bg,
                "border": color,
                "highlight": {"background": color, "border": "#111827"},
            },
            "borderWidth": border_width,
            "font": {"size": 11, "color": font_color if svc == changed else "#374151",
                     "face": "Inter, system-ui, sans-serif"},
            "shape": "box",
            "margin": 8,
            "title": f"{svc}\nTeam: {team} | Tier: {tier}\nFound via: {discovered[svc]}",
        })

    disc_set = set(discovered)
    edges = []
    seen_pairs = set()
    for (a, b) in edges_seen:
        if a in disc_set and b in disc_set and (a, b) not in seen_pairs:
            seen_pairs.add((a, b))
            edges.append({"from": a, "to": b, "arrows": "to",
                          "color": {"color": "#475569", "highlight": "#818cf8"},
                          "width": 1.5, "smooth": {"type": "curvedCW", "roundness": 0.15}})

    nodes_json = _json.dumps(nodes)
    edges_json = _json.dumps(edges)

    # Use iframe srcdoc so vis.js scripts execute (Gradio sanitizes <script> in gr.HTML)
    iframe_html = f"""
<!DOCTYPE html>
<html><head>
<script src="https://unpkg.com/vis-network@9.1.6/standalone/umd/vis-network.min.js"></script>
<style>
  *{{margin:0;padding:0;box-sizing:border-box}}
  body{{font-family:Inter,system-ui,sans-serif;background:#0f172a;color:#e2e8f0}}
  #header{{background:rgba(15,23,42,0.95);padding:10px 14px;border-bottom:1px solid #1e293b;
           display:flex;justify-content:space-between;align-items:center;backdrop-filter:blur(8px)}}
  #header .title{{font-size:11px;color:#94a3b8;font-weight:700;letter-spacing:1px}}
  #header .meta{{font-size:10px;color:#64748b}}
  #legend{{display:flex;gap:10px;padding:6px 14px;background:#1e293b;border-bottom:1px solid #334155;flex-wrap:wrap}}
  .leg{{display:flex;align-items:center;gap:4px;font-size:9px;color:#94a3b8}}
  .leg-dot{{width:8px;height:8px;border-radius:2px}}
  #graph{{width:100%;height:calc(100vh - 68px)}}
</style>
</head><body>
<div id="header">
  <span class="title">LIVE KNOWLEDGE GRAPH</span>
  <span class="meta">{len(discovered)} nodes &middot; {len(seen_pairs)} edges &middot;
    <span style="color:#ef4444;font-weight:700">&#9679;</span> incident source</span>
</div>
<div id="legend">
  <div class="leg"><div class="leg-dot" style="background:#ef4444"></div>Incident</div>
  <div class="leg"><div class="leg-dot" style="background:#818cf8"></div>T1 Gateway</div>
  <div class="leg"><div class="leg-dot" style="background:#60a5fa"></div>T2 Business</div>
  <div class="leg"><div class="leg-dot" style="background:#a78bfa"></div>T3 Support</div>
  <div class="leg"><div class="leg-dot" style="background:#22d3ee"></div>T4 Data</div>
  <div class="leg"><div class="leg-dot" style="background:#94a3b8"></div>T5 Infra</div>
</div>
<div id="graph"></div>
<script>
var nodes = new vis.DataSet({nodes_json});
var edges = new vis.DataSet({edges_json});
var net = new vis.Network(document.getElementById('graph'), {{nodes:nodes,edges:edges}}, {{
  layout:{{hierarchical:{{enabled:true,direction:'UD',sortMethod:'hubsize',
           levelSeparation:90,nodeSpacing:150,treeSpacing:200}}}},
  physics:{{enabled:false}},
  interaction:{{hover:true,tooltipDelay:80,zoomView:true,dragView:true,
                navigationButtons:true,keyboard:true}},
  edges:{{font:{{size:9}},color:{{inherit:false}}}},
}});
net.once('stabilized',function(){{net.fit({{animation:true}})}});
</script>
</body></html>
"""
    escaped = iframe_html.replace('&', '&amp;').replace('"', '&quot;')
    return (
        f'<iframe srcdoc="{escaped}" '
        f'style="width:100%;height:540px;border:1px solid #1e293b;border-radius:10px;'
        f'box-shadow:0 4px 12px rgba(0,0,0,0.15)" frameborder="0"></iframe>'
    )


def _replay_html(
    steps: list,
    discovered: dict,
    edges_seen: list,
    changed: str,
    score: float = 0.0,
) -> str:
    """Build an animated trajectory replay that shows how the graph was built step-by-step."""
    if not steps:
        return (
            '<div style="background:#0f172a;border-radius:10px;padding:60px 20px;'
            'text-align:center;color:#64748b;font-size:13px;border:1px solid #1e293b">'
            '<div style="font-size:36px;margin-bottom:10px">\U0001f3ac</div>'
            'Complete an episode to see the trajectory replay.</div>'
        )

    import json as _json

    tier_colors = {
        1: "#818cf8", 2: "#60a5fa", 3: "#a78bfa", 4: "#22d3ee", 5: "#94a3b8",
    }

    # Build per-step snapshots: which nodes and edges exist at each step
    disc_at_step = {}  # step_n -> cumulative discovered dict
    edges_at_step = {}  # step_n -> cumulative edges list
    cum_disc = {changed: "incident alert"}
    cum_edges = []
    # step 0 = initial
    disc_at_step[0] = dict(cum_disc)
    edges_at_step[0] = list(cum_edges)

    for s in steps:
        # Figure out what was discovered at this step
        action = s["action"]
        target = s.get("target", "")
        # Add target service itself
        if target and target in _ALL_SERVICES_SET:
            if target not in cum_disc:
                cum_disc[target] = action
        # Add edges for this step (from edges_seen that involve target)
        if action in ("query_dependents", "query_dependencies") and target:
            for (a, b) in edges_seen:
                if (action == "query_dependents" and b == target) or \
                   (action == "query_dependencies" and a == target):
                    if (a, b) not in [(e[0], e[1]) for e in cum_edges]:
                        cum_edges.append((a, b))
                    # Also mark the other end as discovered
                    other = a if b == target else b
                    if other not in cum_disc:
                        cum_disc[other] = action
        disc_at_step[s["n"]] = dict(cum_disc)
        edges_at_step[s["n"]] = list(cum_edges)

    # Build all unique nodes/edges across all steps for vis.js
    all_nodes = []
    for svc in sorted(discovered):
        meta = SERVICE_METADATA.get(svc, {})
        tier = meta.get("tier", 3)
        team = meta.get("team", "?")
        color = "#ef4444" if svc == changed else tier_colors.get(tier, "#6b7280")
        bg = "#dc2626" if svc == changed else (color + "33")
        all_nodes.append({
            "id": svc,
            "label": svc.replace("_", "\n"),
            "color": {"background": bg, "border": color,
                      "highlight": {"background": color, "border": "#fff"}},
            "borderWidth": 3 if svc == changed else 2,
            "font": {"size": 11,
                     "color": "#fff" if svc == changed else "#e2e8f0",
                     "face": "Inter, system-ui, sans-serif"},
            "shape": "box", "margin": 8,
            "hidden": True,  # start hidden
            "title": f"{svc} | {team} | Tier {tier}",
        })

    all_edges = []
    seen_set = set()
    for (a, b) in edges_seen:
        if (a, b) not in seen_set and a in discovered and b in discovered:
            seen_set.add((a, b))
            all_edges.append({"from": a, "to": b, "arrows": "to", "id": f"{a}__{b}",
                              "color": {"color": "#475569", "highlight": "#818cf8"},
                              "width": 1.5, "hidden": True,
                              "smooth": {"type": "curvedCW", "roundness": 0.15}})

    # Build step data for JS animation
    step_data = []
    step_keys = sorted(disc_at_step.keys())
    for k in step_keys:
        node_ids = list(disc_at_step[k].keys())
        edge_ids = [f"{a}__{b}" for (a, b) in edges_at_step[k]]
        label = "Incident Alert" if k == 0 else f"Step {k}: {steps[k-1]['action']}"
        target = "" if k == 0 else (steps[k-1].get('target', '') or '')
        if target:
            label += f" → {target}"
        step_data.append({"nodes": node_ids, "edges": edge_ids, "label": label, "step": k})

    nodes_json = _json.dumps(all_nodes)
    edges_json = _json.dumps(all_edges)
    steps_json = _json.dumps(step_data)
    score_pct = int(score * 100)
    score_color = "#10b981" if score_pct >= 70 else ("#f59e0b" if score_pct >= 40 else "#ef4444")

    iframe_html = f"""
<!DOCTYPE html>
<html><head>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap" rel="stylesheet">
<script src="https://unpkg.com/vis-network@9.1.6/standalone/umd/vis-network.min.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Inter',system-ui,sans-serif;background:#0f172a;color:#e2e8f0;overflow:hidden}}
#top-bar{{background:linear-gradient(135deg,#1e293b,#0f172a);padding:12px 16px;
          border-bottom:1px solid #334155;display:flex;align-items:center;justify-content:space-between}}
#top-bar h2{{font-size:13px;color:#e2e8f0;font-weight:800;letter-spacing:0.5px}}
#top-bar .score{{font-size:22px;font-weight:900;color:{score_color}}}
#controls{{background:#1e293b;padding:8px 16px;border-bottom:1px solid #334155;
           display:flex;align-items:center;gap:12px}}
#controls button{{background:#4f46e5;color:#fff;border:none;padding:6px 16px;border-radius:6px;
                  font-size:11px;font-weight:700;cursor:pointer;font-family:inherit}}
#controls button:hover{{background:#4338ca}}
#controls button.secondary{{background:#334155}}
#controls button.secondary:hover{{background:#475569}}
#controls .speed{{color:#94a3b8;font-size:10px;font-weight:600}}
#step-label{{flex:1;text-align:center;color:#94a3b8;font-size:12px;font-weight:600;
             background:#0f172a;padding:6px 12px;border-radius:6px;border:1px solid #334155}}
#step-label .highlight{{color:#818cf8;font-weight:700}}
#progress{{height:3px;background:#1e293b;position:relative}}
#progress-bar{{height:100%;background:linear-gradient(90deg,#4f46e5,#7c3aed);width:0%;transition:width 0.3s}}
#graph{{width:100%;height:calc(100vh - 98px)}}
</style>
</head><body>
<div id="top-bar">
  <h2>\U0001f3ac TRAJECTORY REPLAY</h2>
  <div class="score">{score_pct}%</div>
</div>
<div id="controls">
  <button id="playBtn" onclick="togglePlay()">&#9654; Play</button>
  <button class="secondary" onclick="stepBack()">&laquo; Back</button>
  <button class="secondary" onclick="stepFwd()">Next &raquo;</button>
  <button class="secondary" onclick="resetReplay()">&#8634; Reset</button>
  <span class="speed">Speed:</span>
  <button class="secondary" onclick="setSpeed(2000)">Slow</button>
  <button class="secondary" onclick="setSpeed(1000)">Normal</button>
  <button class="secondary" onclick="setSpeed(400)">Fast</button>
  <div id="step-label">Press Play to begin</div>
</div>
<div id="progress"><div id="progress-bar"></div></div>
<div id="graph"></div>
<script>
var allSteps={steps_json};
var nodesDS=new vis.DataSet({nodes_json});
var edgesDS=new vis.DataSet({edges_json});
var net=new vis.Network(document.getElementById('graph'),{{nodes:nodesDS,edges:edgesDS}},{{
  layout:{{hierarchical:{{enabled:true,direction:'UD',sortMethod:'hubsize',
           levelSeparation:90,nodeSpacing:150,treeSpacing:200}}}},
  physics:{{enabled:false}},
  interaction:{{hover:true,tooltipDelay:80,zoomView:true,dragView:true,
                navigationButtons:true,keyboard:true}},
  edges:{{color:{{inherit:false}}}},
}});
var curStep=-1,playing=false,timer=null,speed=1000;
function showStep(idx){{
  if(idx<0||idx>=allSteps.length) return;
  curStep=idx;
  var s=allSteps[idx];
  // Show nodes up to this step
  var visNodes=new Set();
  var visEdges=new Set();
  for(var i=0;i<=idx;i++){{allSteps[i].nodes.forEach(function(n){{visNodes.add(n)}});
    allSteps[i].edges.forEach(function(e){{visEdges.add(e)}})}}
  nodesDS.forEach(function(n){{nodesDS.update({{id:n.id,hidden:!visNodes.has(n.id)}})}});
  edgesDS.forEach(function(e){{edgesDS.update({{id:e.id,hidden:!visEdges.has(e.id)}})}});
  // Highlight newly added nodes with a pulse
  s.nodes.forEach(function(nid){{
    if(idx>0&&!allSteps[idx-1].nodes.includes(nid)){{
      nodesDS.update({{id:nid,borderWidth:5}});
      setTimeout(function(){{nodesDS.update({{id:nid,borderWidth:2}})}},600);
    }}
  }});
  document.getElementById('step-label').innerHTML=
    '<span class="highlight">['+(idx+1)+'/'+allSteps.length+']</span> '+s.label;
  document.getElementById('progress-bar').style.width=((idx+1)/allSteps.length*100)+'%';
  if(idx===0) setTimeout(function(){{net.fit({{animation:true}})}},200);
}}
function stepFwd(){{if(curStep<allSteps.length-1)showStep(curStep+1);
  else{{playing=false;document.getElementById('playBtn').innerHTML='&#9654; Play'}};}}
function stepBack(){{if(curStep>0)showStep(curStep-1)}}
function togglePlay(){{
  playing=!playing;
  document.getElementById('playBtn').innerHTML=playing?'&#9646;&#9646; Pause':'&#9654; Play';
  if(playing){{if(curStep>=allSteps.length-1)curStep=-1;runPlay()}}
  else clearTimeout(timer);
}}
function runPlay(){{if(!playing)return;stepFwd();
  if(curStep<allSteps.length-1)timer=setTimeout(runPlay,speed);
  else{{playing=false;document.getElementById('playBtn').innerHTML='&#9654; Play'}}}}
function resetReplay(){{playing=false;clearTimeout(timer);curStep=-1;
  document.getElementById('playBtn').innerHTML='&#9654; Play';
  nodesDS.forEach(function(n){{nodesDS.update({{id:n.id,hidden:true}})}});
  edgesDS.forEach(function(e){{edgesDS.update({{id:e.id,hidden:true}})}});
  document.getElementById('step-label').innerHTML='Press Play to begin';
  document.getElementById('progress-bar').style.width='0%';}}
function setSpeed(s){{speed=s}}
</script>
</body></html>
"""
    escaped = iframe_html.replace('&', '&amp;').replace('"', '&quot;')
    return (
        f'<iframe srcdoc="{escaped}" '
        f'style="width:100%;height:560px;border:1px solid #1e293b;border-radius:10px;'
        f'box-shadow:0 4px 16px rgba(0,0,0,0.2)" frameborder="0"></iframe>'
    )


def build_ground_truth_html(seed: int = 0, difficulty: str = "easy") -> str:
    """Build a standalone HTML page showing the full ground-truth graph."""
    G = build_service_graph(seed=seed)
    try:
        from ..graph.graph_builder import get_scenario  # type: ignore
    except ImportError:
        from cascade_mind.server.graph.graph_builder import get_scenario  # type: ignore
    scenario = get_scenario(G, seed)
    changed = scenario["changed_service"]
    affected = get_affected_services(G, changed)

    import json as _json

    tier_colors = {
        1: "#6366f1", 2: "#3b82f6", 3: "#8b5cf6", 4: "#06b6d4", 5: "#64748b",
    }

    nodes = []
    for svc in sorted(G.nodes()):
        meta = SERVICE_METADATA.get(svc, {})
        tier = meta.get("tier", 3)
        team = meta.get("team", "?")
        if svc == changed:
            bg, border, fc = "#dc2626", "#991b1b", "#fff"
            bw = 3
        elif svc in affected:
            bg, border, fc = "#fef3c7", "#f59e0b", "#92400e"
            bw = 2
        else:
            c = tier_colors.get(tier, "#6b7280")
            bg, border, fc = c + "18", c, "#374151"
            bw = 1
        nodes.append({
            "id": svc,
            "label": svc.replace("_", "\n"),
            "color": {"background": bg, "border": border,
                      "highlight": {"background": border, "border": "#111"}},
            "borderWidth": bw,
            "font": {"size": 12, "color": fc, "face": "Inter, system-ui, sans-serif"},
            "shape": "box", "margin": 8,
            "title": f"{svc}  |  Team: {team}  |  Tier: {tier}",
        })

    edges = []
    for (a, b) in G.edges():
        in_blast = a in affected or a == changed
        edges.append({
            "from": a, "to": b, "arrows": "to",
            "color": {"color": "#f59e0b" if in_blast else "#cbd5e1"},
            "width": 2 if in_blast else 1,
            "smooth": {"type": "curvedCW", "roundness": 0.12},
        })

    nodes_json = _json.dumps(nodes)
    edges_json = _json.dumps(edges)
    n_affected = len(affected)
    all_affected_list = _json.dumps(sorted(affected))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>cascade-mind — Ground Truth Graph (seed {seed})</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap" rel="stylesheet">
<script src="https://unpkg.com/vis-network@9.1.6/standalone/umd/vis-network.min.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:'Inter',system-ui,sans-serif; background:#f9fafb; color:#111827; }}
.header {{ background:#fff; border-bottom:1px solid #e5e7eb; padding:16px 24px;
           display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:12px; }}
.header h1 {{ font-size:18px; font-weight:900; }}
.header h1 span {{ color:#4f46e5; }}
.pill {{ display:inline-block; padding:3px 10px; border-radius:99px; font-size:10px; font-weight:700; }}
.legend {{ display:flex; gap:16px; align-items:center; flex-wrap:wrap; }}
.legend-item {{ display:flex; align-items:center; gap:5px; font-size:11px; color:#6b7280; }}
.legend-dot {{ width:12px; height:12px; border-radius:3px; }}
.info-bar {{ background:#fff; border-bottom:1px solid #e5e7eb; padding:10px 24px;
             display:flex; gap:20px; align-items:center; font-size:12px; flex-wrap:wrap; }}
.info-bar .tag {{ background:#f3f4f6; padding:4px 10px; border-radius:6px; font-weight:600; }}
#search {{ padding:6px 12px; border:1px solid #d1d5db; border-radius:6px; font-size:12px;
           width:220px; font-family:inherit; }}
#graph {{ width:100%; height:calc(100vh - 130px); }}
.affected-list {{ position:fixed; bottom:16px; right:16px; background:#fff;
                  border:1px solid #e5e7eb; border-radius:10px; padding:14px 18px;
                  box-shadow:0 4px 12px rgba(0,0,0,0.08); max-height:40vh;
                  overflow-y:auto; width:260px; font-size:11px; z-index:100; }}
.affected-list h3 {{ font-size:11px; color:#6b7280; font-weight:700; letter-spacing:0.5px;
                     margin-bottom:8px; }}
.affected-list code {{ display:block; padding:3px 0; color:#92400e; font-weight:600; font-size:11px; }}
</style>
</head>
<body>
<div class="header">
  <h1>&#129504; <span>cascade-mind</span> &mdash; Ground Truth Graph</h1>
  <div class="legend">
    <div class="legend-item"><div class="legend-dot" style="background:#dc2626"></div>Incident source</div>
    <div class="legend-item"><div class="legend-dot" style="background:#f59e0b"></div>Affected (blast radius)</div>
    <div class="legend-item"><div class="legend-dot" style="background:#6366f1"></div>Tier 1 Gateway</div>
    <div class="legend-item"><div class="legend-dot" style="background:#3b82f6"></div>Tier 2 Business</div>
    <div class="legend-item"><div class="legend-dot" style="background:#8b5cf6"></div>Tier 3 Support</div>
    <div class="legend-item"><div class="legend-dot" style="background:#06b6d4"></div>Tier 4 Data/ML</div>
    <div class="legend-item"><div class="legend-dot" style="background:#64748b"></div>Tier 5 Infra</div>
  </div>
</div>
<div class="info-bar">
  <span>Seed: <span class="tag">#{seed}</span></span>
  <span>Difficulty: <span class="tag" style="color:{'#059669' if difficulty=='easy' else '#d97706' if difficulty=='medium' else '#dc2626'}">{difficulty.upper()}</span></span>
  <span>Changed: <span class="tag" style="color:#dc2626">{changed}</span></span>
  <span>Blast radius: <span class="tag" style="color:#f59e0b">{n_affected} services</span></span>
  <span style="margin-left:auto">Search: <input id="search" placeholder="Filter service..." oninput="filterNodes(this.value)"/></span>
</div>
<div id="graph"></div>
<div class="affected-list">
  <h3>AFFECTED SERVICES ({n_affected})</h3>
  {''.join(f'<code>{s}</code>' for s in sorted(affected))}
</div>
<script>
var allNodes = {nodes_json};
var allEdges = {edges_json};
var nodes = new vis.DataSet(allNodes);
var edges = new vis.DataSet(allEdges);
var container = document.getElementById('graph');
var network = new vis.Network(container, {{nodes: nodes, edges: edges}}, {{
  layout: {{ hierarchical: {{ enabled: true, direction: 'UD', sortMethod: 'hubsize',
             levelSeparation: 90, nodeSpacing: 140 }} }},
  physics: {{ enabled: false }},
  interaction: {{ hover: true, tooltipDelay: 100, zoomView: true, dragView: true,
                  navigationButtons: true, keyboard: true }},
}});
function filterNodes(q) {{
  q = q.toLowerCase();
  allNodes.forEach(function(n) {{
    var match = !q || n.id.includes(q);
    nodes.update({{id: n.id, hidden: !match}});
  }});
}}
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def _env_state_dict(env) -> dict:
    """Serialize env._state to a plain dict for gr.JSON display."""
    try:
        raw = vars(env._state).copy()
        # Remove non-serialisable internal sets/objects
        return {k: (list(v) if isinstance(v, (set, frozenset)) else v)
                for k, v in raw.items()}
    except Exception:
        return {}


def reset_episode(difficulty: str, custom_seed: int, domain_key: str, _state: dict):
    seed = DIFFICULTY_SEED.get(difficulty, 0) if int(custom_seed) < 0 else int(custom_seed)
    domain_config = DOMAINS.get(domain_key)   # None → SRE (default env behaviour)
    env = ServiceImpactEnvironment(domain_config=domain_config)
    obs = env.reset(seed=seed)
    svc, mq = env._changed_service, env._max_queries
    all_svcs = _services_for_domain(domain_key)
    # Preserve scores across episodes
    prev_scores = _state.get("scores", []) if _state else []
    init_discovered = {svc: "incident alert"}
    st = dict(
        env=env, changed=svc, max_q=mq, remaining=mq, done=False,
        discovered=init_discovered,
        edges_seen=[],
        steps=[], scores=prev_scores,
        difficulty=difficulty, seed=seed,
        domain=domain_key,
        # v3: world modeling
        belief_state={},
        ig_history=[],
        world_version=0,
        contradiction_count=0,
    )
    chat = [{"role": "assistant", "content": f"**Incident triggered!**\n\n{obs.message}"}]
    domain_display = dict(DOMAIN_CHOICES).get(domain_key, domain_key)
    return (
        st,
        chat,
        _budget_html(mq, mq),
        _banner_html(svc, difficulty, seed),
        "",                                      # clear score card
        gr.update(value=svc, choices=all_svcs),  # update + pre-select service dropdown
        gr.update(interactive=True),             # enable Execute button
        gr.update(value=[], choices=all_svcs),   # update affected checkboxes
        gr.update(value="query_dependents"),     # reset action radio
        _env_state_dict(env),                    # env state panel
        _discovered_html(init_discovered),       # discovered panel
        "",                                      # clear timeline
        _scoreboard_html(prev_scores),           # refresh scoreboard
        _vis_js_graph_html(init_discovered, [], svc),   # live graph
        _replay_html([], {}, [], svc),           # empty replay
        # Ground truth button (enhanced)
        f'<div style="text-align:center;padding:8px 0">'
        f'<a href="/graph/ground-truth?seed={seed}&difficulty={difficulty}" '
        f'target="_blank" style="background:linear-gradient(135deg,#4f46e5,#7c3aed);'
        f'color:#fff;font-weight:700;font-size:13px;text-decoration:none;'
        f'padding:10px 24px;border-radius:10px;display:inline-flex;align-items:center;gap:8px;'
        f'box-shadow:0 2px 8px rgba(79,70,229,0.3);transition:all 0.2s ease">'
        f'\U0001f50d View Ground Truth Graph '
        f'<span style="background:rgba(255,255,255,0.2);padding:2px 8px;border-radius:6px;'
        f'font-size:10px;font-weight:600">seed #{seed} · {domain_display}</span>'
        f'<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" '
        f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        f'<path d="M6 3H3v10h10v-3"/><path d="M9 2h5v5"/><path d="M14 2L7 9"/></svg></a></div>',
        # v3: world model panels reset
        _belief_heatmap_html({}, 0, 0),
        _ig_sparkline_html([]),
    )


def execute_step(action_type, service_name, affected, confidence, chat, state):
    env = state.get("env")
    chat = list(chat or [])
    empty_extras = ["", "", "", "", ""]  # discovered, timeline, scoreboard, graph, replay
    if env is None:
        chat.append({"role": "assistant", "content": "No active episode. Click **New Episode** to start."})
        return [state, chat, _budget_html(0, 15), "", gr.update(), gr.update(), None] + empty_extras
    if state.get("done"):
        chat.append({"role": "assistant", "content": "Episode complete. Click **New Episode** to play again."})
        return [state, chat, _budget_html(0, state["max_q"]), "", gr.update(), gr.update(), None] + empty_extras

    kw: dict = {"action_type": action_type}
    if action_type in ("query_dependents", "query_dependencies", "query_runbook",
                       "query_monitoring", "query_service_health"):
        kw["service_name"] = service_name or state.get("changed", "")
    if action_type in ("submit", "submit_hypothesis"):
        kw["affected_services"] = list(affected or [])
    if action_type == "submit_hypothesis" and confidence is not None:
        kw["confidence"] = float(confidence)

    budget_before = state["remaining"]

    try:
        obs = env.step(ServiceImpactAction(**kw))
    except Exception as exc:
        chat.append({"role": "assistant", "content": f"**Error:** {exc}"})
        return [state, chat, _budget_html(state["remaining"], state["max_q"]), "",
                gr.update(), gr.update(), None] + empty_extras  # includes graph+replay

    state["remaining"] = obs.queries_remaining
    state["done"] = obs.done
    cost = budget_before - obs.queries_remaining

    svc_label = (
        f" `{service_name}`"
        if service_name and action_type not in ("submit", "submit_hypothesis", "query_topology_diff")
        else ""
    )
    chat.append({"role": "user", "content": f"**{action_type}**{svc_label}"})
    rw = f"\n\n**Reward:** `{obs.reward:+.3f}`" if obs.reward is not None else ""
    chat.append({"role": "assistant", "content": obs.message + rw})

    # --- Track discovered services ---
    discovered = state.get("discovered", {})
    new_svcs = _extract_services(obs.message)
    for s in new_svcs:
        if s not in discovered:
            discovered[s] = action_type
    state["discovered"] = discovered

    # --- Track edges from dependents/dependencies ---
    edges_seen = state.get("edges_seen", [])
    new_edges = _extract_edges(action_type, service_name, obs.message)
    for e in new_edges:
        if e not in edges_seen:
            edges_seen.append(e)
    state["edges_seen"] = edges_seen

    # --- Track timeline ---
    steps = state.get("steps", [])
    steps.append(dict(
        n=len(steps) + 1,
        action=action_type,
        target=service_name if action_type not in ("submit", "query_topology_diff") else "",
        cost=cost,
        budget_left=obs.queries_remaining,
    ))
    state["steps"] = steps

    # --- Score + diff on submit ---
    score = ""
    if obs.done and action_type == "submit" and obs.reward is not None:
        score = _score_html(obs.message, obs.reward)
        # Build diff table from env internals
        try:
            predicted = sorted(set(affected or []))
            correct = sorted(env._correct_affected)
            score += _diff_table_html(predicted, correct)
        except Exception:
            pass
        # Add to scoreboard
        scores = state.get("scores", [])
        scores.append(dict(
            seed=state.get("seed", 0),
            diff=state.get("difficulty", "?"),
            score=obs.reward,
            steps_used=len(steps),
            budget_left=obs.queries_remaining,
        ))
        state["scores"] = scores

    # ── World Modeling state updates (v3) ─────────────────────────────────
    if obs.belief_state:
        state["belief_state"] = obs.belief_state
    if obs.world_version is not None:
        state["world_version"] = obs.world_version
    if obs.contradiction_count is not None:
        state["contradiction_count"] = obs.contradiction_count
    if obs.information_gain is not None:
        ig_hist = state.get("ig_history", [])
        ig_hist.append(round(obs.information_gain, 4))
        state["ig_history"] = ig_hist

    belief_html = _belief_heatmap_html(
        state.get("belief_state", {}),
        state.get("world_version", 0),
        state.get("contradiction_count", 0),
    )
    ig_html = _ig_sparkline_html(state.get("ig_history", []))

    # Build replay only when episode is done
    replay = ""
    if obs.done:
        ep_score = obs.reward if obs.reward is not None else 0.0
        replay = _replay_html(steps, discovered, edges_seen, state.get("changed", ""), ep_score)

    return [
        state, chat,
        _budget_html(obs.queries_remaining, state["max_q"]),
        score,
        gr.update(interactive=not obs.done),
        gr.update(),
        _env_state_dict(env),
        _discovered_html(discovered),
        _timeline_html(steps),
        _scoreboard_html(state.get("scores", [])),
        _vis_js_graph_html(discovered, edges_seen, state.get("changed", ""), obs.done),
        replay,
        belief_html,
        ig_html,
    ]


def toggle_fields(action_type):
    return (
        gr.update(visible=action_type in ("submit", "submit_hypothesis")),
        gr.update(visible=action_type == "submit_hypothesis"),
        gr.update(visible=action_type not in ("submit", "query_topology_diff")),
    )


def _belief_heatmap_html(belief_state: dict, world_version: int = 0, contradiction_count: int = 0) -> str:
    """Render a colour-coded belief-state heatmap as HTML."""
    if not belief_state:
        return (
            '<div style="color:#9ca3af;font-size:13px;text-align:center;padding:24px">'
            'Belief state will appear after the first query.</div>'
        )
    sorted_items = sorted(belief_state.items(), key=lambda x: x[1], reverse=True)
    rows = []
    for svc, conf in sorted_items:
        if conf < 0.001:
            continue
        pct = int(conf * 100)
        # Colour: green → amber → red based on confidence
        if conf >= 0.7:
            bar_color = "#22c55e"
        elif conf >= 0.4:
            bar_color = "#f59e0b"
        else:
            bar_color = "#ef4444"
        rows.append(
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;font-size:12px">'
            f'<span style="width:160px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'
            f'font-family:monospace;color:#374151">{svc}</span>'
            f'<div style="flex:1;background:#f3f4f6;border-radius:4px;height:14px">'
            f'<div style="width:{pct}%;background:{bar_color};height:14px;border-radius:4px;'
            f'transition:width 0.3s ease"></div></div>'
            f'<span style="width:36px;text-align:right;color:#6b7280">{pct}%</span>'
            f'</div>'
        )

    version_badge = (
        f'<span style="background:#dbeafe;color:#1d4ed8;font-size:11px;font-weight:700;'
        f'padding:2px 8px;border-radius:12px;margin-left:8px">v{world_version}</span>'
    )
    contradict_badge = ""
    if contradiction_count > 0:
        contradict_badge = (
            f'<span style="background:#fee2e2;color:#b91c1c;font-size:11px;font-weight:700;'
            f'padding:2px 8px;border-radius:12px;margin-left:8px">'
            f'⚡ {contradiction_count} contradiction{"s" if contradiction_count != 1 else ""}</span>'
        )

    header = (
        f'<div style="font-weight:700;font-size:13px;color:#111827;margin-bottom:10px">'
        f'Belief State{version_badge}{contradict_badge}</div>'
    )
    return (
        '<div style="padding:12px;font-family:system-ui">'
        + header
        + "".join(rows)
        + "</div>"
    )


def _ig_sparkline_html(ig_history: list[float]) -> str:
    """Render a mini sparkline of information-gain values."""
    if not ig_history:
        return ""
    n = len(ig_history)
    width = min(n * 18, 340)
    max_v = max(ig_history) if max(ig_history) > 0 else 1.0
    height = 40
    points = []
    for i, v in enumerate(ig_history):
        x = int(i / max(n - 1, 1) * (width - 4)) + 2
        y = height - 4 - int(v / max_v * (height - 8))
        points.append(f"{x},{y}")
    poly = " ".join(points)
    return (
        f'<div style="margin-top:6px">'
        f'<div style="font-size:11px;color:#6b7280;margin-bottom:2px">IG per step</div>'
        f'<svg width="{width}" height="{height}" style="border:1px solid #e5e7eb;border-radius:4px;background:#f9fafb">'
        f'<polyline points="{poly}" fill="none" stroke="#4f46e5" stroke-width="2"/>'
        f'</svg></div>'
    )


PLAYGROUND_CSS = """
.gradio-container { background: #f9fafb !important; max-width: 100% !important; }
footer { display: none !important; }
"""

PLAYGROUND_THEME = gr.themes.Soft(
    primary_hue="indigo", secondary_hue="gray", neutral_hue="gray",
    font=gr.themes.GoogleFont("Inter"),
)

# ---------------------------------------------------------------------------
# Blocks definition  (Gradio 6 -- title only, no theme/css in gr.Blocks())
# ---------------------------------------------------------------------------

with gr.Blocks(title="cascade-mind Playground") as playground_blocks:

    # Header
    gr.HTML(
        '<div style="text-align:center;padding:20px 0 16px;border-bottom:1px solid #e5e7eb;margin-bottom:18px">'
        '<div style="font-size:26px;font-weight:900;color:#111827;letter-spacing:-0.5px">'
        '&#129504; cascade-mind</div>'
        '<div style="font-size:13px;color:#6b7280;margin-top:5px">'
        'SRE Incident Response Playground &nbsp;&middot;&nbsp; '
        'Identify the blast radius of a breaking microservice change</div>'
        '<div style="margin-top:10px;font-size:12px;display:flex;justify-content:center;gap:20px">'
        '<a href="/docs" target="_blank" style="color:#4f46e5;text-decoration:none;font-weight:600;'
        'display:inline-flex;align-items:center;gap:4px">API Docs'
        '<svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M6 3H3v10h10v-3"/><path d="M9 2h5v5"/><path d="M14 2L7 9"/></svg></a>'
        '<a href="https://github.com/rajkamal2819/cascade-mind" target="_blank" '
        'style="color:#4f46e5;text-decoration:none;font-weight:600;'
        'display:inline-flex;align-items:center;gap:4px">GitHub'
        '<svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M6 3H3v10h10v-3"/><path d="M9 2h5v5"/><path d="M14 2L7 9"/></svg></a>'
        '</div></div>'
    )

    # Ground truth graph button (top, updates on episode reset)
    gt_link = gr.HTML(
        '<div style="text-align:center;padding:8px 0">'
        '<span style="background:#f3f4f6;color:#9ca3af;font-size:12px;font-weight:600;'
        'padding:8px 20px;border-radius:8px;display:inline-flex;align-items:center;gap:6px;'
        'cursor:default">'
        '\U0001f50d Start an episode to view the ground truth graph</span></div>',
    )

    session = gr.State(_empty_state())

    # Section 1: Episode Setup -- full-width horizontal bar with New Episode button
    gr.Markdown("### Episode Setup")
    with gr.Row(equal_height=True):
        diff_radio = gr.Radio(
            ["easy", "medium", "hard"], value="easy",
            label="Difficulty",
            info="easy = clean | medium = 1 noise | hard = 2 noise",
            scale=2,
        )
        domain_dd = gr.Dropdown(
            DOMAIN_CHOICES,
            value="sre",
            label="Domain",
            info="Plugin interface — swap the causal graph domain",
            scale=2,
        )
        seed_sl = gr.Slider(
            -1, 9999, value=-1, step=1,
            label="Custom Seed  (-1 uses difficulty preset)",
            scale=3,
        )
        reset_btn = gr.Button(
            "New Episode", variant="primary", size="lg", scale=1,
        )

    gr.HTML('<div style="height:10px"></div>')

    # Section 2: Budget bar + incident banner (full width)
    budget_bar = gr.HTML(_budget_html(15, 15))
    banner = gr.HTML(_idle_banner())

    gr.HTML('<div style="height:6px"></div>')

    # Section 3: Action controls (left) + Investigation log (right)
    gr.Markdown("### Investigate")
    with gr.Row(equal_height=False):

        # LEFT: action panel
        with gr.Column(scale=1, min_width=320):
            action_radio = gr.Radio(
                ACTION_CHOICES,
                value="query_dependents",
                label="Action Type",
            )
            svc_dd = gr.Dropdown(
                ALL_SERVICES, value=None,
                label="Target Service",
                info="Which service to run the action on",
                visible=True,
            )
            affected_cb = gr.CheckboxGroup(
                ALL_SERVICES, value=[],
                label="Affected Services  (submit / hypothesis only)",
                visible=False,
            )
            conf_sl = gr.Slider(
                0.0, 1.0, 0.8, step=0.05,
                label="Confidence  (submit_hypothesis only)",
                visible=False,
            )
            gr.HTML('<div style="height:6px"></div>')
            step_btn = gr.Button(
                "Execute Action",
                variant="secondary", size="lg",
                interactive=False,
            )
            gr.HTML('<hr style="border-color:#e5e7eb;margin:14px 0">')
            # Discovered services tracker (collapsible)
            with gr.Accordion("Discovered Services", open=True):
                discovered_panel = gr.HTML(_discovered_html({}))

        # RIGHT: investigation log + live graph + replay (tabbed)
        with gr.Column(scale=2, min_width=480):
            # Action timeline (compact)
            timeline_panel = gr.HTML("")
            with gr.Tabs():
                with gr.Tab("📋 Investigation Log"):
                    chatbot = gr.Chatbot(
                        value=[],
                        label="Investigation Log",
                        height=500,
                        render_markdown=True,
                        autoscroll=True,
                        placeholder="Start a new episode to begin investigating...",
                    )
                with gr.Tab("🕸️ Live Knowledge Graph"):
                    graph_panel = gr.HTML(
                        _vis_js_graph_html({}, [], ""),
                    )
                with gr.Tab("🎬 Trajectory Replay"):
                    replay_panel = gr.HTML(
                        _replay_html([], {}, [], ""),
                    )
                with gr.Tab("🧠 World Model"):
                    belief_panel = gr.HTML(
                        _belief_heatmap_html({}, 0, 0),
                        label="Belief State Heatmap",
                    )
                    ig_panel = gr.HTML(
                        _ig_sparkline_html([]),
                        label="Information Gain Timeline",
                    )
            score_card = gr.HTML("")
            with gr.Accordion("Environment State", open=False):
                state_panel = gr.JSON(value=None, label="env._state")

    # Section 4: Scoreboard (full width, collapsed)
    with gr.Accordion("Scoreboard — Episode History", open=False):
        scoreboard_panel = gr.HTML(_scoreboard_html([]))

    # Cheat sheet (collapsed)
    with gr.Accordion("Action Reference", open=False):
        gr.Markdown(
            "**Budget (-1 each):** `query_dependents` `query_dependencies` `submit_hypothesis`\n\n"
            "**Free:** `query_runbook` `query_changelog` `query_monitoring` "
            "`query_service_health` `query_topology_diff`\n\n"
            "**Terminal:** `submit`\n\n"
            "Score = 80% F-beta (b=2) + 20% Brier calibration. Recall counts 4x more than precision. Brier rewards accurate confidence estimates."
        )

    # Event wiring
    action_radio.change(
        toggle_fields, [action_radio],
        [affected_cb, conf_sl, svc_dd],
    )

    domain_dd.change(
        lambda d: (
            gr.update(choices=_services_for_domain(d)),
            gr.update(choices=_services_for_domain(d)),
        ),
        [domain_dd],
        [svc_dd, affected_cb],
    )

    reset_btn.click(
        reset_episode, [diff_radio, seed_sl, domain_dd, session],
        [session, chatbot, budget_bar, banner, score_card,
         svc_dd, step_btn, affected_cb, action_radio, state_panel,
         discovered_panel, timeline_panel, scoreboard_panel,
         graph_panel, replay_panel, gt_link,
         belief_panel, ig_panel],
    )

    step_btn.click(
        execute_step,
        [action_radio, svc_dd, affected_cb, conf_sl, chatbot, session],
        [session, chatbot, budget_bar, score_card, step_btn, affected_cb, state_panel,
         discovered_panel, timeline_panel, scoreboard_panel,
         graph_panel, replay_panel, belief_panel, ig_panel],
    )
