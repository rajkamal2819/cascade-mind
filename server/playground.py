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
    from .graph_builder import SERVICES, SERVICE_METADATA
    from .service_impact_environment import ServiceImpactEnvironment
except ImportError:
    from server.graph_builder import SERVICES, SERVICE_METADATA  # type: ignore
    from server.service_impact_environment import ServiceImpactEnvironment  # type: ignore

try:
    from ..models import ServiceImpactAction
except ImportError:
    from models import ServiceImpactAction  # type: ignore

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


def _empty_state() -> dict:
    return dict(env=None, changed="", max_q=15, remaining=15, done=False)


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
    fbeta = float(f_m.group(1)) if f_m else reward
    prec  = float(p_m.group(1)) if p_m else 0.0
    rec   = float(r_m.group(1)) if r_m else 0.0
    pct   = int(fbeta * 100)
    c     = "#059669" if pct >= 70 else ("#d97706" if pct >= 40 else "#dc2626")
    grade = "EXCELLENT" if pct >= 80 else ("GOOD" if pct >= 60 else ("FAIR" if pct >= 40 else "POOR"))
    return (
        f'<div style="background:#fff;border-top:4px solid {c};border-radius:10px;'
        f'padding:24px 20px;margin-top:10px;text-align:center;'
        f'box-shadow:0 1px 3px rgba(0,0,0,0.06),0 1px 2px rgba(0,0,0,0.04)">'
        f'<div style="font-size:10px;color:#9ca3af;font-weight:700;letter-spacing:1.2px;'
        f'margin-bottom:6px">EPISODE SCORE &middot; F-beta (b=2)</div>'
        f'<div style="font-size:56px;font-weight:900;color:{c};line-height:1">{pct}%</div>'
        f'<div style="font-size:12px;color:{c};font-weight:700;margin:4px 0 14px;'
        f'letter-spacing:0.5px">{grade}</div>'
        f'<div style="background:#f3f4f6;border-radius:99px;height:8px;overflow:hidden;'
        f'margin:0 auto 14px;max-width:180px">'
        f'<div style="width:{pct}%;height:100%;background:{c};border-radius:99px"></div></div>'
        f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;max-width:220px;margin:0 auto">'
        f'<div style="background:#fafafa;border:1px solid #e5e7eb;border-radius:8px;padding:10px">'
        f'<div style="font-size:9px;color:#9ca3af;font-weight:700;letter-spacing:0.5px">PRECISION</div>'
        f'<div style="font-size:22px;font-weight:800;color:#111827">{prec:.0%}</div></div>'
        f'<div style="background:#fafafa;border:1px solid #e5e7eb;border-radius:8px;padding:10px">'
        f'<div style="font-size:9px;color:#9ca3af;font-weight:700;letter-spacing:0.5px">RECALL</div>'
        f'<div style="font-size:22px;font-weight:800;color:#111827">{rec:.0%}</div></div></div></div>'
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


def reset_episode(difficulty: str, custom_seed: int, _state: dict):
    seed = DIFFICULTY_SEED.get(difficulty, 0) if int(custom_seed) < 0 else int(custom_seed)
    env = ServiceImpactEnvironment()
    obs = env.reset(seed=seed)
    svc, mq = env._changed_service, env._max_queries
    st = dict(env=env, changed=svc, max_q=mq, remaining=mq, done=False)
    chat = [{"role": "assistant", "content": f"**Incident triggered!**\n\n{obs.message}"}]
    return (
        st,
        chat,
        _budget_html(mq, mq),
        _banner_html(svc, difficulty, seed),
        "",                                   # clear score card
        gr.update(value=svc),                 # pre-select service dropdown
        gr.update(interactive=True),          # enable Execute button
        gr.update(value=[]),                  # clear affected checkboxes
        gr.update(value="query_dependents"),  # reset action radio
        _env_state_dict(env),                 # env state panel
    )


def execute_step(action_type, service_name, affected, confidence, chat, state):
    env = state.get("env")
    chat = list(chat or [])
    if env is None:
        chat.append({"role": "assistant", "content": "No active episode. Click **New Episode** to start."})
        return state, chat, _budget_html(0, 15), "", gr.update(), gr.update(), None
    if state.get("done"):
        chat.append({"role": "assistant", "content": "Episode complete. Click **New Episode** to play again."})
        return state, chat, _budget_html(0, state["max_q"]), "", gr.update(), gr.update(), None

    kw: dict = {"action_type": action_type}
    if action_type in ("query_dependents", "query_dependencies", "query_runbook",
                       "query_monitoring", "query_service_health"):
        kw["service_name"] = service_name or state.get("changed", "")
    if action_type in ("submit", "submit_hypothesis"):
        kw["affected_services"] = list(affected or [])
    if action_type == "submit_hypothesis" and confidence is not None:
        kw["confidence"] = float(confidence)

    try:
        obs = env.step(ServiceImpactAction(**kw))
    except Exception as exc:
        chat.append({"role": "assistant", "content": f"**Error:** {exc}"})
        return state, chat, _budget_html(state["remaining"], state["max_q"]), "", gr.update(), gr.update(), None

    state["remaining"] = obs.queries_remaining
    state["done"] = obs.done

    svc_label = (
        f" `{service_name}`"
        if service_name and action_type not in ("submit", "submit_hypothesis", "query_topology_diff")
        else ""
    )
    chat.append({"role": "user", "content": f"**{action_type}**{svc_label}"})
    rw = f"\n\n**Reward:** `{obs.reward:+.3f}`" if obs.reward is not None else ""
    chat.append({"role": "assistant", "content": obs.message + rw})

    score = ""
    if obs.done and action_type == "submit" and obs.reward is not None:
        score = _score_html(obs.message, obs.reward)

    return (
        state, chat,
        _budget_html(obs.queries_remaining, state["max_q"]),
        score,
        gr.update(interactive=not obs.done),
        gr.update(),
        _env_state_dict(env),
    )


def toggle_fields(action_type):
    return (
        gr.update(visible=action_type in ("submit", "submit_hypothesis")),
        gr.update(visible=action_type == "submit_hypothesis"),
        gr.update(visible=action_type not in ("submit", "query_topology_diff")),
    )


# ---------------------------------------------------------------------------
# Theme / CSS  (consumed by mount_gradio_app in app.py -- NOT in gr.Blocks)
# ---------------------------------------------------------------------------

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
            gr.Markdown(
                "**Budget (-1 each):** `query_dependents` `query_dependencies` `submit_hypothesis`\n\n"
                "**Free:** `query_runbook` `query_changelog` `query_monitoring` "
                "`query_service_health` `query_topology_diff`\n\n"
                "**Terminal:** `submit`\n\n"
                "Score = F-beta (b=2). Recall counts 4x more than precision."
            )

        # RIGHT: investigation log
        with gr.Column(scale=2, min_width=480):
            chatbot = gr.Chatbot(
                value=[],
                label="Investigation Log",
                height=580,
                render_markdown=True,
                autoscroll=True,
                placeholder="Start a new episode to begin investigating...",
            )
            score_card = gr.HTML("")
            with gr.Accordion("Environment State", open=False):
                state_panel = gr.JSON(value=None, label="env._state")

    # Event wiring
    action_radio.change(
        toggle_fields, [action_radio],
        [affected_cb, conf_sl, svc_dd],
    )

    reset_btn.click(
        reset_episode, [diff_radio, seed_sl, session],
        [session, chatbot, budget_bar, banner, score_card,
         svc_dd, step_btn, affected_cb, action_radio, state_panel],
    )

    step_btn.click(
        execute_step,
        [action_radio, svc_dd, affected_cb, conf_sl, chatbot, session],
        [session, chatbot, budget_bar, score_card, step_btn, affected_cb, state_panel],
    )
