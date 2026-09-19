"""The run controls — one declaration, used by the page, the browser, and the server.

A control in AG's web app has to exist in four places to work: the HTML that draws it,
the JavaScript that remembers it and puts it in the request, the request parser, and
the config override it maps to. When those four are written out by hand, adding a
capability means editing four places and forgetting one of them is invisible — you get
a checkbox that looks live, remembers its state, and changes nothing. A control that
does not affect the run is worse than a missing one: it is a claim the product makes
and does not keep.

So the four are generated from one list. `RUN_CONTROLS` below is the whole surface:

    render_html()    -> the markup, including the help buttons
    spec_json()      -> the same list, for the page's generic save/restore/collect JS
    overrides_for()  -> the config fields a request actually changes

Adding a capability to the GUI is adding one `Control` here. There is nowhere else to
forget, and nothing to keep in sync.

`requires` expresses the other half of the promise: a control whose prerequisite is off
cannot have an effect, so it is disabled in the page AND neutralised server-side. The
UI never offers a switch that does nothing.
"""
from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

TOGGLE = "toggle"
CHOICE = "choice"


@dataclass(frozen=True)
class Control:
    """One run control. `id` is the DOM id, the localStorage key, and the request key —
    deliberately the same string, so a control cannot be half-wired."""

    id: str
    kind: str
    label: str
    default: Any = False
    options: Tuple[Tuple[str, str], ...] = ()   # (value, label) pairs, CHOICE only
    sets: Tuple[str, ...] = ()                  # Config fields this control overrides
    requires: str = ""                          # id of a control that must be on
    help: str = ""                              # "?" popover; omitted when empty
    title: str = ""                             # hover text on the control itself

    def coerce(self, raw: Any) -> Any:
        """Validate one incoming value. Unknown choices fall back to the default rather
        than reaching the config — this is untrusted input from a browser."""
        if self.kind == TOGGLE:
            return bool(raw)
        v = str(raw).lower()
        allowed = {value for value, _ in self.options}
        return v if v in allowed else str(self.default)


# The run controls, in the order they appear in the bar. `model` is deliberately absent:
# its options are discovered at runtime from Ollama and the signed-in backend, so it
# keeps its own loader.
RUN_CONTROLS: Tuple[Control, ...] = (
    Control(
        id="verbose", kind=TOGGLE, label="show reasoning", default=False,
        title="Stream the model's raw output — including its reasoning (Ollama's "
              "<think> blocks) — live as it is generated. Stop ends the response and "
              "returns control immediately; a local model may take a moment more to "
              "wind down in the background.",
    ),
    Control(
        id="think", kind=CHOICE, label="thinking", default="auto",
        options=(("auto", "auto"), ("off", "off · fast"), ("on", "on")),
        sets=("think",),
        help="Extended thinking, like the toggle in the Claude app. 'off' suppresses "
             "the model's step-by-step reasoning (fastest per call); 'on' forces it; "
             "'auto' leaves the model to decide. Only affects thinking-capable models "
             "(Claude, Qwen3); ignored by others.",
    ),
    Control(
        id="web", kind=TOGGLE, label="internet", default=True, sets=("allow_web",),
        title="Search and fetch live sources for this run. Fetched pages are untrusted "
              "reference data, never instructions.",
    ),
    Control(
        id="tools", kind=TOGGLE, label="tools", default=True,
        sets=("allow_local_tools",),
        help="Turns on AG's reason→act→observe loop: it can call tools (exact-math "
             "calc, read local files, recall/save memory, delegate to a sub-agent) "
             "before answering. Off = a single model call with no tool use.",
    ),
    Control(
        id="code_exec", kind=TOGGLE, label="run code", default=False,
        sets=("allow_code_exec",), requires="tools",
        help="DANGEROUS: lets the python_exec tool run real Python in a subprocess on "
             "this machine. Requires 'tools'. Off by default — only enable for prompts "
             "you trust.",
    ),
    Control(
        id="acquire", kind=TOGGLE, label="self-extend", default=False,
        sets=("allow_acquire",), requires="tools",
        help="If AG lacks a capability this prompt needs, it authors a new tested "
             "skill (and may install Python deps / fetch allowlisted code), then uses "
             "it. Requires 'tools'. New skills persist and are reused, and are "
             "inherited by sub-agents.",
    ),
    Control(
        id="autonomy", kind=CHOICE, label="acquire", default="ask",
        options=(("ask", "ask first"), ("auto", "auto")),
        sets=("acquisition_autonomy",), requires="acquire",
        title="How self-extend proceeds. ask = plan and wait for your approval before "
              "installing/running anything (surfaced in the live trace). auto = "
              "complete end-to-end within this run's grants.",
    ),
    Control(
        id="media", kind=TOGGLE, label="media", default=True,
        sets=("allow_image_gen", "allow_video_gen"), requires="tools",
        help="Lets AG generate images and video on this machine's GPU (Chroma1-HD "
             "and Wan 2.2 under ComfyUI, or an Automatic1111 server for images). "
             "Requires 'tools'. Nothing is sent anywhere: the prompt and the output "
             "stay local. A clip takes minutes.",
    ),
    Control(
        id="memory", kind=TOGGLE, label="memory", default=True,
        sets=("use_memory", "auto_memory"),
        title="Recall durable facts from long-term memory into context, and distill "
              "new durable facts after the answer. Off = this run neither reads nor "
              "writes long-term memory.",
    ),
    Control(
        id="uncensored", kind=TOGGLE, label="uncensored", default=False,
        sets=("uncensored",),
        help="Force the abliterated (uncensored) model for every turn. Off (default) = "
             "AG routes plain conversation to a stronger instruct model and uses the "
             "abliterated model only for tool/code tasks. On = the abliterated model "
             "answers everything, for raw, unfiltered output.",
    ),
)

BY_ID: Dict[str, Control] = {c.id: c for c in RUN_CONTROLS}


# --- the page ---------------------------------------------------------------
def _attr(text: str) -> str:
    """Escape for an HTML attribute — quotes included, since every value here lands
    inside one."""
    return html.escape(str(text or ""), quote=True)


def render_html(controls: Sequence[Control] = RUN_CONTROLS, *,
                extra_before: str = "") -> str:
    """The control bar's markup. `extra_before` is spliced in after the first control
    so the runtime-populated model picker can keep its hand-written slot."""
    out = []
    for i, c in enumerate(controls):
        out.append(_control_html(c))
        if i == 0 and extra_before:
            out.append(extra_before)
        if c.help:
            out.append('      <button class="help" data-help="'
                       + _attr(c.help) + '">?</button>')
    return "\n".join(out)


def _control_html(c: Control) -> str:
    title = f' title="{_attr(c.title)}"' if c.title else ""
    wrap_id = f' id="{c.id}-wrap"'
    if c.kind == TOGGLE:
        checked = " checked" if c.default else ""
        return (f'      <label class="ctl"{wrap_id}{title}>'
                f'<input type="checkbox" id="{c.id}" onchange="syncCtls()"{checked}> '
                f'{html.escape(c.label)}</label>')
    opts = "".join(f'<option value="{_attr(v)}">{html.escape(lbl)}</option>'
                   for v, lbl in c.options)
    return (f'      <label class="ctl"{wrap_id}{title}>{html.escape(c.label)} '
            f'<select id="{c.id}" class="ctl-select" onchange="syncCtls()">{opts}'
            f'</select></label>')


def spec_json(controls: Sequence[Control] = RUN_CONTROLS) -> str:
    """The list the page's generic JS drives itself from. Only what the browser needs:
    how to read the control, what to default it to, and what it depends on."""
    return json.dumps([{"id": c.id, "kind": c.kind, "default": c.default,
                        "requires": c.requires} for c in controls],
                      separators=(",", ":"))


# --- the request ------------------------------------------------------------
def values_from(payload: dict,
                controls: Sequence[Control] = RUN_CONTROLS) -> Dict[str, Any]:
    """Read and validate the controls present in a request.

    A missing key means "not sent" (None), which keeps the standing config value — so
    an older client, or a script posting only a prompt, behaves exactly as before.
    """
    out: Dict[str, Any] = {}
    for c in controls:
        raw = payload.get(c.id, None)
        out[c.id] = None if raw is None else c.coerce(raw)
    return out


def overrides_for(payload: dict,
                  controls: Sequence[Control] = RUN_CONTROLS) -> Dict[str, Any]:
    """The Config fields this request changes, for one run only.

    A control whose prerequisite is off is forced off here too. The page already
    disables it, but the page is not the authority: the same request could arrive from
    a script, and 'run code' must not be honoured by a run that has no tool loop.
    """
    vals = values_from(payload, controls)
    out: Dict[str, Any] = {}
    for c in controls:
        if not c.sets:
            continue
        v = vals.get(c.id)
        if v is None:
            continue
        if c.requires and not _prerequisite_met(c, vals, controls):
            if c.kind == TOGGLE:
                v = False
            else:
                continue
        for field_name in c.sets:
            out[field_name] = v
    return out


def _prerequisite_met(c: Control, vals: Dict[str, Any],
                      controls: Sequence[Control]) -> bool:
    """Walk the whole `requires` chain: 'acquire' needs 'tools', and 'autonomy' needs
    'acquire', so autonomy must fall when tools does."""
    seen = set()
    dep_id = c.requires
    while dep_id and dep_id not in seen:
        seen.add(dep_id)
        dep = BY_ID.get(dep_id)
        if dep is None:
            return True
        v = vals.get(dep_id)
        if v is None:          # not sent: the standing config decides, so allow it
            return True
        if dep.kind == TOGGLE and not v:
            return False
        dep_id = dep.requires
    return True
