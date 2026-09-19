"""Tests for the declared run controls.

The property under test: a control in the GUI does something. Every declaration
reaches the markup, the browser's generic JS, and a real Config field — and a control
whose prerequisite is off cannot take effect, however the request was made.
"""
import dataclasses
import json

import pytest

from ag import controls, server
from ag.config import Config


# --- every declaration is wired end to end ----------------------------------
def test_every_control_reaches_the_page_and_the_browser_spec():
    spec = {c["id"]: c for c in json.loads(controls.spec_json())}
    for c in controls.RUN_CONTROLS:
        assert f'id="{c.id}"' in server.PAGE, f"{c.id} is declared but not rendered"
        assert c.id in spec, f"{c.id} is rendered but the page's JS cannot read it"
        assert spec[c.id]["kind"] == c.kind


def test_every_control_maps_to_a_real_config_field():
    """A control naming a field that no longer exists is the silent failure this whole
    module is meant to prevent: it would look live and change nothing."""
    fields = {f.name for f in dataclasses.fields(Config)}
    for c in controls.RUN_CONTROLS:
        for name in c.sets:
            assert name in fields, f"{c.id} sets unknown config field {name!r}"


def test_controls_that_declare_a_prerequisite_name_a_real_one():
    for c in controls.RUN_CONTROLS:
        if c.requires:
            assert c.requires in controls.BY_ID


def test_help_and_titles_are_escaped_into_the_markup():
    html = controls.render_html()
    assert "<think>" not in html            # the verbose title contains angle brackets
    assert "&lt;think&gt;" in html


# --- requests become config overrides ---------------------------------------
def test_absent_keys_keep_the_standing_config():
    """A bare {"prompt": ...} — an older client, or a script — must behave as before
    rather than silently switching everything off."""
    assert controls.overrides_for({"prompt": "hi"}) == {}


def test_toggles_and_choices_become_overrides():
    out = controls.overrides_for({"tools": True, "memory": False, "think": "on",
                                  "web": False})
    assert out["allow_local_tools"] is True
    assert out["use_memory"] is False and out["auto_memory"] is False
    assert out["think"] == "on"
    assert out["allow_web"] is False


def test_a_dead_prerequisite_is_enforced_server_side():
    """The page disables 'run code' when tools is off, but the page is not the
    authority — the same request can arrive from a script."""
    out = controls.overrides_for({"tools": False, "code_exec": True, "acquire": True})
    assert out["allow_local_tools"] is False
    assert out["allow_code_exec"] is False
    assert out["allow_acquire"] is False


def test_prerequisites_are_followed_through_the_whole_chain():
    """autonomy needs acquire, which needs tools — so tools going off must reach
    autonomy, not just its immediate dependency."""
    out = controls.overrides_for({"tools": False, "acquire": True, "autonomy": "auto"})
    assert out["allow_acquire"] is False
    assert "acquisition_autonomy" not in out      # a choice with no live prerequisite


def test_autonomy_survives_when_its_prerequisites_hold():
    out = controls.overrides_for({"tools": True, "acquire": True, "autonomy": "auto"})
    assert out["acquisition_autonomy"] == "auto"


def test_unknown_choice_values_fall_back_to_the_default():
    assert controls.overrides_for({"think": "sideways"})["think"] == "auto"
    assert controls.overrides_for({"tools": True, "acquire": True,
                                   "autonomy": "yolo"})["acquisition_autonomy"] == "ask"


def test_overrides_apply_cleanly_to_a_real_config():
    cfg = dataclasses.replace(Config(), **controls.overrides_for(
        {"tools": True, "code_exec": True, "memory": False, "web": False}))
    assert cfg.allow_local_tools and cfg.allow_code_exec
    assert not cfg.use_memory and not cfg.allow_web


# --- adding one is adding one -----------------------------------------------
def test_a_new_control_needs_no_html_or_javascript():
    """The point of the module: declaring a control is enough to render it, persist it,
    send it, and have it change the run."""
    extra = controls.Control(id="allow_images", kind=controls.TOGGLE,
                             label="images", default=False,
                             sets=("allow_image_gen",), requires="tools")
    specs = controls.RUN_CONTROLS + (extra,)
    html = controls.render_html(specs)
    assert 'id="allow_images"' in html and "images</label>" in html
    assert '"id":"allow_images"' in controls.spec_json(specs)
    out = controls.overrides_for({"tools": True, "allow_images": True}, specs)
    assert out["allow_image_gen"] is True
