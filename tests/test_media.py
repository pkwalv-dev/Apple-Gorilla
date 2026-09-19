"""Tests for AG's local media generation — ComfyUI backend, image and video.

Hermetic: no ComfyUI, no GPU, no weights. What is tested is everything AG can get
wrong on its own — which backend it picks, whether a workflow template is well-formed
and substitutable, that AG's own annotations never reach the server, that a failure
is reported rather than faked, and that the tools are offered only when enabled.
"""
import dataclasses
import json

import pytest

from ag import comfy, images, reason, video
from ag.config import Config
from ag.permissions import PermissionBroker


def _cfg(**kw):
    return dataclasses.replace(Config(), **kw)


# --- the shipped workflows --------------------------------------------------
@pytest.mark.parametrize("name", ["chroma1hd_txt2img", "wan22_ti2v_txt2vid"])
def test_shipped_workflows_load_and_name_their_weights(name):
    g = comfy.load_workflow(name)
    assert comfy.nodes(g), "a workflow must contain nodes"
    # Every node is an API-format node, not the editor format.
    for nid, node in comfy.nodes(g).items():
        assert "class_type" in node, f"{name} node {nid} is not API format"
    # It says which model it is and what files that needs, so AG can tell the user
    # what to install rather than failing with a bare KeyError from the server.
    assert comfy.meta(g).get("model")
    assert comfy.describe(g), f"{name} names no model files"


def test_ag_annotations_never_reach_the_server():
    """The `_ag` block is documentation. ComfyUI would reject it as a node."""
    g = comfy.load_workflow("chroma1hd_txt2img")
    assert "_ag" in g and "_ag" not in comfy.nodes(g)


def test_the_image_workflow_keeps_real_cfg_and_a_negative_prompt():
    """This is the point of Chroma over FLUX: negative prompts actually work. A graph
    that quietly sets cfg 1.0 would silently discard every negative prompt AG sends."""
    g = comfy.load_workflow("chroma1hd_txt2img")
    sampler = next(n for n in comfy.nodes(g).values() if n["class_type"] == "KSampler")
    assert sampler["inputs"]["cfg"] > 1.0
    texts = [n["inputs"]["text"] for n in comfy.nodes(g).values()
             if n["class_type"] == "CLIPTextEncode"]
    assert comfy.PROMPT in texts and comfy.NEGATIVE in texts


def test_a_missing_workflow_says_where_to_put_one():
    with pytest.raises(RuntimeError) as e:
        comfy.load_workflow("no_such_workflow")
    assert "Export (API)" in str(e.value)


# --- substitution -----------------------------------------------------------
def test_placeholders_keep_their_type():
    """ComfyUI validates input types, so a step count must arrive as an int, not the
    string "20" — which is what naive templating would send."""
    g = comfy.load_workflow("chroma1hd_txt2img")
    out = comfy.fill(g, {comfy.STEPS: 20, comfy.WIDTH: 768, comfy.PROMPT: "a bike"})
    sampler = next(n for n in comfy.nodes(out).values() if n["class_type"] == "KSampler")
    assert sampler["inputs"]["steps"] == 20 and isinstance(
        sampler["inputs"]["steps"], int)
    latent = next(n for n in comfy.nodes(out).values()
                  if n["class_type"] == "EmptySD3LatentImage")
    assert latent["inputs"]["width"] == 768


def test_a_placeholder_inside_a_longer_string_is_interpolated():
    filled = comfy.fill({"1": {"inputs": {"text": f"cinematic, {comfy.PROMPT}"}}},
                        {comfy.PROMPT: "a bike"})
    assert filled["1"]["inputs"]["text"] == "cinematic, a bike"


def test_fill_does_not_mutate_the_template():
    g = comfy.load_workflow("chroma1hd_txt2img")
    before = json.dumps(g, sort_keys=True)
    comfy.fill(g, {comfy.PROMPT: "x", comfy.STEPS: 99})
    assert json.dumps(g, sort_keys=True) == before


def test_a_user_workflow_wins_over_the_shipped_one(tmp_path, monkeypatch):
    """Swapping models must not mean patching AG."""
    monkeypatch.setattr(comfy, "USER_WORKFLOW_DIR", tmp_path)
    mine = {"1": {"class_type": "SaveImage", "inputs": {"filename_prefix": "mine"}}}
    (tmp_path / "chroma1hd_txt2img.json").write_text(json.dumps(mine), encoding="utf-8")
    assert comfy.load_workflow("chroma1hd_txt2img") == mine


def test_workflow_names_cannot_escape_the_workflow_directories(monkeypatch, tmp_path):
    monkeypatch.setattr(comfy, "USER_WORKFLOW_DIR", tmp_path)
    assert comfy.workflow_path("../../config") is None


# --- backend choice ---------------------------------------------------------
def test_an_explicit_backend_is_obeyed(monkeypatch):
    monkeypatch.setattr(comfy, "reachable", lambda cfg, **k: False)
    monkeypatch.setattr(images, "sd_reachable", lambda cfg, **k: True)
    assert images.backend_for(_cfg(image_backend="comfy")) == "comfy"
    assert images.backend_for(_cfg(image_backend="a1111")) == "a1111"


def test_auto_prefers_comfyui_because_that_is_where_the_open_models_run(monkeypatch):
    monkeypatch.setattr(comfy, "reachable", lambda cfg, **k: True)
    monkeypatch.setattr(images, "sd_reachable", lambda cfg, **k: True)
    assert images.backend_for(_cfg(image_backend="auto")) == "comfy"


def test_auto_falls_back_so_an_existing_sd_install_keeps_working(monkeypatch):
    monkeypatch.setattr(comfy, "reachable", lambda cfg, **k: False)
    monkeypatch.setattr(images, "sd_reachable", lambda cfg, **k: True)
    assert images.backend_for(_cfg(image_backend="auto")) == "a1111"


# --- failure is reported, never faked ---------------------------------------
def test_an_unreachable_server_names_the_host_and_raises(monkeypatch):
    import urllib.error
    import urllib.request

    def boom(*a, **k):
        raise urllib.error.URLError("refused")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(RuntimeError) as e:
        comfy.run(comfy.load_workflow("chroma1hd_txt2img"), _cfg())
    assert "127.0.0.1:8188" in str(e.value)


def test_video_generation_refuses_when_disabled():
    with pytest.raises(RuntimeError) as e:
        video.generate("a bike", _cfg(allow_video_gen=False))
    assert "disabled" in str(e.value)


def test_an_empty_prompt_is_refused_before_any_gpu_time():
    with pytest.raises(ValueError):
        video.generate("   ", _cfg())


# --- clip length ------------------------------------------------------------
def test_seconds_become_a_frame_count_the_model_accepts(monkeypatch, tmp_path):
    """Wan's latent packing needs 4n+1 frames; asking for 3s at 24fps must not send
    72 and fail deep inside the sampler."""
    seen = {}

    def fake_run(graph, cfg, **k):
        node = next(n for n in comfy.nodes(graph).values()
                    if n["class_type"] == "Wan22ImageToVideoLatent")
        seen["frames"] = node["inputs"]["length"]
        return [("out.mp4", b"\0")]

    monkeypatch.setattr(comfy, "run", fake_run)
    monkeypatch.setattr(comfy, "reachable", lambda cfg, **k: True)
    monkeypatch.setattr(video, "VIDEO_DIR", tmp_path)     # never touch real state/
    r = video.generate("a bike", _cfg(video_fps=24), seconds=3)
    assert seen["frames"] % 4 == 1
    assert 2.5 <= r.seconds <= 3.5


# --- the tools --------------------------------------------------------------
def test_both_media_tools_are_offered_when_enabled():
    names = {t.name for t in reason.available_tools(
        PermissionBroker(allow_external_tools=True), None,
        _cfg(allow_image_gen=True, allow_video_gen=True))}
    assert {"generate_image", "generate_video"} <= names


def test_a_disabled_medium_is_not_offered():
    names = {t.name for t in reason.available_tools(
        PermissionBroker(allow_external_tools=True), None,
        _cfg(allow_image_gen=False, allow_video_gen=False))}
    assert "generate_image" not in names and "generate_video" not in names


def test_a_failed_generation_is_reported_to_the_model_not_swallowed(monkeypatch):
    """The model must learn the truth from the observation — a tool that returns a
    cheerful string on failure teaches it to claim videos it never made."""
    monkeypatch.setattr(video, "generate",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no GPU")))
    tools = {t.name: t for t in reason.available_tools(
        PermissionBroker(allow_external_tools=True), None, _cfg())}
    out = tools["generate_video"].run({"prompt": "a bike"}, None)
    assert out.startswith("generate_video error") and "no GPU" in out


def test_the_media_control_is_declared_and_maps_to_real_fields():
    from ag import controls
    out = controls.overrides_for({"tools": True, "media": False})
    assert out["allow_image_gen"] is False and out["allow_video_gen"] is False
    # And it cannot take effect without the tool loop that would call it.
    assert controls.overrides_for({"tools": False, "media": True})["allow_video_gen"] \
        is False


# --- serving a generated clip back to the page ------------------------------
def test_only_ags_own_output_directories_are_served(tmp_path, monkeypatch):
    """The page hands this endpoint a path, so it must not become a file reader."""
    from ag import server
    from ag.config import STATE_DIR

    vid = STATE_DIR / "video"
    vid.mkdir(parents=True, exist_ok=True)
    mine = vid / "probe-clip.mp4"
    mine.write_bytes(b"fake-clip")
    try:
        assert server.media_file(str(mine)) == mine.resolve()
        assert server.media_type(mine) == "video/mp4"
        # Anything else is simply not found.
        outside = tmp_path / "secrets.txt"
        outside.write_text("nope", encoding="utf-8")
        assert server.media_file(str(outside)) is None
        assert server.media_file(str(vid / ".." / ".." / "config.json")) is None
        assert server.media_file("") is None
    finally:
        mine.unlink(missing_ok=True)
