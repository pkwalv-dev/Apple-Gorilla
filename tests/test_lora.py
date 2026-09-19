"""Tests for the LoRA subsystem — dataset build, feasibility, train preflight.

Hermetic: no GPU and no heavy deps (torch/peft/...). We exercise everything that runs
without them; actual training is guarded by a preflight we assert refuses cleanly."""
from types import SimpleNamespace

from ag import lora
from ag.config import Config


def _patch(monkeypatch, tmp_path):
    from ag import config as cfgmod
    from ag.memory import manager as mgrmod
    from ag.memory.embed import HashingEmbedder
    monkeypatch.setattr(cfgmod, "STATE_DIR", tmp_path)
    monkeypatch.setattr(lora, "LORA_DIR", tmp_path / "lora")
    monkeypatch.setattr(lora, "DATASET_FILE", tmp_path / "lora" / "dataset.jsonl")
    monkeypatch.setattr(lora, "ADAPTERS_DIR", tmp_path / "lora" / "adapters")
    monkeypatch.setattr(mgrmod, "get_embedder", lambda cfg=None, **k: HashingEmbedder())
    import ag.memory as memory
    memory.reset()


class FakeTeacher:
    def complete(self, **kw):
        return SimpleNamespace(text="A precise answer.", input_tokens=0, output_tokens=0)


def test_feasibility_reports_missing_deps_when_absent(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    f = lora.feasibility(Config())
    # The heavy deps aren't installed in the test env → reported, not crashed.
    assert set(f.missing_deps) & set(lora._HEAVY_DEPS)
    assert f.ok is False
    assert isinstance(f.as_dict(), dict)


def test_build_dataset_from_memory(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    import ag.memory as memory
    mgr = memory.get_manager("root", cfg=Config())
    # Answers have to be substantive to train: a bare "4" is not a lesson, and a pair
    # that thin teaches the model to answer with nothing. See _MIN_ANSWER_CHARS.
    mgr.record_episode("What is 2+2?",
                       "It is 4. For anything beyond mental arithmetic, call the calc "
                       "tool rather than computing it in your head.", score=9.0)
    mgr.record_episode("bad one", "wrong", score=2.0)           # low score -> dropped
    mgr.record_episode("terse one", "Yep.", score=9.0)          # too thin -> dropped
    st = lora.build_dataset(Config(), use_teacher=False)
    assert st.from_memory >= 1 and st.total == st.from_memory
    assert lora.dataset_size() == st.total
    # Neither the low-scored nor the threadbare exchange may appear.
    text = lora.DATASET_FILE.read_text(encoding="utf-8")
    assert "2+2" in text and "bad one" not in text and "terse one" not in text


def test_build_dataset_from_teacher(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    st = lora.build_dataset(Config(), use_memory=False, use_teacher=True,
                            client=FakeTeacher())
    assert st.from_teacher > 0 and st.total == st.from_teacher


def test_build_dataset_merges_and_dedups(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    import ag.memory as memory
    mgr = memory.get_manager("root", cfg=Config())
    mgr.record_episode("q", "a", score=8.0)
    st1 = lora.build_dataset(Config(), use_teacher=False)
    st2 = lora.build_dataset(Config(), use_teacher=False)   # rebuild -> no growth (dedup)
    assert st1.total == st2.total


def test_train_refuses_without_deps_or_gpu(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    # Even with a dataset present, train() must refuse cleanly when deps/GPU are absent.
    import ag.memory as memory
    mgr = memory.get_manager("root", cfg=Config())
    for i in range(20):
        mgr.record_episode(f"q{i}", f"a{i}", score=8.0)
    lora.build_dataset(Config(), use_teacher=False)
    res = lora.train(Config())
    assert res.ok is False
    assert ("extras" in res.reason) or ("CUDA" in res.reason)


def test_list_adapters_empty(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    assert lora.list_adapters() == []


def test_available_bases_annotates_fit_by_vram():
    # 8GB: an 8B is 'tight'/'fits'-ish; a 14B won't fit; a 4B fits.
    b8 = {x["id"]: x for x in lora.available_bases(8.0)}
    assert b8["Qwen/Qwen3-4B"]["fit"] == "fits"
    assert b8["Qwen/Qwen3-14B"]["fit"] == "won't fit"
    # 24GB: everything fits.
    b24 = {x["id"]: x for x in lora.available_bases(24.0)}
    assert b24["Qwen/Qwen3-8B"]["fit"] == "fits"


def _proc(text, tags=()):
    from ag.memory.types import Memory, MemoryKind
    return Memory(id="p", text=text, kind=MemoryKind.PROCEDURAL, tags=list(tags))


def test_procedures_are_asked_about_their_own_situation():
    """Each procedure records the situation it applies to; that is the question. Filing
    them all under one generic prompt trains the model to ignore the prompt."""
    p = lora._procedure_pair(_proc("arith — when math prompts — use calc; return exact"))
    assert p["instruction"] == "What is the best approach when math prompts?"
    assert p["output"] == "use calc; return exact"   # the steps, not the label

    p = lora._procedure_pair(_proc("For arithmetic, call calc first."))
    assert p["instruction"] == "What is a good approach for arithmetic?"

    p = lora._procedure_pair(_proc("When the build is red, bisect before guessing."))
    assert p["instruction"] == "What is a good approach when the build is red?"

    p = lora._procedure_pair(_proc("Acquired skill 'fx': converts currencies.",
                                   tags=["skill", "fx"]))
    assert "'fx' skill" in p["instruction"]


def test_shapeless_procedures_are_skipped_not_given_a_generic_prompt():
    assert lora._procedure_pair(_proc("Prefer small commits.")) is None


def test_procedure_instructions_do_not_collide(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    import ag.memory as memory
    from ag.memory import MemoryKind, Origin
    mgr = memory.get_manager("root", cfg=Config())
    for name, when, steps in (("arith", "math prompts", "call calc first"),
                              ("triage", "the build is red", "bisect, then revert"),
                              ("review", "a diff is large", "read tests first")):
        mgr.remember(f"{name} — when {when} — {steps}", kind=MemoryKind.PROCEDURAL,
                     origin=Origin.OBSERVED)
    pairs = lora._pairs_from_memory(Config())
    assert len(pairs) == 3
    assert len({p["instruction"] for p in pairs}) == 3   # one prompt each, not one shared


def test_both_training_paths_target_the_same_modules():
    """Falling back from Unsloth to transformers+peft must change speed and memory, not
    which fine-tune you end up with."""
    assert set(lora.LORA_TARGET_MODULES) == {
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    import pathlib
    src = pathlib.Path(lora.__file__).read_text(encoding="utf-8")
    # Both call sites read the one list — neither hardcodes its own.
    assert src.count("target_modules=list(LORA_TARGET_MODULES)") == 2
    assert "target_modules=[" not in src


class _StubTokenizer:
    """Word-per-token stand-in, so the windowing logic can be tested without torch."""

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False):
        raise RuntimeError("no chat template")  # exercise the plain-text fallback

    def __call__(self, text, truncation=False, max_length=None):
        ids = list(range(len(text.split())))
        if truncation and max_length:
            ids = ids[:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def test_format_example_reports_whether_the_pair_fits():
    """Training on a clipped answer teaches clipped answers, so the pair has to say
    whether it survived the window — the caller drops it if not."""
    tok = _StubTokenizer()
    short = lora._format_example(tok, "a question", "a short answer", max_seq=64)
    assert short["complete"] is True
    long = lora._format_example(tok, "a question", "word " * 500, max_seq=64)
    assert long["complete"] is False
    assert len(long["input_ids"]) == 64            # it WAS cut, hence the flag


def test_format_example_masks_the_prompt():
    tok = _StubTokenizer()
    enc = lora._format_example(tok, "the question here", "the answer here",
                               max_seq=64)
    assert enc["labels"][0] == -100                # loss is on the answer only
    assert any(l != -100 for l in enc["labels"])


def test_recommended_config_scales_with_vram():
    assert lora.recommended_config(8.0)["base"] == "Qwen/Qwen3-8B"
    # Tight VRAM shortens the window, but not below what a whole answer needs — a
    # window that clips the training target is the more expensive kind of saving.
    assert 768 <= lora.recommended_config(8.0)["max_seq"] < 1024
    assert lora.recommended_config(5.0)["base"] == "Qwen/Qwen3-4B"
    assert lora.recommended_config(24.0)["max_seq"] >= 1024
    assert lora.recommended_config(8.0)["grad_checkpointing"] is True


def test_merge_feasibility_reports_missing(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    mf = lora.merge_feasibility(Config())
    assert isinstance(mf["ok"], bool)
    assert "notes" in mf
    # Deps aren't installed in the test env, so it must not be ready and must say why.
    assert mf["ok"] is False and mf["notes"]


def test_merge_refuses_cleanly_without_stack(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    res = lora.merge_to_gguf(Config(), "nonexistent-adapter")
    assert res.ok is False and res.reason
