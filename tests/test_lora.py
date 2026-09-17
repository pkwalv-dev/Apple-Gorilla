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
    mgr.record_episode("What is 2+2?", "4", score=9.0)          # high score -> kept
    mgr.record_episode("bad one", "wrong", score=2.0)           # low score -> dropped
    st = lora.build_dataset(Config(), use_teacher=False)
    assert st.from_memory >= 1 and st.total == st.from_memory
    assert lora.dataset_size() == st.total
    # The low-scored exchange must not appear.
    text = lora.DATASET_FILE.read_text(encoding="utf-8")
    assert "2+2" in text and "bad one" not in text


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
