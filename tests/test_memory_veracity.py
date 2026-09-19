"""Tests for the veracity layer: what AG BELIEVES, as opposed to what it was told.

The property under test throughout is that memory grades its sources. A claim's
standing must depend on where it came from and what independently backs it — never on
how loudly or how often it was asserted. Hermetic and offline: the stdlib
HashingEmbedder stands in for Ollama.
"""
import time

import pytest

from ag import memory
from ag.memory import MemoryKind, MemoryManager, Origin
from ag.memory.embed import HashingEmbedder
from ag.memory.types import Memory, combine_confidence, prior_for


def mgr(tmp_path, agent="root", parents=()):
    return MemoryManager(None, agent=agent, parents=parents,
                         base_dir=tmp_path, embedder=HashingEmbedder())


def _days_ago(n: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() - n * 86400))


# --- priors: where a claim came from decides where it starts ----------------
def test_origin_sets_the_prior(tmp_path):
    m = mgr(tmp_path)
    said = m.remember("User cycles to work.", origin=Origin.USER)
    guessed = m.remember("User enjoys long meetings.", origin=Origin.INFERENCE)
    read = m.remember("Saturn has 146 moons.", origin=Origin.WEB)
    assert said.confidence > guessed.confidence > read.confidence
    assert said.confidence >= m.trust_threshold      # testimony is believed
    assert guessed.confidence < m.trust_threshold    # inference is a hypothesis
    assert read.confidence < m.trust_threshold       # so is anything a page asserted


def test_legacy_records_are_graded_not_trusted(tmp_path):
    """Memory written before veracity existed carries no confidence field. It must be
    graded by its source on load, not waved through at face value."""
    old = Memory.from_dict({"id": "x1", "text": "User prefers hands-on work.",
                            "kind": MemoryKind.SEMANTIC, "source": "reflect"})
    assert old.origin == Origin.INFERENCE
    assert old.confidence == pytest.approx(prior_for(Origin.INFERENCE))
    user_said = Memory.from_dict({"id": "x2", "text": "User is in Berlin.",
                                  "kind": MemoryKind.SEMANTIC, "source": "manual"})
    assert user_said.origin == Origin.USER
    assert user_said.confidence > old.confidence


# --- corroboration: repetition is not evidence ------------------------------
def test_repetition_from_the_same_origin_buys_no_belief(tmp_path):
    """The gullibility test. Hearing the same claim from the same place three times is
    one piece of evidence repeated, not three pieces of evidence."""
    m = mgr(tmp_path)
    first = m.remember("The build server is flaky.", origin=Origin.WEB)
    start = first.confidence
    for _ in range(3):
        again = m.remember("The build server is flaky.", origin=Origin.WEB)
    assert again.id == first.id                 # merged, not duplicated
    assert again.confidence == pytest.approx(start)
    assert len(again.evidence) == 1
    assert again.confidence < m.trust_threshold  # still not believed


def test_independent_origin_corroborates(tmp_path):
    m = mgr(tmp_path)
    claim = m.remember("The build server is flaky.", origin=Origin.WEB)
    assert claim.confidence < m.trust_threshold
    confirmed = m.remember("The build server is flaky.", origin=Origin.USER)
    assert confirmed.id == claim.id
    assert confirmed.confidence == pytest.approx(
        combine_confidence(prior_for(Origin.WEB), prior_for(Origin.USER)))
    assert confirmed.confidence >= m.trust_threshold  # two origins settle it
    assert {e["origin"] for e in confirmed.evidence} == {Origin.WEB, Origin.USER}


def test_weak_corroboration_accumulates_slowly(tmp_path):
    """Several weak, independent origins can add up — but not in one step."""
    m = mgr(tmp_path)
    a = m.remember("Trail closes at dusk.", origin=Origin.WEB)
    b = m.remember("Trail closes at dusk.", origin=Origin.INFERENCE)
    assert a.id == b.id
    assert b.confidence > a.confidence
    assert b.confidence < m.trust_threshold  # two weak sources are still not enough


# --- the recall gate --------------------------------------------------------
def test_disbelieved_memory_is_withheld_entirely(tmp_path):
    m = mgr(tmp_path)
    m.remember("A wildly unsupported claim about quasars.", confidence=0.05)
    assert m.recall("quasars") == []
    # ...but it is still on file, and a caller can lower the bar deliberately.
    assert m.recall("quasars", min_confidence=0.0)


def test_context_separates_known_from_merely_reported(tmp_path):
    m = mgr(tmp_path)
    m.remember("User writes Rust daily.", origin=Origin.USER)
    m.remember("Rust compiles faster than C in every case.", origin=Origin.WEB)
    ctx = m.context("tell me about rust", k=5)
    assert any("writes Rust daily" in t for t in ctx["known"])
    assert any("compiles faster" in t for t in ctx["reported"])
    assert "Unconfirmed" in ctx["text"]
    # The unconfirmed line carries its standing inline, so the executor cannot read it
    # as an established fact even if it skims the heading.
    assert "confidence" in ctx["text"] and Origin.WEB in ctx["text"]


def test_context_stays_plain_when_everything_is_established(tmp_path):
    m = mgr(tmp_path)
    m.remember("User prefers concise answers.", origin=Origin.USER)
    ctx = m.context("how should answers be written?")
    assert ctx["reported"] == []
    assert ctx["text"].startswith("- ")  # no ceremony when there is nothing to caveat


# --- contradiction ----------------------------------------------------------
def test_stable_contradiction_disputes_both_instead_of_picking(tmp_path):
    """Two claims that cannot both be true, about something that does not change. AG
    must not silently choose; it must lose confidence in both and say so."""
    m = mgr(tmp_path)
    a = m.remember("User uses metric units.", origin=Origin.USER)
    b = m.remember("User does not use metric units.", origin=Origin.INFERENCE)
    assert a.id != b.id
    a = m.store.get("root", a.id)
    assert a.disputed and b.disputed
    assert m.belief(a) < a.confidence            # demoted while unresolved
    assert {x.id for x in m.disputed()} == {a.id, b.id}
    assert b.meta.get("contradicts") == a.id


def test_verifying_settles_a_dispute(tmp_path):
    m = mgr(tmp_path)
    a = m.remember("User uses metric units.", origin=Origin.USER)
    m.remember("User does not use metric units.", origin=Origin.INFERENCE)
    settled = m.verify(a.id, by=Origin.USER)
    assert not settled.disputed
    assert m.belief(settled) >= m.trust_threshold


def test_volatile_contradiction_supersedes_rather_than_disputes(tmp_path):
    """Where the subject legitimately changes, the newer claim simply wins — flagging
    a dispute would punish AG for the world moving on."""
    m = mgr(tmp_path)
    old = m.remember("User is using version 3 of the toolchain.", origin=Origin.USER)
    new = m.remember("User is using version 4 of the toolchain.", origin=Origin.USER)
    assert new.id != old.id
    old = m.store.get("root", old.id)
    assert old.meta.get("superseded_by") == new.id
    assert not new.disputed and not old.disputed
    assert m.belief(old) < m.belief(new)


def test_two_preferences_are_not_a_contradiction(tmp_path):
    """The false-positive guard: knowing two things must never be punished."""
    m = mgr(tmp_path)
    a = m.remember("User prefers Python for scripting.", origin=Origin.USER)
    b = m.remember("User prefers Rust for systems work.", origin=Origin.USER)
    assert not a.disputed and not b.disputed
    assert m.disputed() == []


# --- decay and re-verification ---------------------------------------------
def test_volatile_belief_decays_toward_unknown(tmp_path):
    m = mgr(tmp_path)
    fact = m.remember("User currently works at Acme.", origin=Origin.USER)
    assert fact.volatile and m.belief(fact) >= m.trust_threshold
    fact.created = _days_ago(400)
    m.store.update(fact)
    fact = m.store.get("root", fact.id)
    assert m.belief(fact) < m.trust_threshold      # no longer asserted as current
    assert m.belief(fact) > 0.5                    # decays to doubt, never to "false"
    assert fact.id in {x.id for x in m.needs_verification()}


def test_stable_belief_does_not_decay(tmp_path):
    m = mgr(tmp_path)
    fact = m.remember("User was born in Lisbon.", origin=Origin.USER)
    fact.created = _days_ago(4000)
    m.store.update(fact)
    fact = m.store.get("root", fact.id)
    assert not fact.volatile
    assert m.belief(fact) == pytest.approx(fact.confidence)


def test_verify_resets_staleness_and_reject_collapses(tmp_path):
    m = mgr(tmp_path)
    fact = m.remember("User currently works at Acme.", origin=Origin.USER)
    fact.created = _days_ago(400)
    m.store.update(fact)
    refreshed = m.verify(fact.id, by=Origin.USER)
    assert m.belief(refreshed) >= m.trust_threshold
    assert refreshed.verified_by == Origin.USER

    wrong = m.remember("User owns a boat.", origin=Origin.INFERENCE)
    rejected = m.verify(wrong.id, by=Origin.USER, confirmed=False)
    assert rejected.confidence < m.confidence_floor


# --- retention and inheritance ---------------------------------------------
def test_doubtful_memory_is_forgotten_before_believed_memory(tmp_path):
    m = mgr(tmp_path)
    m.caps[MemoryKind.SEMANTIC] = 1
    m.remember("Some page claimed the office moved.", origin=Origin.WEB)
    kept = m.remember("User signs off on Fridays.", origin=Origin.USER)
    survivors = m.store.all("root", [MemoryKind.SEMANTIC])
    assert [x.id for x in survivors] == [kept.id]


def test_promotion_does_not_launder_a_belief(tmp_path):
    """Inheritance must not turn one agent's claim into the fleet's fact."""
    child = mgr(tmp_path, agent="child-09", parents=["root"])
    own = child.remember("The staging cluster reboots nightly.", origin=Origin.USER)
    child.promote(own.id, to="shared")
    shared = mgr(tmp_path, agent="shared")
    copy = [x for x in shared.store.all("shared") if "staging cluster" in x.text][0]
    assert copy.origin == Origin.AGENT
    assert copy.confidence < own.confidence


# --- reflection stores hypotheses, with real provenance ---------------------
class _FakeClient:
    def __init__(self, payload):
        self.payload = payload

    def complete(self, **kwargs):
        import json
        from types import SimpleNamespace
        return SimpleNamespace(text=json.dumps(self.payload))


def test_reflection_is_stored_as_unconfirmed(tmp_path):
    from ag.memory.reflect import reflect
    m = mgr(tmp_path)
    m.record_episode("convert 5 miles to km", "8.05 km", score=9.0)
    reflect(m, _FakeClient({"facts": ["User works in metric."], "procedures": []}),
            cfg=None)
    fact = m.store.all("root", [MemoryKind.SEMANTIC])[0]
    assert fact.origin == Origin.INFERENCE
    assert fact.confidence < m.trust_threshold  # a guess about someone is not knowledge


def test_reflection_links_to_the_episodes_it_cites(tmp_path):
    """Provenance must point at the evidence, not at whatever was most recent — a wrong
    edge is worse than none, because an audit would trust it."""
    from ag.memory.reflect import reflect
    m = mgr(tmp_path)
    e0 = m.record_episode("convert 5 miles to km", "8.05 km", score=9.0)
    m.record_episode("who won in 1998", "France", score=7.0)
    reflect(m, _FakeClient({"facts": [{"fact": "User works in metric.", "from": [0]}],
                            "procedures": []}), cfg=None)
    fact = m.store.all("root", [MemoryKind.SEMANTIC])[0]
    assert fact.links == [e0.id]


def test_reflection_without_citations_falls_back_to_best_match(tmp_path):
    from ag.memory.reflect import reflect
    m = mgr(tmp_path)
    m.record_episode("plan the Lisbon trip itinerary", "day one: Alfama", score=8.0)
    reflect(m, _FakeClient({"facts": ["User is planning a Lisbon trip itinerary."],
                            "procedures": []}), cfg=None)
    fact = m.store.all("root", [MemoryKind.SEMANTIC])[0]
    assert len(fact.links) == 1  # grounded by overlap rather than left dangling


# --- the tool surface the model itself can write through --------------------
def test_model_written_memory_cannot_claim_user_authority(tmp_path, monkeypatch):
    """The reasoning loop's `remember` tool is the model asserting something. It must
    not be able to write itself a high-confidence belief about the user."""
    from ag import config as cfgmod
    from ag.memory import manager as mgrmod
    from ag.tools import local
    monkeypatch.setattr(cfgmod, "STATE_DIR", tmp_path)
    monkeypatch.setattr(mgrmod, "get_embedder", lambda cfg=None, **k: HashingEmbedder())
    memory.reset()
    out = local.memory_remember("User is an expert in quantum optics.")
    assert "unconfirmed" in out
    stored = memory.all_memories()[0]
    assert stored.origin == Origin.INFERENCE
    assert stored.confidence < memory.get_manager("root").trust_threshold
    # ...and recall hands it back labelled, not as plain fact.
    assert "[unconfirmed" in local.memory_recall("quantum optics")
    memory.reset()
