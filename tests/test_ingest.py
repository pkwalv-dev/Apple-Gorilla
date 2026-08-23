"""Tests for claude.ai export ingestion."""
import json
import zipfile

from ag.config import Config
from ag.ingest import (build_corpus, distill, extract_human_messages,
                       _messages_from_obj)
from ag.model import make_client

SAMPLE = [
    {
        "name": "chat1",
        "chat_messages": [
            {"sender": "human", "text": "I'm a founder building AI dev tools."},
            {"sender": "assistant", "text": "Great, tell me more."},
            {"sender": "human", "text": "Explain RAG vs fine-tuning briefly."},
        ],
    }
]


def test_messages_from_obj_list_form():
    msgs = _messages_from_obj(SAMPLE)
    assert msgs == [
        "I'm a founder building AI dev tools.",
        "Explain RAG vs fine-tuning briefly.",
    ]  # assistant messages excluded


def test_messages_from_obj_content_parts():
    obj = [{"chat_messages": [
        {"sender": "human", "content": [{"type": "text", "text": "hello world"}]},
    ]}]
    assert _messages_from_obj(obj) == ["hello world"]


def test_extract_from_json_file(tmp_path):
    f = tmp_path / "conversations.json"
    f.write_text(json.dumps(SAMPLE))
    assert len(extract_human_messages(f)) == 2


def test_extract_from_zip(tmp_path):
    z = tmp_path / "export.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("conversations.json", json.dumps(SAMPLE))
    assert len(extract_human_messages(z)) == 2


def test_extract_from_dir(tmp_path):
    (tmp_path / "conversations.json").write_text(json.dumps(SAMPLE))
    assert len(extract_human_messages(tmp_path)) == 2


def test_build_corpus_respects_budget():
    msgs = ["x" * 100 for _ in range(50)]
    corpus = build_corpus(msgs, max_chars=250)
    assert len(corpus) <= 260  # a couple messages + separators, not all 50


def test_distill_dry_run_returns_markdown():
    md = distill(make_client(dry_run=True), Config(), ["I do data science."])
    assert md.startswith("# About Me")
    assert "## Standing preferences" in md
