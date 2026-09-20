"""Format interpretation: the claim is 'read anything, and say how sure you are'.

These tests police the second half of that claim as hard as the first. A wrong
answer delivered confidently is worse than an honest "I'm not sure", so most of
what is asserted here is about *calibration*, not just parsing.
"""
from __future__ import annotations

import base64
import io
import json
import sqlite3
import struct
import tarfile
import zipfile

import pytest

from ag import formats


# --------------------------------------------------------------------------
# core dispatch
# --------------------------------------------------------------------------

def test_json_is_recognised_and_structured():
    i = formats.interpret(b'{"a": 1, "b": [1, 2, 3]}', filename="x.json")
    assert i.kind == "json"
    assert i.confidence >= 0.95          # a strict parse succeeded: near-certain
    assert i.structured["shape"]["a"] == "int"


def test_jsonl_is_distinguished_from_json():
    data = b'{"a": 1}\n{"a": 2}\n{"a": 3}\n'
    i = formats.interpret(data, filename="x.jsonl")
    assert i.kind == "jsonl"
    assert i.structured["records"] == 3


def test_csv_delimiter_is_inferred_not_assumed():
    for delim, name in ((b";", "semicolon"), (b"\t", "tab"), (b",", "comma")):
        body = b"name" + delim + b"age\nalice" + delim + b"30\nbob" + delim + b"41\n"
        i = formats.interpret(body, filename="people.csv")
        assert i.kind == "csv", name
        assert i.structured["columns"] == ["name", "age"], name


def test_ragged_csv_lowers_confidence():
    """A file that parses but parses *badly* must not claim certainty."""
    clean = b"a,b,c\n1,2,3\n4,5,6\n7,8,9\n"
    ragged = b"a,b,c\n1,2\n4,5,6,7,8\n9\n"
    assert formats.interpret(clean, filename="c.csv").confidence > \
           formats.interpret(ragged, filename="r.csv").confidence


def test_xml_and_html_are_separated():
    assert formats.interpret(b"<?xml version='1.0'?><r><a/></r>",
                             filename="f.xml").kind == "xml"
    h = formats.interpret(b"<!doctype html><html><body><p>hi</p></body></html>",
                          filename="f.html")
    assert h.kind == "html"
    assert "hi" in h.text


def test_zip_container_is_refined_to_ooxml():
    """A .docx IS a zip. Reporting 'zip archive' is technically true and useless."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("[Content_Types].xml", "<Types/>")
        z.writestr("word/document.xml",
                   "<w:document><w:p><w:t>hello world</w:t></w:p></w:document>")
    i = formats.interpret(buf.getvalue(), filename="report.docx")
    assert i.kind == "docx"
    assert "hello world" in i.text


def test_plain_zip_stays_a_zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("notes.txt", "hi")
    i = formats.interpret(buf.getvalue(), filename="bundle.zip")
    assert i.kind == "zip"
    assert "notes.txt" in i.text


def test_tar_and_gzip():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as t:
        info = tarfile.TarInfo("a.txt")
        payload = b"hello"
        info.size = len(payload)
        t.addfile(info, io.BytesIO(payload))
    assert formats.interpret(buf.getvalue(), filename="a.tar").kind == "tar"

    import gzip as _gz
    # The gzip handler reports what is INSIDE the container ("gzip:text"), which is
    # the useful answer; "gzip" alone would just name the wrapper.
    gz = formats.interpret(_gz.compress(b"hello" * 100), filename="a.gz")
    assert gz.kind.startswith("gzip")
    assert "hello" in gz.text


def test_sqlite_database(tmp_path):
    db = tmp_path / "t.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE people (id INTEGER, name TEXT)")
    con.execute("INSERT INTO people VALUES (1, 'alice')")
    con.commit()
    con.close()
    i = formats.read_path(str(db))
    assert i.kind == "sqlite"
    assert "people" in i.text


def test_png_dimensions_are_read_from_the_header():
    png = (b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR"
           + struct.pack(">II", 800, 600) + b"\x08\x06\x00\x00\x00" + b"\x00" * 4)
    i = formats.interpret(png, filename="x.png")
    assert i.kind == "png"
    assert "800" in i.text and "600" in i.text


def test_binary_without_any_signature_is_honest():
    """The important case: AG does not know. It must say so, not guess loudly."""
    blob = bytes(range(256)) * 8
    i = formats.interpret(blob, filename="mystery.bin")
    assert i.confidence < 0.6
    assert i.notes, "an unidentified blob must carry an explanatory note"


def test_unknown_binary_still_produces_usable_evidence():
    blob = bytes(range(256)) * 8
    i = formats.interpret(blob, filename="mystery.bin")
    assert i.text.strip(), "must still offer a hexdump/analysis, not an empty string"
    guess = formats.infer_unknown(blob)
    hyp = guess.structured.get("hypotheses") or []
    assert hyp, "infer_unknown must offer at least one ranked hypothesis"
    confs = [h["confidence"] for h in hyp]
    assert all(0.0 <= c <= 1.0 for c in confs)
    assert confs == sorted(confs, reverse=True), "hypotheses must be ranked"


def test_fixed_width_records_are_detected():
    """Interpreting a *new* format means finding structure with no magic number."""
    rec = struct.pack("<IIf", 1, 2, 3.5)
    guess = formats.infer_unknown(rec * 64)
    blob = (guess.text + " " + json.dumps(guess.structured, default=str)).lower()
    assert "record" in blob or "fixed" in blob or "struct" in blob


def test_base64_payload_is_decoded():
    # Padded to cross the 64-char evidence threshold: below that, "looks like the
    # base64 alphabet" is too weak a signal to act on.
    inner = json.dumps({"hello": "world", "pad": "x" * 60}).encode()
    i = formats.interpret(base64.b64encode(inner), filename="payload.txt")
    assert "base64" in (i.kind + " " + " ".join(i.notes)).lower()
    assert "world" in i.text, "must interpret the DECODED payload, not the envelope"


def test_prose_is_not_mistaken_for_base64():
    """The base64 alphabet matches ordinary lowercase text; decoding must confirm."""
    prose = b"hello there this is ordinary prose with no punctuation at all " * 3
    assert formats.interpret(prose, filename="a.txt").kind == "text"


def test_utf16_is_decoded_not_mangled():
    i = formats.interpret("héllo wörld".encode("utf-16"), filename="u.txt")
    assert "héllo" in i.text


def test_latin1_fallback_never_raises():
    i = formats.interpret(b"caf\xe9 na\xefve", filename="l.txt")
    assert "caf" in i.text


# --------------------------------------------------------------------------
# invariants
# --------------------------------------------------------------------------

@pytest.mark.parametrize("payload", [
    b"", b"\x00", b"\xff" * 10_000, b"{", b"{'not': json}",
    b"<html", b"PK\x03\x04truncated", b"\x89PNG\r\n\x1a\n", b"%PDF-1.4 broken",
    b"SQLite format 3\x00truncated", b"a" * 100_000, bytes(range(256)),
])
def test_interpret_never_raises(payload):
    """The contract the whole tool layer leans on: this function cannot throw.

    Every caller (read_any, the CLI, MCP, the server) treats interpretation as
    infallible. If it can raise on malformed input then hostile or corrupt files
    become crashes, and 'reads any format' becomes false at the worst moment.
    """
    i = formats.interpret(payload, filename="x.bin")
    assert isinstance(i.text, str)
    assert 0.0 <= i.confidence <= 1.0
    assert i.kind


def test_read_path_on_missing_file_is_an_error_not_an_exception():
    i = formats.read_path("/nonexistent/nope.txt")
    assert i.kind == "error"
    # Confidence describes the INTERPRETATION, and "this file does not exist" is a
    # certain one. Reporting 0.0 would wrongly suggest doubt about the failure.
    assert i.confidence == 1.0
    assert "nope.txt" in i.text or "not" in i.text.lower()


def test_read_path_on_a_directory_is_handled(tmp_path):
    i = formats.read_path(str(tmp_path))
    assert i.kind in ("error", "directory")


def test_large_file_is_truncated_with_a_flag(tmp_path):
    big = tmp_path / "big.txt"
    big.write_text("x" * (formats.MAX_READ_BYTES + 5000))
    i = formats.read_path(str(big))
    assert i.truncated
    assert len(i.text) <= formats.MAX_TEXT_CHARS + 2000


def test_confidence_is_always_in_range_across_formats():
    samples = [
        (b'{"a":1}', "a.json"), (b"a,b\n1,2\n", "a.csv"), (b"<x/>", "a.xml"),
        (b"hello", "a.txt"), (b"\x00\x01\x02", "a.bin"), (b"[s]\nk=v\n", "a.ini"),
    ]
    for data, name in samples:
        i = formats.interpret(data, filename=name)
        assert 0.0 <= i.confidence <= 1.0, name


def test_extension_lies_are_caught_by_content():
    """A .txt that is really a PNG should be reported as a PNG."""
    png = (b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR"
           + struct.pack(">II", 4, 4) + b"\x08\x06\x00\x00\x00" + b"\x00" * 4)
    i = formats.interpret(png, filename="notreally.txt")
    assert i.kind == "png", "content must outrank the extension"


def test_supported_formats_is_advertised():
    fmts = formats.supported_formats()
    assert len(fmts) > 15
    kinds = {f["kind"] for f in fmts}
    for key in ("json", "csv", "pdf", "sqlite", "zip"):
        assert key in kinds
    assert all(f.get("handler") for f in fmts)
