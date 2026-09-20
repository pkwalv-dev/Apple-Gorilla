"""Universal format layer — read anything, and reason about what you can't.

AG's file access was `path.read_text(encoding="utf-8", errors="replace")`. That is
fine for source code and catastrophic for everything else: a PDF, a spreadsheet, a
SQLite database, or a zip arrives as replacement characters, and the model then
confidently reasons about mojibake. This module is the fix, in two halves.

**Half one — handlers for formats we know.** A sniff step (magic bytes first, then
extension, then content shape) selects a handler, and the handler returns a
`Interpretation`: a text rendering the model can actually read, plus structured
metadata. Everything is stdlib: OOXML is a zip of XML, so `.docx`/`.xlsx`/`.pptx`
are parsed directly; PDF text comes out of the content streams via `zlib`; image
dimensions come from header parsing, not Pillow. No new dependency, which is what
keeps AG portable to the machine that has nothing installed.

**Half two — inference for formats we don't.** When nothing matches, AG does not
shrug. `infer_unknown()` runs structural forensics — encoding and BOM detection,
printable ratio, Shannon entropy (compressed/encrypted vs. structured vs. text),
magic-prefix capture, line-structure and delimiter histograms, fixed-record-size
detection by scoring divisors of the file length, and repeated-token dictionary
extraction — and returns a *hypothesis with a confidence*, plus the evidence behind
it. That is the honest form of "interpret formats that don't exist yet": not a
claim to understand, but a structured description good enough for the model to
reason from, and good enough to tell whether it should ask for guidance or author a
handler skill instead.

Design rules:
  - Never raise. A handler that fails degrades to the next candidate, and the last
    candidate is always the inference path. Reading a file must not be able to kill
    a run.
  - Always bound. Every reader takes a byte budget; nothing loads a 4GB file to
    describe it.
  - Always report confidence. A caller must be able to tell "I parsed this" from
    "I guessed at this".
"""
from __future__ import annotations

import binascii
import io
import json
import math
import re
import struct
import zlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# Read budgets. Sniffing needs almost nothing; full interpretation is capped so a
# huge file costs a bounded amount of memory and a bounded amount of context.
SNIFF_BYTES = 8192
MAX_READ_BYTES = 8 * 1024 * 1024
MAX_TEXT_CHARS = 20000


@dataclass
class Interpretation:
    """What AG made of a blob of bytes.

    `text` is what goes to the model. `confidence` is how much the caller should
    trust `kind`: 1.0 means a magic number matched and the parse succeeded, 0.2
    means this is a structural guess from entropy and byte histograms.
    """

    kind: str                       # "pdf", "sqlite", "csv", "unknown:binary", ...
    label: str                      # human-readable name of the format
    text: str                       # model-readable rendering
    confidence: float = 1.0         # 0..1 — how sure we are of `kind`
    structured: dict = field(default_factory=dict)  # parsed metadata/content
    n_bytes: int = 0
    truncated: bool = False
    handler: str = ""               # which handler produced this
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["text"] = self.text[:MAX_TEXT_CHARS]
        return d

    def summary(self) -> str:
        """One-line description, for logs and tool observations."""
        conf = f"{self.confidence:.2f}"
        t = f"{self.label} ({self.kind}, confidence {conf}, {self.n_bytes} bytes)"
        return t + (" [truncated]" if self.truncated else "")


# --------------------------------------------------------------------------- #
# Magic-number table. Ordered longest-prefix-first at match time, so a generic
# prefix never shadows a specific one (e.g. a zip prefix vs. an OOXML zip).
# --------------------------------------------------------------------------- #
MAGIC: Tuple[Tuple[bytes, str, str], ...] = (
    (b"%PDF-", "pdf", "PDF document"),
    (b"SQLite format 3\x00", "sqlite", "SQLite database"),
    (b"PK\x03\x04", "zip", "ZIP archive"),
    (b"PK\x05\x06", "zip", "ZIP archive (empty)"),
    (b"\x1f\x8b", "gzip", "gzip stream"),
    (b"BZh", "bzip2", "bzip2 stream"),
    (b"\xfd7zXZ\x00", "xz", "xz stream"),
    (b"\x28\xb5\x2f\xfd", "zstd", "zstandard stream"),
    (b"7z\xbc\xaf\x27\x1c", "7z", "7-Zip archive"),
    (b"Rar!\x1a\x07", "rar", "RAR archive"),
    (b"\x89PNG\r\n\x1a\n", "png", "PNG image"),
    (b"\xff\xd8\xff", "jpeg", "JPEG image"),
    (b"GIF87a", "gif", "GIF image"),
    (b"GIF89a", "gif", "GIF image"),
    (b"BM", "bmp", "BMP image"),
    (b"II*\x00", "tiff", "TIFF image"),
    (b"MM\x00*", "tiff", "TIFF image"),
    (b"\x00\x00\x01\x00", "ico", "Windows icon"),
    (b"<?xml", "xml", "XML document"),
    (b"\x7fELF", "elf", "ELF executable"),
    (b"MZ", "pe", "Windows PE executable"),
    (b"\xca\xfe\xba\xbe", "macho-fat", "Mach-O universal binary"),
    (b"\xcf\xfa\xed\xfe", "macho", "Mach-O executable"),
    (b"\xce\xfa\xed\xfe", "macho", "Mach-O executable (32-bit)"),
    (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", "ole2", "Legacy MS Office (OLE2)"),
    (b"RIFF", "riff", "RIFF container"),
    (b"OggS", "ogg", "Ogg container"),
    (b"fLaC", "flac", "FLAC audio"),
    (b"ID3", "mp3", "MP3 audio"),
    (b"\x1a\x45\xdf\xa3", "matroska", "Matroska/WebM container"),
    (b"\xed\xab\xee\xdb", "rpm", "RPM package"),
    (b"!<arch>", "ar", "ar archive (.deb/.a)"),
    (b"\xd4\xc3\xb2\xa1", "pcap", "pcap capture"),
    (b"\x0a\x0d\x0d\x0a", "pcapng", "pcapng capture"),
    (b"\x93NUMPY", "npy", "NumPy array"),
    (b"\x08", "", ""),   # sentinel: never matches meaningfully, keeps tuple non-empty
)

EXT_HINTS: Dict[str, str] = {
    ".txt": "text", ".log": "text", ".text": "text", ".rst": "text",
    ".md": "markdown", ".markdown": "markdown",
    ".json": "json", ".jsonl": "jsonl", ".ndjson": "jsonl", ".geojson": "json",
    ".csv": "csv", ".tsv": "tsv", ".psv": "delimited",
    ".xml": "xml", ".svg": "xml", ".rss": "xml", ".atom": "xml", ".xsd": "xml",
    ".html": "html", ".htm": "html", ".xhtml": "html",
    ".yaml": "yaml", ".yml": "yaml",
    ".toml": "toml", ".ini": "ini", ".cfg": "ini", ".conf": "ini",
    ".env": "dotenv", ".properties": "ini",
    ".py": "source", ".js": "source", ".ts": "source", ".tsx": "source",
    ".jsx": "source", ".rs": "source", ".go": "source", ".c": "source",
    ".h": "source", ".cpp": "source", ".hpp": "source", ".java": "source",
    ".rb": "source", ".php": "source", ".sh": "source", ".bash": "source",
    ".zsh": "source", ".ps1": "source", ".bat": "source", ".lua": "source",
    ".sql": "source", ".r": "source", ".swift": "source", ".kt": "source",
    ".scala": "source", ".pl": "source", ".vim": "source", ".el": "source",
    ".docx": "docx", ".xlsx": "xlsx", ".pptx": "pptx",
    ".pdf": "pdf", ".sqlite": "sqlite", ".db": "sqlite", ".sqlite3": "sqlite",
    ".zip": "zip", ".tar": "tar", ".gz": "gzip", ".tgz": "gzip", ".bz2": "bzip2",
    ".wav": "wav", ".mp3": "mp3", ".mp4": "mp4", ".m4a": "mp4", ".mov": "mp4",
    ".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg", ".gif": "gif", ".bmp": "bmp",
    ".webp": "webp", ".ico": "ico", ".tiff": "tiff", ".tif": "tiff",
    ".parquet": "parquet", ".avro": "avro", ".pkl": "pickle", ".pickle": "pickle",
    ".npy": "npy", ".npz": "zip", ".b64": "base64", ".pem": "pem", ".crt": "pem",
}

# Extensions whose content is plain text and should be read as such.
TEXTUAL_KINDS = frozenset({
    "text", "markdown", "source", "json", "jsonl", "csv", "tsv", "delimited",
    "xml", "html", "yaml", "toml", "ini", "dotenv", "pem", "base64",
})


# --------------------------------------------------------------------------- #
# Byte-level primitives
# --------------------------------------------------------------------------- #

def shannon_entropy(data: bytes) -> float:
    """Bits per byte, 0..8. ~0 = uniform, ~4-5 = text, >7.5 = compressed/encrypted.

    This is the single most informative number about an unknown blob: it separates
    "there is structure here to find" from "this is a compressed container and the
    bytes are meaningless without decompressing".
    """
    if not data:
        return 0.0
    counts = [0] * 256
    for b in data:
        counts[b] += 1
    n = len(data)
    ent = 0.0
    for c in counts:
        if c:
            p = c / n
            ent -= p * math.log2(p)
    return round(ent, 3)


def printable_ratio(data: bytes) -> float:
    """Fraction of bytes that are printable ASCII, tab, LF, or CR."""
    if not data:
        return 0.0
    ok = sum(1 for b in data if 32 <= b < 127 or b in (9, 10, 13))
    return round(ok / len(data), 4)


def detect_encoding(data: bytes) -> Tuple[str, str]:
    """Return (encoding, how_we_know). BOM first, then decode trials.

    Deliberately conservative: we only claim an encoding we could actually decode
    with, because the caller will use it and a wrong answer produces mojibake the
    model cannot detect.
    """
    boms = (
        (b"\xef\xbb\xbf", "utf-8-sig", "UTF-8 BOM"),
        (b"\xff\xfe\x00\x00", "utf-32-le", "UTF-32 LE BOM"),
        (b"\x00\x00\xfe\xff", "utf-32-be", "UTF-32 BE BOM"),
        (b"\xff\xfe", "utf-16-le", "UTF-16 LE BOM"),
        (b"\xfe\xff", "utf-16-be", "UTF-16 BE BOM"),
    )
    for bom, enc, how in boms:
        if data.startswith(bom):
            return enc, how
    try:
        data.decode("utf-8")
        return "utf-8", "clean decode"
    except (UnicodeDecodeError, LookupError):
        pass
    # UTF-16 WITHOUT a BOM is only proposed on positive evidence: real UTF-16 text
    # in the Latin range carries one NUL per character, so the NUL density is near
    # 50%. Testing "did it decode?" alone is not evidence at all — any even-length
    # byte string decodes as UTF-16 into *something*, which silently turned
    # `café naïve` into CJK mojibake. Requiring the interleaving pattern first
    # means we only claim UTF-16 when the bytes actually look like it.
    if len(data) >= 4 and len(data) % 2 == 0:
        nul_even = sum(1 for i in range(0, min(len(data), 512), 2) if data[i] == 0)
        nul_odd = sum(1 for i in range(1, min(len(data), 512), 2) if data[i] == 0)
        half = max(1, min(len(data), 512) // 2)
        for enc, count in (("utf-16-le", nul_odd), ("utf-16-be", nul_even)):
            if count / half < 0.3:
                continue
            try:
                text = data.decode(enc)
            except (UnicodeDecodeError, LookupError):
                continue
            if text.count("\x00") <= len(text) * 0.05:
                return enc, "NUL-interleaved decode"
    # Latin-1 always "works" — say so honestly rather than implying detection.
    return "latin-1", "fallback (no clean decode; bytes preserved 1:1)"


def hexdump(data: bytes, *, width: int = 16, max_lines: int = 24) -> str:
    """Classic offset/hex/ASCII dump — the universal last resort rendering."""
    out = []
    for i in range(0, min(len(data), width * max_lines), width):
        chunk = data[i:i + width]
        hx = " ".join(f"{b:02x}" for b in chunk).ljust(width * 3 - 1)
        asc = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        out.append(f"{i:08x}  {hx}  |{asc}|")
    if len(data) > width * max_lines:
        out.append(f"... ({len(data) - width * max_lines} more bytes)")
    return "\n".join(out)


def _decode(data: bytes) -> str:
    enc, _ = detect_encoding(data)
    try:
        return data.decode(enc, errors="replace")
    except LookupError:
        return data.decode("utf-8", errors="replace")


def _clip(text: str, limit: int = MAX_TEXT_CHARS) -> Tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit] + f"\n…[truncated at {limit} chars]", True


# --------------------------------------------------------------------------- #
# Sniffing: magic bytes -> extension -> content shape
# --------------------------------------------------------------------------- #

def _is_textual(data: bytes) -> bool:
    """Is this text in SOME encoding?

    `printable_ratio` alone answers "is this text in an 8-bit encoding?" — UTF-16
    and UTF-32 fail it by construction, since every other byte of Latin-range text
    is a NUL. Asking the question after a decode attempt is what makes the answer
    encoding-independent, so a Windows UTF-16 .txt is treated as the text it is.
    """
    if not data:
        return True
    if printable_ratio(data) >= 0.7:
        return True
    enc, _ = detect_encoding(data)
    if enc in ("latin-1", "utf-8"):
        return False
    try:
        decoded = data.decode(enc, errors="ignore")
    except (LookupError, UnicodeDecodeError):
        return False
    if not decoded.strip():
        return False
    printable = sum(1 for ch in decoded if ch.isprintable() or ch in "\r\n\t")
    return printable / len(decoded) >= 0.85


def sniff(data: bytes, *, filename: str = "") -> Tuple[str, str, float]:
    """Identify a format. Returns (kind, label, confidence).

    Magic numbers are near-certain (0.95+); an extension alone is a hint (0.6);
    content-shape inference on text is moderate (0.5-0.8).
    """
    # 1. Magic numbers — sorted longest-first so specific beats generic.
    for prefix, kind, label in sorted(MAGIC, key=lambda m: -len(m[0])):
        if not kind or not prefix:
            continue
        if data.startswith(prefix):
            refined = _refine_container(data, kind, filename)
            if refined:
                return refined
            return kind, label, 0.97

    # 2. Extension.
    ext = Path(filename).suffix.lower() if filename else ""
    if ext in EXT_HINTS:
        kind = EXT_HINTS[ext]
        # Verify a textual claim against the bytes, so a mislabeled .txt that is
        # actually binary does not get decoded as text.
        if kind in TEXTUAL_KINDS and not _is_textual(data[:SNIFF_BYTES]):
            return "unknown", f"binary data with a {ext} extension", 0.3
        shaped = _shape_of_text(data) if kind in ("text", "source") else None
        if shaped and shaped[0] not in ("text",):
            return shaped[0], shaped[1], 0.75
        return kind, f"{ext.lstrip('.') or 'file'} file", 0.62

    # 3. Content shape, for extensionless text.
    if printable_ratio(data[:SNIFF_BYTES]) >= 0.85:
        shaped = _shape_of_text(data)
        if shaped:
            return shaped[0], shaped[1], shaped[2]
        return "text", "plain text", 0.7

    # 3b. UTF-16/32 text fails the printable-ratio test, because every other byte is
    # a NUL. Judging "is this text?" on raw bytes only works for 8-bit encodings; for
    # wide encodings the question has to be asked after decoding. Without this, every
    # UTF-16 file on Windows — a very large share of the world's text — was reported
    # as "unrecognised binary".
    if _is_textual(data[:SNIFF_BYTES]):
        enc, how = detect_encoding(data[:SNIFF_BYTES])
        shaped = _shape_of_text(data)
        if shaped:
            return shaped[0], shaped[1], shaped[2]
        return "text", f"plain text ({enc}, {how})", 0.8

    return "unknown", "unrecognised binary", 0.15


def _refine_container(data: bytes, kind: str, filename: str) -> Optional[Tuple[str, str, float]]:
    """A zip may be a .docx; a RIFF may be a .wav. Look inside before answering."""
    if kind == "zip":
        head = data[:4096]
        if b"word/document.xml" in head or b"word/" in head:
            return "docx", "Word document (OOXML)", 0.95
        if b"xl/workbook.xml" in head or b"xl/" in head:
            return "xlsx", "Excel workbook (OOXML)", 0.95
        if b"ppt/presentation.xml" in head or b"ppt/" in head:
            return "pptx", "PowerPoint deck (OOXML)", 0.95
        if b"META-INF/container.xml" in head or b"mimetypeapplication/epub" in head:
            return "epub", "EPUB book", 0.9
        if b"AndroidManifest.xml" in head:
            return "apk", "Android package", 0.9
        if b"META-INF/MANIFEST.MF" in head:
            return "jar", "Java archive", 0.9
        ext = Path(filename).suffix.lower()
        if ext in (".odt", ".ods", ".odp"):
            return "opendoc", "OpenDocument file", 0.9
        return None
    if kind == "riff":
        if data[8:12] == b"WAVE":
            return "wav", "WAV audio", 0.96
        if data[8:12] == b"WEBP":
            return "webp", "WebP image", 0.96
        if data[8:12] == b"AVI ":
            return "avi", "AVI video", 0.96
        return None
    if kind == "xml":
        head = data[:2048].lower()
        if b"<svg" in head:
            return "xml", "SVG image (XML)", 0.95
        if b"<html" in head or b"<!doctype html" in head:
            return "html", "HTML document", 0.95
    return None


def _shape_of_text(data: bytes) -> Optional[Tuple[str, str, float]]:
    """Infer a structured text format from its shape alone (no extension)."""
    text = _decode(data[:SNIFF_BYTES]).strip()
    if not text:
        return ("text", "empty or whitespace-only text", 0.9)
    head = text[:4]
    # JSON / JSONL
    if head[:1] in ("{", "["):
        try:
            json.loads(text if len(data) <= SNIFF_BYTES else text[:text.rfind("}") + 1])
            return ("json", "JSON document", 0.9)
        except Exception:
            pass
    lines = [ln for ln in text.splitlines() if ln.strip()][:20]
    if lines and all(ln.lstrip().startswith(("{", "[")) for ln in lines):
        ok = 0
        for ln in lines:
            try:
                json.loads(ln)
                ok += 1
            except Exception:
                pass
        if ok >= max(2, len(lines) // 2):
            return ("jsonl", "JSON Lines / NDJSON", 0.85)
    if text.lstrip().startswith("<"):
        return ("xml", "markup document", 0.7)
    if text.lstrip().startswith("---") and ":" in text:
        return ("yaml", "YAML document", 0.7)
    if re.match(r"^\s*\[[^\]\n]+\]\s*$", lines[0] if lines else "", re.M):
        return ("ini", "INI/config file", 0.7)
    # Delimited data: a consistent separator count across lines is a strong signal.
    delim = _guess_delimiter(lines)
    if delim:
        name = {",": "CSV", "\t": "TSV", ";": "semicolon-delimited",
                "|": "pipe-delimited"}.get(delim, "delimited")
        kind = {",": "csv", "\t": "tsv"}.get(delim, "delimited")
        return (kind, f"{name} data", 0.8)
    if re.search(r"^#{1,6}\s+\S", text, re.M) or re.search(r"^[-*]\s+\S", text, re.M):
        return ("markdown", "Markdown document", 0.65)
    # Base64: a long run of the alphabet and nothing else — but that pattern also
    # matches ordinary lowercase prose ("hellohellohello…" is a valid base64 string),
    # which previously hijacked plain text into a garbage decode. So the claim has to
    # be *verified*: decode it and require the result to be recognisable.
    compact = re.sub(r"\s", "", text)
    if (len(compact) >= 64 and len(compact) % 4 == 0
            and re.fullmatch(r"[A-Za-z0-9+/]+={0,2}", compact)
            and _base64_decodes_to_something(compact)):
        return ("base64", "base64-encoded data", 0.75)
    return None


def _base64_decodes_to_something(compact: str) -> bool:
    """Does decoding this actually yield recognisable content?

    Evidence for the base64 hypothesis, not just consistency with it. Encoded
    payloads decode to a known magic number or clean readable text; lowercase prose
    that merely *looks* like base64 decodes to high-entropy noise, which fails here.
    """
    try:
        raw = binascii.a2b_base64(compact.encode("ascii", "ignore"))
    except Exception:
        return False
    if len(raw) < 8:
        return False
    for prefix, _kind, _label in MAGIC:
        if prefix and raw.startswith(prefix):
            return True
    head = raw[:512]
    if printable_ratio(head) >= 0.9:
        try:
            head.decode("utf-8")
            return True
        except UnicodeDecodeError:
            return False
    return False


def _guess_delimiter(lines: List[str]) -> Optional[str]:
    """Pick the delimiter whose per-line count is both high and consistent."""
    if len(lines) < 2:
        return None
    best, best_score = None, 0.0
    for d in (",", "\t", ";", "|"):
        counts = [ln.count(d) for ln in lines]
        if min(counts) < 1:
            continue
        mean = sum(counts) / len(counts)
        # Consistency: all lines having the same field count is what makes it a table.
        consistent = sum(1 for c in counts if c == counts[0]) / len(counts)
        score = consistent * min(mean, 10)
        # A two-column table has only ONE delimiter per line, so a raw score
        # threshold of 1.5 structurally excluded every 2-column CSV/TSV — a very
        # common shape. Perfect consistency across several lines is strong enough
        # evidence on its own; requiring 3+ lines keeps a one-off "hello, world"
        # from being read as a table.
        if consistent == 1.0 and counts[0] >= 1 and len(lines) >= 3:
            score = max(score, 1.5)
        if score > best_score:
            best, best_score = d, score
    return best if best_score >= 1.5 else None


# --------------------------------------------------------------------------- #
# Handlers. Each takes raw bytes and returns an Interpretation.
# --------------------------------------------------------------------------- #

def _h_text(data: bytes, kind: str, label: str, name: str) -> Interpretation:
    enc, how = detect_encoding(data)
    text, trunc = _clip(_decode(data))
    lines = text.count("\n") + 1
    return Interpretation(
        kind=kind, label=label, text=text, confidence=1.0,
        structured={"encoding": enc, "encoding_evidence": how, "lines": lines,
                    "chars": len(text)},
        n_bytes=len(data), truncated=trunc, handler="text")


def _h_json(data: bytes, kind: str, label: str, name: str) -> Interpretation:
    raw = _decode(data)
    try:
        obj = json.loads(raw)
    except Exception as e:
        # Malformed JSON is still useful as text, but say so — the model must not
        # treat a broken document as authoritative structure.
        text, trunc = _clip(raw)
        return Interpretation(kind="json", label="JSON (malformed)", text=text,
                              confidence=0.6, structured={"parse_error": str(e)},
                              n_bytes=len(data), truncated=trunc, handler="json",
                              notes=[f"JSON did not parse: {e}"])
    shape = _json_shape(obj)
    pretty, trunc = _clip(json.dumps(obj, indent=2, ensure_ascii=False, default=str))
    return Interpretation(
        kind="json", label="JSON document", text=pretty, confidence=1.0,
        structured={"shape": shape, "top_level": type(obj).__name__},
        n_bytes=len(data), truncated=trunc, handler="json")


def _json_shape(obj, depth: int = 0, max_depth: int = 4):
    """A compact schema of a JSON value — the part a model actually needs."""
    if depth >= max_depth:
        return "…"
    if isinstance(obj, dict):
        return {k: _json_shape(v, depth + 1, max_depth)
                for k, v in list(obj.items())[:40]}
    if isinstance(obj, list):
        if not obj:
            return []
        return [_json_shape(obj[0], depth + 1, max_depth), f"…×{len(obj)}"]
    return type(obj).__name__


def _h_jsonl(data: bytes, kind: str, label: str, name: str) -> Interpretation:
    raw = _decode(data)
    rows, bad = [], 0
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            bad += 1
    preview = json.dumps(rows[:20], indent=2, ensure_ascii=False, default=str)
    text, trunc = _clip(f"{len(rows)} records ({bad} unparseable)\n\n{preview}")
    keys: set = set()
    for r in rows[:200]:
        if isinstance(r, dict):
            keys |= set(r.keys())
    return Interpretation(
        kind="jsonl", label="JSON Lines", text=text, confidence=1.0,
        structured={"records": len(rows), "unparseable": bad,
                    "keys": sorted(keys)[:60]},
        n_bytes=len(data), truncated=trunc, handler="jsonl")


def _h_delimited(data: bytes, kind: str, label: str, name: str) -> Interpretation:
    import csv as _csv
    raw = _decode(data)
    sample_lines = [ln for ln in raw.splitlines() if ln.strip()][:50]
    # Measure the delimiter from the data first. ".csv" is a claim about the file's
    # role, not its syntax: European exports are routinely semicolon-separated and
    # still named .csv, and forcing "," on those produced a single fused column
    # ("name;age") that looked like a successful parse. Evidence outranks the
    # extension; the extension is only the fallback when the data is inconclusive.
    delim = (_guess_delimiter(sample_lines)
             or {"csv": ",", "tsv": "\t"}.get(kind)
             or ",")
    try:
        rows = list(_csv.reader(io.StringIO(raw), delimiter=delim))
    except Exception:
        rows = [ln.split(delim) for ln in raw.splitlines()]
    rows = [r for r in rows if r]
    header = rows[0] if rows else []
    body = rows[1:]
    # Column typing: the single most useful thing to tell a model about a table.
    types = []
    for i in range(len(header)):
        col = [r[i] for r in body[:200] if i < len(r) and r[i] != ""]
        types.append(_column_type(col))
    lines = [f"{len(rows)} rows × {len(header)} columns (delimiter {delim!r})", ""]
    lines.append(" | ".join(f"{h} <{t}>" for h, t in zip(header, types)))
    lines.append("-" * 60)
    for r in body[:50]:
        lines.append(" | ".join(str(c) for c in r))
    if len(body) > 50:
        lines.append(f"… {len(body) - 50} more rows")
    text, trunc = _clip("\n".join(lines))
    # Earn the confidence from the data: a real table has the same number of fields
    # on (almost) every row. A csv reader "succeeds" on any text, so ragged rows
    # mean this probably is not tabular data and the caller must know that.
    if body and header:
        consistent = sum(1 for r in body if len(r) == len(header)) / len(body)
    else:
        consistent = 0.0
    conf = 0.95 if consistent >= 0.98 else (0.8 if consistent >= 0.85 else 0.55)
    notes = ([] if consistent >= 0.85 else
             [f"only {consistent:.0%} of rows have the header's field count — this "
              f"may not really be {delim!r}-delimited data"])
    return Interpretation(
        kind=kind or "delimited", label=label, text=text, confidence=conf,
        structured={"delimiter": delim, "columns": header, "column_types": types,
                    "rows": len(body), "row_consistency": round(consistent, 3)},
        n_bytes=len(data), truncated=trunc, handler="delimited", notes=notes)


def _column_type(values: List[str]) -> str:
    if not values:
        return "empty"
    def _is_int(v):
        try:
            int(v.strip())
            return True
        except Exception:
            return False
    def _is_float(v):
        try:
            float(v.strip())
            return True
        except Exception:
            return False
    if all(_is_int(v) for v in values):
        return "int"
    if all(_is_float(v) for v in values):
        return "float"
    if all(re.fullmatch(r"\d{4}-\d{2}-\d{2}([T ].*)?", v.strip()) for v in values):
        return "date"
    if all(v.strip().lower() in ("true", "false", "yes", "no", "0", "1")
           for v in values):
        return "bool"
    return "str"


def _h_xml(data: bytes, kind: str, label: str, name: str) -> Interpretation:
    import xml.etree.ElementTree as ET
    raw = _decode(data)
    try:
        root = ET.fromstring(raw)
    except Exception as e:
        text, trunc = _clip(raw)
        return Interpretation(kind="xml", label="XML (malformed)", text=text,
                              confidence=0.6, structured={"parse_error": str(e)},
                              n_bytes=len(data), truncated=trunc, handler="xml",
                              notes=[f"XML did not parse: {e}"])
    counts: Dict[str, int] = {}
    for el in root.iter():
        tag = el.tag.split("}")[-1]
        counts[tag] = counts.get(tag, 0) + 1
    texts = [t.strip() for t in root.itertext() if t.strip()]
    body = "\n".join(texts)
    outline = "\n".join(f"  <{t}> ×{c}" for t, c in
                        sorted(counts.items(), key=lambda kv: -kv[1])[:40])
    text, trunc = _clip(f"root: <{root.tag.split('}')[-1]}>\nelements:\n{outline}\n\n"
                        f"--- text content ---\n{body}")
    return Interpretation(
        kind="xml", label=label, text=text, confidence=1.0,
        structured={"root": root.tag, "element_counts": counts,
                    "attributes": dict(list(root.attrib.items())[:20])},
        n_bytes=len(data), truncated=trunc, handler="xml")


def _h_html(data: bytes, kind: str, label: str, name: str) -> Interpretation:
    raw = _decode(data)
    title = ""
    m = re.search(r"<title[^>]*>(.*?)</title>", raw, re.I | re.S)
    if m:
        title = re.sub(r"\s+", " ", m.group(1)).strip()
    links = re.findall(r'<a[^>]+href="([^"]+)"', raw, re.I)[:60]
    heads = [re.sub(r"<[^>]+>", "", h).strip()
             for h in re.findall(r"<h[1-3][^>]*>(.*?)</h[1-3]>", raw, re.I | re.S)][:40]
    stripped = re.sub(r"(?is)<(script|style|noscript).*?</\1>", " ", raw)
    body = re.sub(r"\s+", " ", re.sub(r"(?s)<[^>]+>", " ", stripped)).strip()
    import html as _html
    body = _html.unescape(body)
    parts = [f"title: {title}" if title else "", ]
    if heads:
        parts.append("headings:\n" + "\n".join(f"  - {h}" for h in heads))
    parts.append("--- text ---\n" + body)
    text, trunc = _clip("\n".join(p for p in parts if p))
    return Interpretation(
        kind="html", label="HTML document", text=text, confidence=1.0,
        structured={"title": title, "headings": heads, "links": links[:40],
                    "n_links": len(links)},
        n_bytes=len(data), truncated=trunc, handler="html")


def _h_ini(data: bytes, kind: str, label: str, name: str) -> Interpretation:
    import configparser
    raw = _decode(data)
    cp = configparser.ConfigParser(strict=False, interpolation=None,
                                   allow_no_value=True)
    parsed: Dict[str, dict] = {}
    try:
        cp.read_string(raw)
        parsed = {s: dict(cp.items(s)) for s in cp.sections()}
        if cp.defaults():
            parsed["DEFAULT"] = dict(cp.defaults())
    except Exception:
        # Key=value with no section header is extremely common (.env, .properties).
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith(("#", ";")) or "=" not in line:
                continue
            k, v = line.split("=", 1)
            parsed.setdefault("(no section)", {})[k.strip()] = v.strip()
    text, trunc = _clip(json.dumps(parsed, indent=2, default=str) or raw)
    return Interpretation(
        kind="ini", label=label, text=text, confidence=0.9,
        structured={"sections": list(parsed.keys()), "values": parsed},
        n_bytes=len(data), truncated=trunc, handler="ini")


def _h_yaml(data: bytes, kind: str, label: str, name: str) -> Interpretation:
    """YAML without PyYAML: parse the common subset, and say so.

    Honest scope — this reads nested mappings, sequences, and scalars with
    indentation. Anchors, multi-line blocks, and flow-style nesting are left as
    text. The model is told which it got, so it never assumes a full parse.
    """
    raw = _decode(data)
    try:
        import yaml  # type: ignore
        obj = yaml.safe_load(raw)
        text, trunc = _clip(json.dumps(obj, indent=2, default=str))
        return Interpretation(kind="yaml", label="YAML document", text=text,
                              confidence=1.0, structured={"parsed": True},
                              n_bytes=len(data), truncated=trunc, handler="yaml")
    except ImportError:
        pass
    except Exception as e:
        text, trunc = _clip(raw)
        return Interpretation(kind="yaml", label="YAML (parse failed)", text=text,
                              confidence=0.6, structured={"parse_error": str(e)},
                              n_bytes=len(data), truncated=trunc, handler="yaml")
    obj, ok = _mini_yaml(raw)
    rendered = json.dumps(obj, indent=2, default=str)
    note = ("parsed with AG's built-in YAML subset reader (PyYAML not installed): "
            "mappings, sequences and scalars only")
    text, trunc = _clip(f"{rendered}\n\n--- raw ---\n{raw}" if not ok else rendered)
    return Interpretation(
        kind="yaml", label="YAML document", text=text, confidence=0.75 if ok else 0.5,
        structured={"parsed": ok, "keys": list(obj.keys()) if isinstance(obj, dict) else []},
        n_bytes=len(data), truncated=trunc, handler="yaml", notes=[note])


def _mini_yaml(raw: str):
    """Indentation-driven parse of the YAML subset that covers most config files."""
    root: dict = {}
    stack: List[Tuple[int, object]] = [(-1, root)]
    ok = True
    for line in raw.splitlines():
        if not line.strip() or line.lstrip().startswith("#") or line.strip() == "---":
            continue
        indent = len(line) - len(line.lstrip())
        body = line.strip()
        while stack and stack[-1][0] >= indent:
            stack.pop()
        if not stack:
            stack = [(-1, root)]
        parent = stack[-1][1]
        if body.startswith("- "):
            item = _yaml_scalar(body[2:].strip())
            if isinstance(parent, list):
                parent.append(item)
            elif isinstance(parent, dict):
                ok = False
            continue
        if ":" in body:
            k, _, v = body.partition(":")
            k, v = k.strip().strip('"\''), v.strip()
            if not isinstance(parent, dict):
                ok = False
                continue
            if v == "":
                child: object = {}
                parent[k] = child
                stack.append((indent, child))
                # A following "- " line means it is actually a list; patch lazily.
                parent[k] = child
            else:
                parent[k] = _yaml_scalar(v)
        else:
            ok = False
    return root, ok


def _yaml_scalar(v: str):
    v = v.strip()
    if v.lower() in ("true", "yes", "on"):
        return True
    if v.lower() in ("false", "no", "off"):
        return False
    if v.lower() in ("null", "~", ""):
        return None
    if re.fullmatch(r"-?\d+", v):
        return int(v)
    if re.fullmatch(r"-?\d*\.\d+([eE][-+]?\d+)?", v):
        return float(v)
    return v.strip('"\'')


def _h_zip(data: bytes, kind: str, label: str, name: str) -> Interpretation:
    import zipfile
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
        infos = zf.infolist()
    except Exception as e:
        return _h_binary(data, kind, f"{label} (unreadable)", name,
                         note=f"zip open failed: {e}")
    lines = [f"ZIP archive: {len(infos)} entries"]
    total = sum(i.file_size for i in infos)
    lines.append(f"uncompressed total: {total} bytes")
    lines.append("")
    for i in infos[:120]:
        lines.append(f"  {i.filename}  ({i.file_size} bytes)")
    if len(infos) > 120:
        lines.append(f"  … {len(infos) - 120} more entries")
    # Show the first small text entry — an archive is usually opened to find one.
    for i in infos:
        if i.file_size and i.file_size < 32768 and not i.is_dir():
            try:
                inner = zf.read(i)
            except Exception:
                continue
            if printable_ratio(inner[:2048]) > 0.85:
                lines.append(f"\n--- preview: {i.filename} ---")
                lines.append(_decode(inner)[:2000])
                break
    text, trunc = _clip("\n".join(lines))
    return Interpretation(
        kind="zip", label=label, text=text, confidence=1.0,
        structured={"entries": [i.filename for i in infos[:300]],
                    "n_entries": len(infos), "uncompressed_bytes": total},
        n_bytes=len(data), truncated=trunc, handler="zip")


def _ooxml_text(data: bytes, kind: str) -> Tuple[str, dict]:
    """Pull text out of an OOXML package with zipfile + regex on the XML.

    A real XML parse is overkill here and more fragile: the runs we want are
    `<w:t>`, `<a:t>`, and shared strings, and they are flat. Stdlib only, so
    `.docx`/`.pptx`/`.xlsx` work on a machine with nothing installed.
    """
    import zipfile
    zf = zipfile.ZipFile(io.BytesIO(data))
    names = zf.namelist()
    meta: dict = {"parts": len(names)}
    out: List[str] = []
    if kind == "docx":
        xml = zf.read("word/document.xml").decode("utf-8", "replace")
        paras = re.findall(r"<w:p[ >].*?</w:p>|<w:p/>", xml, re.S)
        for p in paras:
            runs = re.findall(r"<w:t[^>]*>(.*?)</w:t>", p, re.S)
            line = "".join(runs)
            line = re.sub(r"<[^>]+>", "", line)
            out.append(_unxml(line))
        meta["paragraphs"] = len(paras)
    elif kind == "pptx":
        slides = sorted(n for n in names
                        if re.fullmatch(r"ppt/slides/slide\d+\.xml", n))
        for i, s in enumerate(slides, 1):
            xml = zf.read(s).decode("utf-8", "replace")
            runs = [_unxml(t) for t in re.findall(r"<a:t>(.*?)</a:t>", xml, re.S)]
            out.append(f"--- slide {i} ---")
            out.extend(runs)
        meta["slides"] = len(slides)
    elif kind == "xlsx":
        shared: List[str] = []
        if "xl/sharedStrings.xml" in names:
            sx = zf.read("xl/sharedStrings.xml").decode("utf-8", "replace")
            shared = [_unxml(re.sub(r"<[^>]+>", "",
                                    "".join(re.findall(r"<t[^>]*>(.*?)</t>", si, re.S))))
                      for si in re.findall(r"<si>(.*?)</si>", sx, re.S)]
        sheets = sorted(n for n in names
                        if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", n))
        for s in sheets:
            xml = zf.read(s).decode("utf-8", "replace")
            out.append(f"--- {s.split('/')[-1]} ---")
            for row in re.findall(r"<row[^>]*>(.*?)</row>", xml, re.S)[:400]:
                cells = []
                for c in re.findall(r"<c\b([^>]*)>(.*?)</c>|<c\b([^>]*)/>", row, re.S):
                    attrs, inner = (c[0], c[1]) if c[1] or c[0] else (c[2], "")
                    v = re.search(r"<v>(.*?)</v>", inner, re.S)
                    raw = _unxml(v.group(1)) if v else ""
                    if 't="s"' in attrs and raw.isdigit() and int(raw) < len(shared):
                        raw = shared[int(raw)]
                    elif 't="inlineStr"' in attrs:
                        it = re.search(r"<t[^>]*>(.*?)</t>", inner, re.S)
                        raw = _unxml(it.group(1)) if it else ""
                    cells.append(raw)
                out.append("\t".join(cells))
        meta["sheets"] = len(sheets)
        meta["shared_strings"] = len(shared)
    return "\n".join(out), meta


def _unxml(s: str) -> str:
    import html as _html
    return _html.unescape(re.sub(r"<[^>]+>", "", s))


def _h_ooxml(data: bytes, kind: str, label: str, name: str) -> Interpretation:
    try:
        body, meta = _ooxml_text(data, kind)
    except Exception as e:
        out = _h_zip(data, "zip", f"{label} (as zip)", name)
        out.notes.append(f"OOXML extraction failed ({e}); listed as a zip instead")
        out.confidence = 0.5
        return out
    text, trunc = _clip(body or "(no extractable text)")
    return Interpretation(
        kind=kind, label=label, text=text, confidence=0.93, structured=meta,
        n_bytes=len(data), truncated=trunc, handler="ooxml",
        notes=["text extracted from the OOXML package with the stdlib; "
               "formatting, images, and embedded objects are not represented"])


def _h_pdf(data: bytes, kind: str, label: str, name: str) -> Interpretation:
    """Extract text from PDF content streams with zlib + a Tj/TJ scanner.

    This is not a full PDF renderer — it decodes FlateDecode streams and pulls the
    string operands out of text-showing operators. That covers the large majority
    of text PDFs and fails cleanly (reporting low confidence) on scanned/image
    PDFs, which is the honest answer there: those need OCR, not a parser.
    """
    # `/Type/Pages` is the page *tree* node, not a page — counting it inflates the
    # page count by one on every document. Require a non-name character after.
    pages = len(re.findall(rb"/Type\s*/Page(?![s])", data))
    chunks: List[str] = []
    for m in re.finditer(rb"stream\r?\n(.*?)endstream", data, re.S):
        blob = m.group(1)
        text = None
        try:
            text = zlib.decompress(blob).decode("latin-1", "replace")
        except Exception:
            if printable_ratio(blob[:512]) > 0.85:
                text = blob.decode("latin-1", "replace")
        if not text:
            continue
        # Walk the operators in order so positioning survives into the output: Td/TD/T*
        # move to a new line and ET ends a text object, and without honouring them
        # every line of the page concatenates into one unreadable run.
        for tm in re.finditer(
                r"\((?:\\.|[^\\()])*\)\s*(?:Tj|')|"
                r"\[(?:[^\[\]\\]|\\.)*\]\s*TJ|"
                r"\b(?:Td|TD|T\*|ET)\b", text):
            frag = tm.group(0)
            if re.fullmatch(r"\b(?:Td|TD|T\*|ET)\b", frag):
                chunks.append("\n")
                continue
            for s in re.findall(r"\((?:\\.|[^\\()])*\)", frag):
                chunks.append(_pdf_unescape(s[1:-1]))
    body = "".join(chunks)
    body = re.sub(r"[ \t]{2,}", " ", body)
    body = re.sub(r" *\n *", "\n", body)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    # Calibrate against the document's own size: 20 characters is plenty for a
    # one-page form and nowhere near enough for a 40-page report, so judge the text
    # layer per page rather than against one absolute number.
    per_page = len(body) / max(1, pages)
    if per_page >= 200:
        conf = 0.9
    elif per_page >= 40:
        conf = 0.8
    elif body:
        conf = 0.55
    else:
        conf = 0.3
    notes = []
    if per_page < 40:
        notes.append(
            f"Only {len(body)} characters of text across {pages} page(s) — this is "
            "likely a scanned or image-only PDF. Reading it would need OCR, which "
            "AG does not have locally unless a skill is acquired for it. Do not "
            "assume the extracted text is the whole document.")
    meta = {}
    for key in ("Title", "Author", "Producer", "Creator", "CreationDate"):
        km = re.search(rb"/" + key.encode() + rb"\s*\(([^)]*)\)", data)
        if km:
            meta[key.lower()] = km.group(1).decode("latin-1", "replace")
    vm = re.match(rb"%PDF-(\d\.\d)", data)
    if vm:
        meta["version"] = vm.group(1).decode()
    meta["pages"] = pages
    text, trunc = _clip(body or "(no extractable text layer)")
    return Interpretation(
        kind="pdf", label="PDF document", text=text, confidence=conf,
        structured=meta, n_bytes=len(data), truncated=trunc, handler="pdf",
        notes=notes)


def _pdf_unescape(s: str) -> str:
    out = s.replace(r"\(", "(").replace(r"\)", ")").replace(r"\\", "\\")
    out = out.replace(r"\n", "\n").replace(r"\r", "\n").replace(r"\t", "\t")
    return re.sub(r"\\(\d{1,3})", lambda m: chr(int(m.group(1), 8)), out)


def _h_sqlite(data: bytes, kind: str, label: str, name: str) -> Interpretation:
    """Open the database read-only from a temp copy and describe its schema.

    Bytes rather than a path is the awkward case (the caller may have handed us a
    buffer), so we write a temp file. Read-only URI mode means we can never modify
    the user's database through this path.
    """
    import sqlite3
    import tempfile
    import os as _os
    tmp = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    try:
        tmp.write(data)
        tmp.close()
        con = sqlite3.connect(f"file:{tmp.name}?mode=ro", uri=True)
        cur = con.cursor()
        cur.execute("SELECT type, name, sql FROM sqlite_master "
                    "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name")
        objects = cur.fetchall()
        lines, tables = [], []
        for typ, nm, sql in objects:
            lines.append(f"{typ}: {nm}")
            if sql:
                lines.append(f"  {sql.strip()[:400]}")
            if typ == "table":
                tables.append(nm)
        lines.append("")
        for t in tables[:20]:
            try:
                cur.execute(f'SELECT COUNT(*) FROM "{t}"')
                n = cur.fetchone()[0]
                cur.execute(f'SELECT * FROM "{t}" LIMIT 5')
                cols = [d[0] for d in cur.description]
                rows = cur.fetchall()
            except Exception as e:
                lines.append(f"--- {t}: unreadable ({e})")
                continue
            lines.append(f"--- {t} ({n} rows) ---")
            lines.append(" | ".join(cols))
            for r in rows:
                lines.append(" | ".join(str(c)[:60] for c in r))
        con.close()
        text, trunc = _clip("\n".join(lines))
        return Interpretation(
            kind="sqlite", label="SQLite database", text=text, confidence=1.0,
            structured={"tables": tables,
                        "objects": [{"type": t, "name": n} for t, n, _ in objects]},
            n_bytes=len(data), truncated=trunc, handler="sqlite")
    except Exception as e:
        return _h_binary(data, kind, label, name, note=f"sqlite open failed: {e}")
    finally:
        try:
            _os.unlink(tmp.name)
        except OSError:
            pass


def _h_image(data: bytes, kind: str, label: str, name: str) -> Interpretation:
    """Dimensions and colour info from header bytes — no imaging library needed."""
    w = h = None
    extra: dict = {}
    try:
        if kind == "png":
            w, h = struct.unpack(">II", data[16:24])
            depth, ctype = data[24], data[25]
            extra = {"bit_depth": depth,
                     "color_type": {0: "grayscale", 2: "rgb", 3: "palette",
                                    4: "gray+alpha", 6: "rgba"}.get(ctype, ctype)}
        elif kind == "jpeg":
            i = 2
            while i < len(data) - 9:
                if data[i] != 0xFF:
                    i += 1
                    continue
                marker = data[i + 1]
                if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                              0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                    h, w = struct.unpack(">HH", data[i + 5:i + 9])
                    extra = {"components": data[i + 9] if i + 9 < len(data) else None}
                    break
                seg = struct.unpack(">H", data[i + 2:i + 4])[0]
                i += 2 + seg
        elif kind == "gif":
            w, h = struct.unpack("<HH", data[6:10])
            extra = {"frames": data.count(b"\x00\x21\xf9\x04")}
        elif kind == "bmp":
            w, h = struct.unpack("<ii", data[18:26])
        elif kind == "webp":
            if data[12:16] == b"VP8X":
                w = int.from_bytes(data[24:27], "little") + 1
                h = int.from_bytes(data[27:30], "little") + 1
            elif data[12:16] == b"VP8 ":
                w = struct.unpack("<H", data[26:28])[0] & 0x3FFF
                h = struct.unpack("<H", data[28:30])[0] & 0x3FFF
            elif data[12:16] == b"VP8L":
                b0, b1, b2, b3 = data[21:25]
                bits = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
                w = (bits & 0x3FFF) + 1
                h = ((bits >> 14) & 0x3FFF) + 1
    except Exception:
        pass
    dims = f"{w}×{h}" if w and h else "unknown dimensions"
    note = ("AG can read this image's structure but not its content — describing "
            "what is depicted needs a vision model, which this local setup does "
            "not have wired. Say so rather than guessing at the subject.")
    text = (f"{label}: {dims}\n"
            + "\n".join(f"{k}: {v}" for k, v in extra.items())
            + f"\n\n{note}")
    return Interpretation(
        kind=kind, label=label, text=text, confidence=0.95,
        structured={"width": w, "height": h, **extra},
        n_bytes=len(data), handler="image", notes=[note])


def _h_media(data: bytes, kind: str, label: str, name: str) -> Interpretation:
    info: dict = {}
    try:
        if kind == "wav":
            info["riff_size"] = struct.unpack("<I", data[4:8])[0]
            i = 12
            while i < min(len(data) - 8, 65536):
                cid = data[i:i + 4]
                size = struct.unpack("<I", data[i + 4:i + 8])[0]
                if cid == b"fmt ":
                    fmt, ch, rate, _, _, bits = struct.unpack("<HHIIHH",
                                                              data[i + 8:i + 24])
                    info.update({"channels": ch, "sample_rate": rate,
                                 "bits_per_sample": bits, "format_code": fmt})
                elif cid == b"data":
                    info["data_bytes"] = size
                    br = info.get("sample_rate", 0) * info.get("channels", 0) * \
                        max(1, info.get("bits_per_sample", 8) // 8)
                    if br:
                        info["duration_s"] = round(size / br, 2)
                    break
                i += 8 + size + (size % 2)
        elif kind == "mp4":
            boxes = []
            i = 0
            while i < min(len(data) - 8, 1 << 20):
                size = struct.unpack(">I", data[i:i + 4])[0]
                typ = data[i + 4:i + 8].decode("latin-1", "replace")
                boxes.append(typ)
                if size <= 1:
                    break
                i += size
            info["boxes"] = boxes[:20]
            bm = re.search(rb"ftyp(....)", data[:64])
            if bm:
                info["brand"] = bm.group(1).decode("latin-1", "replace")
        elif kind == "mp3":
            info["has_id3"] = data.startswith(b"ID3")
            if info["has_id3"]:
                info["id3_version"] = f"2.{data[3]}.{data[4]}"
    except Exception:
        pass
    text = (f"{label}\n" + "\n".join(f"{k}: {v}" for k, v in info.items())
            + "\n\nAG reads this container's metadata only; decoding the audio or "
              "video content would need a codec library or an acquired skill.")
    return Interpretation(kind=kind, label=label, text=text, confidence=0.9,
                          structured=info, n_bytes=len(data), handler="media")


def _h_executable(data: bytes, kind: str, label: str, name: str) -> Interpretation:
    info: dict = {}
    try:
        if kind == "elf":
            info["class"] = {1: "32-bit", 2: "64-bit"}.get(data[4], data[4])
            info["endian"] = {1: "little", 2: "big"}.get(data[5], data[5])
            info["os_abi"] = data[7]
            etype = struct.unpack("<H", data[16:18])[0]
            info["type"] = {1: "relocatable", 2: "executable", 3: "shared object",
                            4: "core dump"}.get(etype, etype)
            machine = struct.unpack("<H", data[18:20])[0]
            info["machine"] = {3: "x86", 62: "x86-64", 40: "ARM",
                               183: "AArch64", 243: "RISC-V"}.get(machine, machine)
        elif kind == "pe":
            off = struct.unpack("<I", data[0x3C:0x40])[0]
            if data[off:off + 4] == b"PE\x00\x00":
                machine = struct.unpack("<H", data[off + 4:off + 6])[0]
                info["machine"] = {0x14c: "x86", 0x8664: "x86-64",
                                   0xAA64: "ARM64"}.get(machine, hex(machine))
                info["sections"] = struct.unpack("<H", data[off + 6:off + 8])[0]
    except Exception:
        pass
    strings = _extract_strings(data, min_len=6, limit=40)
    text = (f"{label}\n" + "\n".join(f"{k}: {v}" for k, v in info.items())
            + "\n\nprintable strings (first 40):\n"
            + "\n".join(f"  {s}" for s in strings)
            + "\n\nThis is a compiled binary. AG reports its headers and strings; "
              "it does not execute or disassemble it.")
    return Interpretation(kind=kind, label=label, text=text, confidence=0.93,
                          structured={**info, "strings": strings},
                          n_bytes=len(data), handler="executable")


def _h_gzip(data: bytes, kind: str, label: str, name: str) -> Interpretation:
    import gzip as _gzip
    try:
        inner = _gzip.decompress(data)
    except Exception as e:
        return _h_binary(data, kind, label, name, note=f"gunzip failed: {e}")
    sub = interpret(inner, filename=Path(name).stem if name else "")
    sub.notes.insert(0, f"gzip container ({len(data)} bytes) decompressed to "
                        f"{len(inner)} bytes; interpreted the contents below")
    sub.kind = f"gzip:{sub.kind}"
    sub.label = f"gzip → {sub.label}"
    sub.n_bytes = len(data)
    sub.handler = "gzip"
    return sub


def _h_tar(data: bytes, kind: str, label: str, name: str) -> Interpretation:
    import tarfile
    try:
        tf = tarfile.open(fileobj=io.BytesIO(data))
        members = tf.getmembers()
    except Exception as e:
        return _h_binary(data, kind, label, name, note=f"tar open failed: {e}")
    lines = [f"tar archive: {len(members)} entries"]
    for m in members[:120]:
        lines.append(f"  {m.name}  ({m.size} bytes)")
    if len(members) > 120:
        lines.append(f"  … {len(members) - 120} more")
    text, trunc = _clip("\n".join(lines))
    return Interpretation(kind="tar", label="tar archive", text=text, confidence=1.0,
                          structured={"entries": [m.name for m in members[:300]],
                                      "n_entries": len(members)},
                          n_bytes=len(data), truncated=trunc, handler="tar")


def _h_base64(data: bytes, kind: str, label: str, name: str) -> Interpretation:
    raw = re.sub(rb"\s", b"", data)
    try:
        decoded = binascii.a2b_base64(raw + b"=" * (-len(raw) % 4))
    except Exception as e:
        return _h_text(data, "text", "text (base64 decode failed)", name)
    sub = interpret(decoded, filename="")
    sub.notes.insert(0, f"base64 payload ({len(data)} chars) decoded to "
                        f"{len(decoded)} bytes; interpreted below")
    sub.kind = f"base64:{sub.kind}"
    sub.label = f"base64 → {sub.label}"
    sub.handler = "base64"
    return sub


def _h_binary(data: bytes, kind: str, label: str, name: str,
              note: str = "") -> Interpretation:
    """Known-binary fallback: structure report + hexdump + strings."""
    ent = shannon_entropy(data[:65536])
    strings = _extract_strings(data, min_len=5, limit=40)
    parts = [f"{label} — {len(data)} bytes, entropy {ent}/8.0",
             "", "hexdump (first 384 bytes):", hexdump(data[:384])]
    if strings:
        parts += ["", "printable strings:"] + [f"  {s}" for s in strings]
    if note:
        parts += ["", f"note: {note}"]
    text, trunc = _clip("\n".join(parts))
    return Interpretation(kind=kind or "binary", label=label, text=text,
                          confidence=0.6, structured={"entropy": ent,
                                                      "strings": strings},
                          n_bytes=len(data), truncated=trunc, handler="binary",
                          notes=[note] if note else [])


def _extract_strings(data: bytes, *, min_len: int = 5, limit: int = 40) -> List[str]:
    """`strings(1)` in ten lines — often the fastest route to what a blob is."""
    out, cur = [], bytearray()
    for b in data[:1 << 20]:
        if 32 <= b < 127:
            cur.append(b)
        else:
            if len(cur) >= min_len:
                out.append(cur.decode("ascii"))
                if len(out) >= limit:
                    return out
            cur = bytearray()
    if len(cur) >= min_len and len(out) < limit:
        out.append(cur.decode("ascii"))
    return out


# --------------------------------------------------------------------------- #
# The unknown-format inference engine
# --------------------------------------------------------------------------- #

@dataclass
class Hypothesis:
    claim: str
    confidence: float
    evidence: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def infer_unknown(data: bytes, *, filename: str = "") -> Interpretation:
    """Reason about bytes that match no known format.

    Produces ranked hypotheses from structural evidence. This is the mechanism for
    "interpret formats that don't exist yet": AG cannot know a format it has never
    seen, but it *can* measure whether the bytes are compressed, record-structured,
    text-with-an-unusual-encoding, or a container with a magic prefix — and hand
    the model a description precise enough to act on, plus an honest confidence so
    it knows whether to proceed, acquire a handler skill, or ask for guidance.
    """
    n = len(data)
    ent = shannon_entropy(data[:1 << 16])
    pr = printable_ratio(data[:1 << 16])
    hyps: List[Hypothesis] = []
    evidence: List[str] = [
        f"{n} bytes", f"entropy {ent}/8.0", f"printable ratio {pr:.2%}",
    ]

    # Magic prefix: even an unknown format usually announces itself.
    prefix = data[:16]
    ascii_prefix = "".join(chr(b) if 32 <= b < 127 else "." for b in prefix)
    evidence.append(f"first 16 bytes: {prefix.hex()} ({ascii_prefix!r})")
    magic_word = re.match(rb"([A-Za-z][A-Za-z0-9_\-!]{2,7})", data[:8])
    if magic_word:
        tag = magic_word.group(1).decode("ascii")
        hyps.append(Hypothesis(
            f"Container or tagged format with the signature {tag!r}",
            0.55,
            [f"file begins with the ASCII tag {tag!r}, the conventional position "
             f"for a format magic number"]))

    # Entropy-driven classification.
    if ent > 7.5:
        hyps.append(Hypothesis(
            "Compressed, encrypted, or already-packed data",
            0.8,
            [f"entropy {ent} is at the ceiling (8.0); byte values are near-uniform, "
             "which happens after compression or encryption",
             "structure cannot be recovered without the right decoder/key"]))
    elif ent < 1.5:
        hyps.append(Hypothesis(
            "Sparse or heavily padded data (mostly one byte value)",
            0.7,
            [f"entropy {ent} is very low — the file is dominated by a single value, "
             "typical of zero-padding, a sparse image, or a preallocated buffer"]))

    # Text hiding behind an unusual encoding.
    if pr < 0.8:
        for enc in ("utf-16-le", "utf-16-be", "utf-32-le", "cp1252", "latin-1",
                    "shift_jis", "euc-jp", "gb18030", "koi8-r"):
            try:
                t = data[:4096].decode(enc)
            except (UnicodeDecodeError, LookupError):
                continue
            letters = sum(1 for c in t if c.isprintable() or c in "\n\r\t")
            if letters / max(1, len(t)) > 0.95 and len(t) > 32:
                hyps.append(Hypothesis(
                    f"Text encoded as {enc}",
                    0.65,
                    [f"decodes cleanly as {enc} with {letters / len(t):.0%} "
                     "printable characters",
                     f"sample: {t[:120]!r}"]))
                break

    # Fixed-size records: score every plausible divisor of the length.
    rec = _detect_record_size(data)
    if rec:
        size, score = rec
        hyps.append(Hypothesis(
            f"Fixed-length record file with a {size}-byte record",
            min(0.3 + score * 0.5, 0.75),
            [f"{n} bytes divides evenly into {n // size} records of {size} bytes",
             f"byte-position correlation across record boundaries scores {score:.2f} "
             "(columns repeat, which is what a record table looks like)"]))

    # Line structure: even unknown text formats usually have lines.
    if pr > 0.85:
        text = data.decode("utf-8", "replace")
        lines = text.splitlines()
        nonempty = [ln for ln in lines if ln.strip()]
        if nonempty:
            lengths = [len(ln) for ln in nonempty]
            avg = sum(lengths) / len(lengths)
            uniform = sum(1 for l in lengths if abs(l - avg) < 2) / len(lengths)
            evidence.append(f"{len(lines)} lines, average length {avg:.1f}")
            delim = _guess_delimiter(nonempty[:50])
            if delim:
                hyps.append(Hypothesis(
                    f"Delimited records using {delim!r}",
                    0.7,
                    [f"every sampled line contains a consistent number of {delim!r} "
                     "separators — a table with an unusual extension"]))
            if uniform > 0.9 and len(nonempty) > 4:
                hyps.append(Hypothesis(
                    "Fixed-width (column-aligned) text records",
                    0.6,
                    [f"{uniform:.0%} of lines are within 2 characters of the mean "
                     "length, which indicates padded columns rather than free text"]))
            kv = sum(1 for ln in nonempty[:60] if re.match(r"^\s*[\w.\-]+\s*[:=]\s*\S", ln))
            if kv > len(nonempty[:60]) * 0.6:
                hyps.append(Hypothesis(
                    "Key/value configuration format",
                    0.65,
                    [f"{kv} of the first {len(nonempty[:60])} lines match "
                     "`key: value` or `key = value`"]))
            ts = sum(1 for ln in nonempty[:60]
                     if re.match(r"^\s*[\[<(]?\d{4}[-/]\d{2}[-/]\d{2}|"
                                 r"^\s*[\[<(]?\d{2}:\d{2}:\d{2}", ln))
            if ts > len(nonempty[:60]) * 0.5:
                hyps.append(Hypothesis(
                    "Timestamped log file",
                    0.7,
                    [f"{ts} of the first {len(nonempty[:60])} lines begin with a "
                     "date or clock time"]))

    # Repeated byte sequences = a dictionary/table, not free-form content.
    tokens = _repeated_tokens(data)
    if tokens:
        evidence.append("repeated byte sequences: "
                        + ", ".join(f"{t!r}×{c}" for t, c in tokens[:5]))
        hyps.append(Hypothesis(
            "Structured format with a repeating tag/dictionary",
            0.5,
            [f"the sequence {tokens[0][0]!r} recurs {tokens[0][1]} times at "
             "irregular offsets, which is how tagged formats mark fields"]))

    # Embedded known formats — a container wrapping something we do know.
    embedded = _find_embedded(data)
    if embedded:
        hyps.append(Hypothesis(
            f"Container embedding {', '.join(k for k, _ in embedded[:3])}",
            0.6,
            [f"found a {k} signature at offset {off}" for k, off in embedded[:3]]))

    hyps.sort(key=lambda h: -h.confidence)
    if not hyps:
        hyps.append(Hypothesis(
            "No structural signal found",
            0.1,
            ["none of the entropy, line, record, or signature probes matched",
             "the bytes below are the complete evidence available"]))

    top = hyps[0]
    lines_out = [
        f"UNRECOGNISED FORMAT — AG could not match this to a known format, so what "
        f"follows is inference, not a parse.",
        "",
        f"Best hypothesis: {top.claim}  (confidence {top.confidence:.2f})",
        "",
        "Measurements:",
    ]
    lines_out += [f"  - {e}" for e in evidence]
    lines_out += ["", "Ranked hypotheses:"]
    for h in hyps[:5]:
        lines_out.append(f"  [{h.confidence:.2f}] {h.claim}")
        for e in h.evidence:
            lines_out.append(f"        · {e}")
    strings = _extract_strings(data, min_len=6, limit=30)
    if strings:
        lines_out += ["", "Printable strings found inside:"]
        lines_out += [f"  {s}" for s in strings]
    lines_out += ["", "Hexdump (first 384 bytes):", hexdump(data[:384])]
    lines_out += [
        "",
        "How to proceed: if this format matters, AG can author a parser skill for "
        "it (acquire_skill) once it has a sample and a spec, or you can name the "
        "format and AG will handle it directly. AG will not pretend to have read "
        "content it only inferred.",
    ]
    text, trunc = _clip("\n".join(lines_out))
    return Interpretation(
        kind="unknown", label=f"unrecognised format ({top.claim})", text=text,
        confidence=round(top.confidence * 0.6, 3),   # confidence in the *reading*
        structured={"hypotheses": [h.as_dict() for h in hyps],
                    "entropy": ent, "printable_ratio": pr,
                    "magic_hex": prefix.hex(), "strings": strings[:20],
                    "embedded": [{"kind": k, "offset": o} for k, o in embedded[:10]]},
        n_bytes=n, truncated=trunc, handler="infer",
        notes=["interpretation is inferred from structure, not parsed"])


def _detect_record_size(data: bytes, *, max_size: int = 4096) -> Optional[Tuple[int, float]]:
    """Find a fixed record length by testing divisors for columnar self-similarity.

    If a file is an array of N-byte records, then byte i of every record tends to
    come from the same small alphabet (a type tag, a flag, a high byte that is
    usually zero). We score each candidate N by how concentrated those columns are.
    """
    n = len(data)
    if n < 64:
        return None
    sample = data[:min(n, 1 << 16)]
    best: Optional[Tuple[int, float]] = None
    for size in range(4, min(max_size, n // 4) + 1):
        if n % size:
            continue
        rows = len(sample) // size
        if rows < 4:
            continue
        score = 0.0
        cols = min(size, 32)
        for c in range(cols):
            vals = [sample[r * size + c] for r in range(rows)]
            distinct = len(set(vals))
            score += 1.0 - (distinct - 1) / max(1, len(vals) - 1)
        score /= cols
        if best is None or score > best[1]:
            best = (size, round(score, 3))
    if best and best[1] > 0.55:
        return best
    return None


def _repeated_tokens(data: bytes, *, length: int = 4,
                     limit: int = 8) -> List[Tuple[bytes, int]]:
    """Most frequent fixed-length byte sequences, ignoring runs of one value."""
    sample = data[:1 << 16]
    counts: Dict[bytes, int] = {}
    for i in range(0, len(sample) - length, 2):
        tok = sample[i:i + length]
        if len(set(tok)) == 1:
            continue
        counts[tok] = counts.get(tok, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    return [(t, c) for t, c in ranked[:limit] if c >= 3]


def _find_embedded(data: bytes) -> List[Tuple[str, int]]:
    """Known signatures appearing *inside* a blob — the container tell."""
    found: List[Tuple[str, int]] = []
    for prefix, kind, _label in MAGIC:
        if not kind or len(prefix) < 4:
            continue
        off = data.find(prefix, 1)
        if off > 0:
            found.append((kind, off))
    return sorted(found, key=lambda kv: kv[1])[:10]


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #

HANDLERS: Dict[str, Callable[[bytes, str, str, str], Interpretation]] = {
    "text": _h_text, "markdown": _h_text, "source": _h_text, "pem": _h_text,
    "dotenv": _h_ini, "ini": _h_ini, "toml": _h_ini,
    "json": _h_json, "jsonl": _h_jsonl,
    "csv": _h_delimited, "tsv": _h_delimited, "delimited": _h_delimited,
    "xml": _h_xml, "html": _h_html, "yaml": _h_yaml,
    "zip": _h_zip, "tar": _h_tar, "gzip": _h_gzip,
    "docx": _h_ooxml, "xlsx": _h_ooxml, "pptx": _h_ooxml,
    "pdf": _h_pdf, "sqlite": _h_sqlite, "base64": _h_base64,
    "png": _h_image, "jpeg": _h_image, "gif": _h_image, "bmp": _h_image,
    "webp": _h_image, "ico": _h_image, "tiff": _h_image,
    "wav": _h_media, "mp3": _h_media, "mp4": _h_media, "flac": _h_media,
    "ogg": _h_media, "avi": _h_media, "matroska": _h_media,
    "elf": _h_executable, "pe": _h_executable, "macho": _h_executable,
}


def interpret(data: bytes, *, filename: str = "",
              max_text: int = MAX_TEXT_CHARS) -> Interpretation:
    """Turn bytes into something a model can read. Never raises.

    Order: sniff → handler → (on failure) binary fallback → (on no match)
    structural inference. Every path returns an Interpretation with an honest
    confidence.
    """
    if not data:
        return Interpretation(kind="empty", label="empty file", text="(empty file)",
                              confidence=1.0, n_bytes=0, handler="empty")
    kind, label, conf = sniff(data, filename=filename)
    handler = HANDLERS.get(kind)
    if handler is not None:
        try:
            out = handler(data, kind, label, filename)
            # Reconcile the two confidences honestly. A handler that performed a
            # *strict* parse (>=0.9 — JSON loaded, XML parsed, zip opened, SQLite
            # schema read) has proven the format, so it overrides a weak sniff: the
            # parse is stronger evidence than the extension that suggested it.
            # Anything softer is capped by the sniff, because a lenient reader
            # "succeeds" on almost anything and must not manufacture certainty.
            if out.confidence >= 0.9:
                out.confidence = round(max(out.confidence, conf), 3)
            else:
                out.confidence = round(min(out.confidence, max(conf, 0.5)), 3)
            return out
        except Exception as e:
            fallback = _h_binary(data, kind, f"{label} (handler failed)", filename,
                                 note=f"{kind} handler raised: {e}")
            fallback.confidence = 0.4
            return fallback
    if kind == "unknown":
        return infer_unknown(data, filename=filename)
    # Known-but-unhandled (parquet, pickle, rar, …): describe honestly.
    out = _h_binary(data, kind, label, filename,
                    note=f"recognised as {label}, but AG has no parser for it yet; "
                         f"acquire_skill could author one")
    out.confidence = round(conf * 0.7, 3)
    return out


def read_path(path, *, max_bytes: int = MAX_READ_BYTES) -> Interpretation:
    """Interpret a file on disk. Directories get a listing; errors are reported."""
    p = Path(path).expanduser()
    try:
        if p.is_dir():
            return _describe_dir(p)
        size = p.stat().st_size
        with p.open("rb") as f:
            data = f.read(max_bytes)
        out = interpret(data, filename=p.name)
        if size > len(data):
            out.truncated = True
            out.n_bytes = size
            out.notes.append(f"read the first {len(data)} of {size} bytes")
        return out
    except FileNotFoundError:
        return Interpretation(kind="error", label="not found",
                              text=f"No such file or directory: {p}",
                              confidence=1.0, handler="error")
    except PermissionError:
        return Interpretation(kind="error", label="permission denied",
                              text=f"Permission denied reading {p}",
                              confidence=1.0, handler="error")
    except OSError as e:
        return Interpretation(kind="error", label="unreadable",
                              text=f"Could not read {p}: {e}",
                              confidence=1.0, handler="error")


def _describe_dir(p: Path) -> Interpretation:
    entries = []
    try:
        for child in sorted(p.iterdir())[:400]:
            try:
                st = child.stat()
                entries.append({"name": child.name, "dir": child.is_dir(),
                                "bytes": st.st_size})
            except OSError:
                entries.append({"name": child.name, "dir": False, "bytes": None})
    except OSError as e:
        return Interpretation(kind="error", label="unreadable directory",
                              text=f"Could not list {p}: {e}", handler="error")
    lines = [f"directory: {p}  ({len(entries)} entries)"]
    for e in entries:
        mark = "/" if e["dir"] else ""
        size = "" if e["bytes"] is None else f"  {e['bytes']} bytes"
        lines.append(f"  {e['name']}{mark}{size}")
    text, trunc = _clip("\n".join(lines))
    return Interpretation(kind="directory", label="directory", text=text,
                          confidence=1.0, structured={"entries": entries},
                          truncated=trunc, handler="dir")


def supported_formats() -> List[dict]:
    """What AG can read, for the inventory and the GUI."""
    seen: Dict[str, dict] = {}
    for kind, fn in HANDLERS.items():
        seen[kind] = {"kind": kind, "handler": fn.__name__.lstrip("_")}
    return sorted(seen.values(), key=lambda d: d["kind"])
