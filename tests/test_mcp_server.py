"""MCP server.

Two things are being protected here. First, protocol correctness: a malformed
frame must produce a JSON-RPC error, never a traceback that desynchronises the
stream. Second — and more important — REACHABILITY IS NOT AUTHORITY: exposing AG
to other agents must not hand them permissions the local user never granted.
"""
from __future__ import annotations

import dataclasses
import io
import json

import pytest

from ag.config import Config
from ag.mcp_server import MCPServer


@pytest.fixture
def server():
    return MCPServer(dataclasses.replace(Config(), backend="dry"))


def drive(server, messages):
    """Run a list of messages through the stdio loop and return parsed replies."""
    inp = io.StringIO("\n".join(json.dumps(m) for m in messages) + "\n")
    out = io.StringIO()
    server.serve_stdio(inp, out)
    return [json.loads(line) for line in out.getvalue().splitlines() if line.strip()]


# --------------------------------------------------------------------------
# protocol
# --------------------------------------------------------------------------

def test_initialize_announces_protocol_and_capabilities(server):
    r = server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                       "params": {"protocolVersion": "2024-11-05"}})
    res = r["result"]
    assert res["protocolVersion"]
    assert res["serverInfo"]["name"]
    assert "tools" in res["capabilities"]


def test_ping_is_answered(server):
    assert server.handle({"jsonrpc": "2.0", "id": 2, "method": "ping"})["result"] == {}


def test_tools_list_is_well_formed(server):
    tools = server.handle({"jsonrpc": "2.0", "id": 3,
                           "method": "tools/list"})["result"]["tools"]
    assert tools
    for t in tools:
        assert t["name"] and t["description"]
        schema = t["inputSchema"]
        assert schema["type"] == "object"
        assert "properties" in schema


def test_unknown_method_is_an_error_not_a_crash(server):
    r = server.handle({"jsonrpc": "2.0", "id": 4, "method": "no/such/method"})
    assert r["error"]["code"] == -32601


def test_unknown_tool_is_reported_in_band(server):
    """A bad tool name is the CALLER's error, so it comes back as an errored tool
    result the model can read and recover from, not a protocol-level failure."""
    r = server.handle({"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                       "params": {"name": "nope", "arguments": {}}})
    assert r["result"]["isError"] is True
    assert "nope" in r["result"]["content"][0]["text"]


def test_notifications_get_no_reply(server):
    """A JSON-RPC notification has no id; replying to one corrupts the stream."""
    assert server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


def test_malformed_json_yields_a_parse_error(server):
    out = io.StringIO()
    server.serve_stdio(io.StringIO('{"not valid json\n'), out)
    replies = [json.loads(l) for l in out.getvalue().splitlines() if l.strip()]
    assert replies and replies[0]["error"]["code"] == -32700


def test_batch_requests_are_supported(server):
    out = io.StringIO()
    batch = [{"jsonrpc": "2.0", "id": 1, "method": "ping"},
             {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}]
    server.serve_stdio(io.StringIO(json.dumps(batch) + "\n"), out)
    replies = json.loads(out.getvalue().strip())
    assert isinstance(replies, list) and len(replies) == 2


def test_stdout_carries_protocol_only(server, capsys):
    """Anything printed to stdout corrupts the JSON-RPC stream. Diagnostics must
    go to stderr — a rule that is easy to break and silently fatal."""
    out = io.StringIO()
    server.serve_stdio(io.StringIO(json.dumps(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}) + "\n"), out)
    assert capsys.readouterr().out == ""
    for line in out.getvalue().splitlines():
        json.loads(line)


# --------------------------------------------------------------------------
# tools and resources
# --------------------------------------------------------------------------

def test_os_info_tool_runs(server):
    r = server.handle({"jsonrpc": "2.0", "id": 6, "method": "tools/call",
                       "params": {"name": "os_info", "arguments": {}}})
    assert r["result"]["isError"] is False
    assert "platform" in r["result"]["content"][0]["text"].lower()


def test_inspect_format_tool_runs(server):
    r = server.handle({"jsonrpc": "2.0", "id": 7, "method": "tools/call",
                       "params": {"name": "inspect_format",
                                  "arguments": {"path": "config.json"}}})
    assert r["result"]["isError"] is False
    assert "json" in r["result"]["content"][0]["text"].lower()


def test_resources_list_and_read(server):
    res = server.handle({"jsonrpc": "2.0", "id": 8,
                         "method": "resources/list"})["result"]["resources"]
    assert res
    for r in res:
        assert r["uri"].startswith("ag://")
        assert r["name"]
    got = server.handle({"jsonrpc": "2.0", "id": 9, "method": "resources/read",
                         "params": {"uri": res[0]["uri"]}})
    assert got["result"]["contents"][0]["text"]


def test_unknown_resource_errors_cleanly(server):
    r = server.handle({"jsonrpc": "2.0", "id": 10, "method": "resources/read",
                       "params": {"uri": "ag://does/not/exist"}})
    assert "error" in r or r["result"].get("isError")


def test_a_tool_raising_is_contained(server, monkeypatch):
    """One broken tool must not take down the session."""
    import ag.osadapt as osadapt
    monkeypatch.setattr(osadapt, "report", lambda: 1 / 0)
    r = server.handle({"jsonrpc": "2.0", "id": 11, "method": "tools/call",
                       "params": {"name": "os_info", "arguments": {}}})
    assert r["result"]["isError"] is True
    assert server.handle({"jsonrpc": "2.0", "id": 12, "method": "ping"})["result"] == {}


# --------------------------------------------------------------------------
# authority
# --------------------------------------------------------------------------

def test_remote_callers_get_no_code_execution_by_default():
    """The load-bearing security test. An MCP client is just another caller: it
    goes through the same PermissionBroker, so it cannot obtain a capability the
    user did not grant in config. Being reachable must never mean being trusted."""
    srv = MCPServer(dataclasses.replace(Config(), backend="dry",
                                        allow_code_exec=False))
    assert srv._broker().check("code_exec") is False
    names = {t["name"] for t in srv.handle(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})["result"]["tools"]}
    assert "code_exec" not in names
    assert "python_exec" not in names


def test_permissions_follow_config_not_the_caller():
    """The permission answer must come from configuration, and asking differently
    over the wire must not change it."""
    denied = MCPServer(dataclasses.replace(Config(), backend="dry",
                                           allow_code_exec=False))
    allowed = MCPServer(dataclasses.replace(Config(), backend="dry",
                                            allow_code_exec=True))
    assert denied._broker().check("code_exec") is False
    assert allowed._broker().check("code_exec") is True


def test_read_tools_do_not_escape_the_permission_broker(server):
    """read_any is gated on filesystem_read like every other reader."""
    r = server.handle({"jsonrpc": "2.0", "id": 13, "method": "tools/call",
                       "params": {"name": "read_any",
                                  "arguments": {"path": "/etc/shadow"}}})
    text = r["result"]["content"][0]["text"].lower()
    assert r["result"]["isError"] or "permission" in text or "denied" in text \
        or "error" in text or "not" in text
