"""A tiny built-in web app for AG (stdlib only).

`python -m ag serve` starts a local server with a minimal browser UI. Open it on
this machine, or from your phone/another device on the same network (bind --host
0.0.0.0). Cross-platform by virtue of being a web page — any OS with a browser.

The UI just drives the same pipeline (optimize -> execute -> critique -> iterate),
so profile principles and self-improvement are unchanged. This IS an inbound server
(for YOUR use); it is not a data-farming endpoint — bind to localhost unless you
deliberately want LAN/phone access.
"""
from __future__ import annotations

import html
import json
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .config import Config
from .model import make_client
from .pipeline import run as run_pipeline

PAGE = """<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Apple-Gorilla</title>
<style>
 :root{color-scheme:light dark}
 body{font-family:system-ui,sans-serif;max-width:760px;margin:0 auto;padding:16px}
 h1{font-size:1.3rem} textarea{width:100%;min-height:90px;font-size:1rem;padding:8px}
 button{font-size:1rem;padding:10px 16px;margin-top:8px;cursor:pointer}
 #out{white-space:pre-wrap;margin-top:16px;padding:12px;border:1px solid #8884;border-radius:8px}
 .meta{color:#8888;font-size:.85rem}
</style></head><body>
<h1>🦍 Apple-Gorilla</h1>
<div class="meta" id="status">ready</div>
<textarea id="p" placeholder="Ask AG anything..."></textarea><br>
<button onclick="go()">Run</button>
<div id="out"></div>
<script>
async function go(){
  const p=document.getElementById('p').value.trim(); if(!p)return;
  const s=document.getElementById('status'), o=document.getElementById('out');
  s.textContent='thinking...'; o.textContent='';
  try{
    const r=await fetch('/run',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({prompt:p})});
    const d=await r.json();
    o.textContent=d.answer||('error: '+(d.error||'unknown'));
    s.textContent='done · '+(d.meta||'');
  }catch(e){o.textContent='request failed: '+e; s.textContent='error';}
}
</script></body></html>"""


class _Handler(BaseHTTPRequestHandler):
    cfg: Config = Config()

    def _send(self, code, body, ctype="text/html; charset=utf-8"):
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE)
        else:
            self._send(404, "not found", "text/plain")

    def do_POST(self):
        if self.path != "/run":
            self._send(404, "not found", "text/plain")
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(n) or b"{}")
            prompt = str(payload.get("prompt", "")).strip()
            if not prompt:
                raise ValueError("empty prompt")
            answer, meta = run_prompt(self.cfg, prompt)
            self._send(200, json.dumps({"answer": answer, "meta": meta}),
                       "application/json")
        except Exception as e:
            self._send(200, json.dumps({"error": str(e)}), "application/json")

    def log_message(self, *a):  # quiet
        pass


def run_prompt(cfg: Config, prompt: str) -> tuple[str, str]:
    """Run one prompt through the pipeline; returns (answer, meta-string)."""
    client = make_client(cfg)
    broker = None
    if cfg.allow_web:
        from .permissions import PermissionBroker
        broker = PermissionBroker(allow_external_tools=True)
        broker.grant("network")
    rec = run_pipeline(client, cfg, prompt, web=cfg.allow_web, broker=broker)
    meta = f"iterations={rec.iterations} dry_run={rec.dry_run}"
    return rec.answer, meta


def _lan_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = False):
    _Handler.cfg = Config.load()
    httpd = ThreadingHTTPServer((host, port), _Handler)
    local = f"http://127.0.0.1:{port}"
    print(f"Apple-Gorilla web app running:")
    print(f"  this machine : {local}")
    if host == "0.0.0.0":
        print(f"  on your phone: http://{_lan_ip()}:{port}  (same wifi)")
    else:
        print(f"  (localhost only; use --host 0.0.0.0 to reach it from your phone)")
    print("Ctrl+C to stop.")
    if open_browser:
        try:
            import webbrowser
            webbrowser.open(local)
        except Exception:
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()
