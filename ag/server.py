"""A tiny built-in web app for AG (stdlib only).

`python -m ag serve` starts a local server with a browser UI. Open it on this
machine, or from your phone/another device on the same network (bind --host
0.0.0.0). Cross-platform by virtue of being a web page — any OS with a browser.

The UI drives the same pipeline (optimize -> execute -> critique -> iterate) and
streams a **realtime log** of AG's thought process, tool/app calls, internet usage,
errors, and the accuracy/quality/speed scorecard as they happen (newline-delimited
JSON over a single streamed response). This IS an inbound server (for YOUR use); it
is not a data-farming endpoint — bind to localhost unless you deliberately want
LAN/phone access.
"""
from __future__ import annotations

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
 :root{color-scheme:light dark;--bd:#8884;--muted:#8889;
   --tool:#3b82f6;--web:#14b8a6;--err:#ef4444;--res:#22c55e;--info:#9ca3af}
 *{box-sizing:border-box}
 body{font-family:system-ui,sans-serif;max-width:860px;margin:0 auto;padding:16px}
 h1{font-size:1.3rem;margin:.2rem 0}
 .row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
 textarea{width:100%;min-height:80px;font-size:1rem;padding:8px;border-radius:8px;
   border:1px solid var(--bd);background:transparent;color:inherit}
 button{font-size:.95rem;padding:9px 15px;cursor:pointer;border-radius:8px;
   border:1px solid var(--bd);background:transparent;color:inherit}
 button.primary{background:#2563eb;color:#fff;border-color:#2563eb}
 .meta{color:var(--muted);font-size:.85rem}
 label.web{font-size:.85rem;color:var(--muted);display:flex;gap:5px;align-items:center}
 h2{font-size:.9rem;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);
   margin:18px 0 6px}
 #log{border:1px solid var(--bd);border-radius:8px;padding:8px;max-height:320px;
   overflow:auto;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:.82rem}
 .ev{padding:3px 8px;border-left:3px solid var(--info);margin:2px 0;white-space:pre-wrap}
 .ev .tag{display:inline-block;min-width:74px;color:var(--muted)}
 .ev.tool{border-color:var(--tool)} .ev.web{border-color:var(--web)}
 .ev.error{border-color:var(--err);color:var(--err)}
 .ev.result{border-color:var(--res)}
 #cards{display:none;gap:14px;margin-top:8px}
 .card{flex:1;min-width:120px}
 .card .n{font-size:1.5rem;font-weight:600} .card .l{font-size:.75rem;color:var(--muted)}
 .bar{height:6px;border-radius:3px;background:var(--bd);margin-top:4px;overflow:hidden}
 .bar>i{display:block;height:100%;background:var(--res)}
 #answer{white-space:pre-wrap;margin-top:6px;padding:12px;border:1px solid var(--bd);
   border-radius:8px;min-height:40px}
 #tools{display:none;margin-top:8px;font-size:.85rem;border-collapse:collapse;width:100%}
 #tools th,#tools td{text-align:left;padding:4px 8px;border-bottom:1px solid var(--bd)}
 #tools td.n{text-align:right}
 .pill{font-size:.72rem;padding:1px 7px;border-radius:20px;border:1px solid var(--bd)}
 .pill.available{color:var(--res)} .pill.degraded{color:#f59e0b}
 .pill.unavailable{color:var(--muted)}
</style></head><body>
<h1>🦍 Apple-Gorilla</h1>
<div class="meta" id="status">ready</div>
<textarea id="p" placeholder="Ask AG anything..."></textarea>
<div class="row" style="margin-top:8px">
  <button class="primary" id="runbtn" onclick="go()">Run</button>
  <button onclick="loadTools()">Tools &amp; friction</button>
  <label class="web"><input type="checkbox" id="web" checked> use internet</label>
</div>

<table id="tools"></table>

<h2>Live log — thoughts · tool calls · internet · errors</h2>
<div id="log"></div>

<div id="cards">
  <div class="card"><div class="n" id="acc">–</div><div class="l">accuracy</div>
    <div class="bar"><i id="accb"></i></div></div>
  <div class="card"><div class="n" id="qual">–</div><div class="l">quality</div>
    <div class="bar"><i id="qualb"></i></div></div>
  <div class="card"><div class="n" id="spd">–</div><div class="l">speed</div>
    <div class="bar"><i id="spdb"></i></div></div>
  <div class="card"><div class="n" id="ovr">–</div><div class="l">overall</div>
    <div class="bar"><i id="ovrb"></i></div></div>
</div>

<h2>Answer</h2>
<div id="answer"></div>

<script>
const $=id=>document.getElementById(id);
function addEv(ev){
  const d=document.createElement('div');
  d.className='ev '+(ev.level||'info');
  d.innerHTML='<span class="tag">'+ev.stage+'</span>'+escapeHtml(ev.msg);
  $('log').appendChild(d); $('log').scrollTop=$('log').scrollHeight;
}
function escapeHtml(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function setCard(id,v){ $(id).textContent=(v==null?'–':v);
  $(id+'b').style.width=((Number(v)||0)*10)+'%'; }
async function go(){
  const p=$('p').value.trim(); if(!p)return;
  $('runbtn').disabled=true; $('status').textContent='running…';
  $('log').innerHTML=''; $('answer').textContent=''; $('cards').style.display='none';
  ['acc','qual','spd','ovr'].forEach(x=>setCard(x,null));
  try{
    const r=await fetch('/run',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({prompt:p,web:$('web').checked})});
    const reader=r.body.getReader(), dec=new TextDecoder(); let buf='';
    while(true){
      const {value,done}=await reader.read(); if(done)break;
      buf+=dec.decode(value,{stream:true}); let nl;
      while((nl=buf.indexOf('\\n'))>=0){
        const line=buf.slice(0,nl).trim(); buf=buf.slice(nl+1);
        if(!line)continue; handle(JSON.parse(line));
      }
    }
  }catch(e){ addEv({stage:'error',level:'error',msg:'request failed: '+e}); $('status').textContent='error'; }
  $('runbtn').disabled=false;
}
function handle(ev){
  if(ev.stage==='done'){
    const d=ev.data||{}; $('answer').textContent=d.answer||'(no answer)';
    $('status').textContent='done · '+(d.iterations||0)+' iter · '+(d.elapsed_s||0)+'s'
      +(d.dry_run?' · dry-run':'');
    return;
  }
  addEv(ev);
  const sc=(ev.data&&ev.data.scorecard);
  if(sc){ $('cards').style.display='flex';
    setCard('acc',sc.accuracy); setCard('qual',sc.quality);
    setCard('spd',sc.speed); setCard('ovr',sc.overall); }
}
async function loadTools(){
  const t=$('tools');
  if(t.style.display==='table'){ t.style.display='none'; return; }
  const d=await (await fetch('/tools')).json();
  let h='<tr><th>tool</th><th>category</th><th>status</th>'
    +'<th class="n">integ</th><th class="n">friction</th></tr>';
  for(const x of d.tools){ h+='<tr><td>'+escapeHtml(x.name)+'</td><td>'+x.category
    +'</td><td><span class="pill '+x.status+'">'+x.status+'</span></td>'
    +'<td class="n">'+x.integration+'</td><td class="n">'+x.friction+'</td></tr>'; }
  h+='<tr><td colspan="5" class="meta">avg integration '+d.avg_integration
    +' · avg friction '+d.avg_friction+' · best evolve target: '
    +escapeHtml(d.highest_friction_wired||'–')+'</td></tr>';
  t.innerHTML=h; t.style.display='table';
}
</script></body></html>"""


def _build_broker(cfg: Config):
    if not cfg.allow_web:
        return None
    from .permissions import PermissionBroker
    broker = PermissionBroker(allow_external_tools=True)
    broker.grant("network")
    return broker


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
        elif self.path == "/tools":
            from . import inventory
            self._send(200, json.dumps(inventory.summary(self.cfg)),
                       "application/json")
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
        except Exception as e:
            self._send(200, json.dumps({"error": str(e)}), "application/json")
            return
        want_web = payload.get("web", None)
        self._stream_run(prompt, want_web)

    def _stream_run(self, prompt: str, want_web):
        """Run the pipeline, streaming each stage event as one NDJSON line."""
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def write(ev: dict) -> None:
            self.wfile.write((json.dumps(ev) + "\n").encode("utf-8"))
            self.wfile.flush()

        cfg = self.cfg
        web_eff = cfg.allow_web if want_web is None else bool(want_web)
        broker = _build_broker(cfg) if web_eff else None
        try:
            client = make_client(cfg)
            rec = run_pipeline(client, cfg, prompt, web=web_eff, broker=broker,
                               emit=write)
            write({"stage": "done", "level": "result", "msg": "done", "data": {
                "answer": rec.answer, "scorecard": rec.scorecard,
                "iterations": rec.iterations, "elapsed_s": rec.elapsed_s,
                "dry_run": rec.dry_run}})
        except (BrokenPipeError, ConnectionError):
            return  # client navigated away mid-stream
        except Exception as e:
            try:
                write({"stage": "error", "level": "error", "msg": str(e), "data": {}})
            except Exception:
                pass

    def log_message(self, *a):  # quiet
        pass


def run_prompt(cfg: Config, prompt: str) -> tuple[str, str]:
    """Run one prompt through the pipeline; returns (answer, meta-string).

    Non-streaming helper kept for programmatic use and tests; the web UI uses the
    streaming path in `_Handler._stream_run`.
    """
    client = make_client(cfg)
    broker = _build_broker(cfg)
    rec = run_pipeline(client, cfg, prompt, web=cfg.allow_web, broker=broker)
    sc = rec.scorecard or {}
    meta = (f"iterations={rec.iterations} dry_run={rec.dry_run} "
            f"overall={sc.get('overall', '?')}")
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
