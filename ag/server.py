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

_PAGE_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Apple-Gorilla</title>
__FONTS__
<style>__THEME__</style>
<style>
#improve .controls{flex-wrap:wrap;gap:8px;align-items:center}
#improveout{margin-top:10px}
.result{padding:12px;border-radius:8px;background:rgba(127,127,127,.08);
  line-height:1.55;font-size:14px}
.result.ok{border-left:3px solid #3fb950}
.result.bad{border-left:3px solid #f85149}
table.hist{width:100%;border-collapse:collapse;margin-top:6px;font-size:13px}
table.hist td{padding:3px 6px;border-bottom:1px solid rgba(127,127,127,.15);
  vertical-align:top}
table.hist td.rat{color:#8b949e;font-style:italic}
#improve label.toggle{font-size:13px;opacity:.85}
#improve select{margin-left:6px}
</style></head><body>
<div class="wrap">
<header>
  <div class="logo">🦍</div>
  <div class="brand"><h1>Apple-Gorilla</h1>
    <div class="sub">self-improving prompt executor</div></div>
  <div id="status">ready</div>
</header>

<div id="update">
  <span class="g" id="updmsg"></span>
  <button class="yes" onclick="applyUpdate()">Yes, update</button>
  <button onclick="document.getElementById('update').style.display='none'">Dismiss</button>
</div>

<div class="panel">
  <textarea id="p" placeholder="Ask Apple-Gorilla anything…"></textarea>
  <div class="controls">
    <button class="primary" id="runbtn" onclick="go()">Run</button>
    <button onclick="loadTools()">Tools &amp; friction</button>
    <label class="toggle"><input type="checkbox" id="web" checked> use internet</label>
  </div>
</div>

<table id="tools" class="panel"></table>

<div class="panel" id="improve">
  <div class="controls">
    <button onclick="doBench()">📊 Benchmark</button>
    <button class="primary" onclick="doEvolve()">🧬 Evolve</button>
    <button onclick="loadHistory()">📜 History</button>
    <button onclick="loadDoctor()">🩺 Status</button>
    <label class="toggle">proposer
      <select id="proposer">
        <option value="">deploy backend</option>
        <option value="anthropic">Claude (anthropic)</option>
        <option value="ollama">Ollama (local)</option>
      </select>
    </label>
  </div>
  <div id="improveout"></div>
</div>

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

<h2>Live trace · thoughts · tool &amp; internet calls · errors</h2>
<div class="panel" style="padding:8px"><div id="log"></div></div>

<h2>Answer</h2>
<div class="panel" id="answer"></div>
</div>

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
    checkUpdate();   // one on-demand check AFTER the run — never a background poll
    return;
  }
  addEv(ev);
  const sc=(ev.data&&ev.data.scorecard);
  if(sc){ $('cards').style.display='flex';
    setCard('acc',sc.accuracy); setCard('qual',sc.quality);
    setCard('spd',sc.speed); setCard('ovr',sc.overall); }
}
async function checkUpdate(){
  try{
    const s=await (await fetch('/update/check')).json();
    if(s.available){
      $('updmsg').textContent='A newer build of '+s.model+' is available. Update now?';
      $('update').style.display='flex';
    }
  }catch(e){ /* check is best-effort; stay silent on failure */ }
}
async function applyUpdate(){
  $('update').style.display='none';
  addEv({stage:'update',level:'tool',msg:'starting model update…'});
  try{
    const r=await fetch('/update/apply',{method:'POST'});
    const reader=r.body.getReader(), dec=new TextDecoder(); let buf='';
    while(true){
      const {value,done}=await reader.read(); if(done)break;
      buf+=dec.decode(value,{stream:true}); let nl;
      while((nl=buf.indexOf('\\n'))>=0){
        const line=buf.slice(0,nl).trim(); buf=buf.slice(nl+1);
        if(!line)continue; const ev=JSON.parse(line);
        addEv(ev.stage==='done'?{stage:'update',level:ev.level,msg:'update: '+ev.msg}:ev);
      }
    }
  }catch(e){ addEv({stage:'error',level:'error',msg:'update failed: '+e}); }
}
async function streamPost(url, body, onEvent){
  const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify(body||{})});
  const reader=r.body.getReader(), dec=new TextDecoder(); let buf='';
  while(true){ const {value,done}=await reader.read(); if(done)break;
    buf+=dec.decode(value,{stream:true}); let nl;
    while((nl=buf.indexOf('\\n'))>=0){ const line=buf.slice(0,nl).trim(); buf=buf.slice(nl+1);
      if(line) onEvent(JSON.parse(line)); } }
}
async function doBench(){
  $('log').innerHTML=''; $('improveout').innerHTML=''; $('status').textContent='benchmarking…';
  try{ await streamPost('/bench',{}, ev=>{
    if(ev.stage==='done'){ const d=ev.data||{};
      $('improveout').innerHTML='<div class="result"><b>Fitness '+d.fitness+'/10</b> · '
        +d.passed+'/'+d.n+' passed'
        +(d.failing&&d.failing.length?'<br>failing: '+escapeHtml(d.failing.join(', ')):'')+'</div>';
      $('status').textContent='bench: '+d.fitness+'/10'; return; }
    addEv(ev);
  }); }catch(e){ addEv({stage:'error',level:'error',msg:'bench failed: '+e});
    $('status').textContent='error'; }
}
async function doEvolve(){
  $('log').innerHTML=''; $('improveout').innerHTML=''; $('status').textContent='evolving…';
  try{ await streamPost('/evolve',{proposer:$('proposer').value}, ev=>{
    if(ev.stage==='done'){ const d=ev.data||{};
      const cls=d.adopted?'ok':(d.rolled_back?'bad':'');
      let h='<div class="result '+cls+'"><b>'
        +(d.adopted?'✓ Adopted':(d.rolled_back?'↩ Rolled back':'No change'))
        +'</b><br>'+escapeHtml(d.reason||'')+'<br>';
      if(d.incumbent!=null) h+='fitness '+d.incumbent+'→'+(d.candidate!=null?d.candidate:'?')
        +'/10'+(d.delta!=null?' (Δ'+d.delta+')':'')+'<br>';
      if(d.changed&&d.changed.length) h+='files: '+escapeHtml(d.changed.join(', '))+'<br>';
      if(d.rationale) h+='<i>'+escapeHtml(d.rationale)+'</i>';
      h+='</div>'; $('improveout').innerHTML=h;
      $('status').textContent='evolve: '+(d.adopted?'adopted':(d.rolled_back?'rolled back':'no change'));
      return; }
    addEv(ev);
  }); }catch(e){ addEv({stage:'error',level:'error',msg:'evolve failed: '+e});
    $('status').textContent='error'; }
}
async function loadHistory(){
  const d=await (await fetch('/evolve/history')).json();
  if(!d.history||!d.history.length){
    $('improveout').innerHTML='<div class="result">No evolution history yet — click Evolve.</div>';
    return; }
  let h='<div class="result"><b>Fitness lineage</b> (newest first)<table class="hist">';
  for(const r of d.history){
    const mark=r.adopted?'✓ adopted':'· '+(r.verdict||'n/a');
    let fit='';
    if(r.incumbent_fitness!=null&&r.candidate_fitness!=null)
      fit=r.incumbent_fitness+'→'+r.candidate_fitness+'/10 (Δ'+r.delta+')';
    h+='<tr><td>'+escapeHtml(r.ts||'')+'</td><td>'+mark+'</td><td>'+fit+'</td></tr>';
    if(r.rationale) h+='<tr><td colspan="3" class="rat">'+escapeHtml(r.rationale.slice(0,140))+'</td></tr>';
  }
  h+='</table></div>'; $('improveout').innerHTML=h;
}
async function loadDoctor(){
  const d=await (await fetch('/doctor')).json();
  let h='<div class="result"><b>Status</b><br>';
  h+='backend: '+d.backend+' → '+d.effective_backend+'<br>';
  h+='ollama: '+(d.ollama_reachable?('reachable — '+(d.ollama_models.join(', ')||'no models pulled'))
    :'not reachable')+'<br>';
  h+='fitness gate: '+(d.fitness_gate?'ON':'off')+' — '+d.bench_tasks+' tasks, '
    +d.bench_samples+' samples ('+d.bench_mode+')<br>';
  h+='evolve gate: '+d.evolve_gate+' · snapshots: '+d.snapshots+'<br>';
  if(d.evolver_backend) h+='proposer backend: '+d.evolver_backend+'<br>';
  if(d.last_improvement){ const l=d.last_improvement;
    h+='last improvement: Δ'+l.delta+' → '+l.candidate_fitness+'/10<br>'; }
  h+='</div>'; $('improveout').innerHTML=h;
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
  h+='<tr><td colspan="5">avg integration '+d.avg_integration
    +' · avg friction '+d.avg_friction+' · best evolve target: '
    +escapeHtml(d.highest_friction_wired||'–')+'</td></tr>';
  t.innerHTML=h; t.style.display='table';
}
</script></body></html>"""


def _render_page() -> str:
    """Assemble the page from the evolvable theme (fonts + CSS) and the template."""
    from . import theme
    return (_PAGE_TEMPLATE
            .replace("__FONTS__", theme.FONT_LINK)
            .replace("__THEME__", theme.THEME_CSS))


PAGE = _render_page()


def _build_broker(cfg: Config, *, web=None):
    web = cfg.allow_web if web is None else web
    if not (web or cfg.allow_local_tools):
        return None
    from .permissions import PermissionBroker
    broker = PermissionBroker(allow_external_tools=True)
    if web:
        broker.grant("network")
    if cfg.allow_local_tools:
        broker.grant("filesystem_read")
        broker.grant("code_exec")
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
        elif self.path == "/update/check":
            # On-demand (the page pings this once after a run). Never polled.
            from . import update
            status = update.check_model_update(self.cfg)
            self._send(200, json.dumps(status.as_dict()), "application/json")
        elif self.path == "/doctor":
            self._send(200, json.dumps(_doctor_data(self.cfg)), "application/json")
        elif self.path == "/evolve/history":
            from . import archive
            self._send(200, json.dumps({"history": archive.history(limit=20)}),
                       "application/json")
        else:
            self._send(404, "not found", "text/plain")

    def do_POST(self):
        if self.path == "/update/apply":
            self._stream_update()
            return
        if self.path == "/bench":
            # Drain the request body first: closing the socket with an unread body
            # makes the client see a TCP reset instead of the streamed response.
            try:
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
            except Exception:
                pass
            self._stream_bench()
            return
        if self.path == "/evolve":
            # Evolve self-modifies source + commits, so gate it to the local machine
            # even when the app is bound to 0.0.0.0 for phone/LAN *viewing*.
            if self.client_address and self.client_address[0] not in ("127.0.0.1", "::1"):
                self._send(403, json.dumps({"error": "evolve is local-only"}),
                           "application/json")
                return
            try:
                n = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(n) or b"{}")
            except Exception:
                payload = {}
            self._stream_evolve(payload)
            return
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

    def _ndjson_writer(self):
        """Begin a streamed NDJSON response and return a write(event) callback."""
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def write(ev: dict) -> None:
            self.wfile.write((json.dumps(ev) + "\n").encode("utf-8"))
            self.wfile.flush()
        return write

    def _stream_update(self):
        """Pull/update the configured Ollama model, streaming progress (user-approved
        via the page's yes/no click — this endpoint only runs on an explicit POST)."""
        from . import update
        write = self._ndjson_writer()
        cfg = self.cfg
        try:
            write({"stage": "update", "level": "tool",
                   "msg": f"updating {cfg.ollama_model}…", "data": {}})
            ok, final = update.pull_model(cfg, emit=write)
            write({"stage": "done", "level": "result" if ok else "error",
                   "msg": final, "data": {"ok": ok, "final": final}})
        except (BrokenPipeError, ConnectionError):
            return
        except Exception as e:
            try:
                write({"stage": "error", "level": "error", "msg": str(e), "data": {}})
            except Exception:
                pass

    def _stream_bench(self):
        """Score AG on its objective benchmark, streaming per-task progress."""
        from . import bench
        write = self._ndjson_writer()
        cfg = self.cfg
        try:
            client = make_client(cfg)
            res = bench.run_benchmark(client, cfg, emit=write)
            write({"stage": "done", "level": "result", "msg": "benchmark complete",
                   "data": {"fitness": res.fitness, "pass_rate": res.pass_rate,
                            "passed": res.passed, "n": res.n, "mode": res.mode,
                            "failing": res.failed_ids}})
        except (BrokenPipeError, ConnectionError):
            return
        except Exception as e:
            try:
                write({"stage": "error", "level": "error", "msg": str(e), "data": {}})
            except Exception:
                pass

    def _stream_evolve(self, payload):
        """Run one gated self-improvement cycle, streaming its progress + verdict."""
        from . import evolve as evolve_mod
        write = self._ndjson_writer()
        cfg = self.cfg
        try:
            client = make_client(cfg)
            # Optional split backend: a stronger model proposes; deploy backend measures.
            evolver_client = None
            prop = str((payload or {}).get("proposer", "")).strip()
            if prop and prop != cfg.backend:
                try:
                    evolver_client = make_client(cfg, backend=prop)
                    write({"stage": "evolve", "level": "tool", "data": {},
                           "msg": f"proposer: {prop}  |  fitness measured on: {cfg.backend}"})
                except Exception as e:
                    write({"stage": "evolve", "level": "error", "data": {},
                           "msg": f"proposer '{prop}' unavailable ({e}); using deploy backend"})
            write({"stage": "evolve", "level": "tool",
                   "msg": "starting self-improvement cycle…", "data": {}})
            res = evolve_mod.evolve(client, cfg, emit=write, evolver_client=evolver_client)
            write({"stage": "done",
                   "level": "result" if res.adopted else "info",
                   "msg": res.reason, "data": {
                       "adopted": res.adopted, "rolled_back": res.rolled_back,
                       "verdict": res.verdict, "incumbent": res.incumbent_fitness,
                       "candidate": res.candidate_fitness, "delta": res.fitness_delta,
                       "changed": res.changed, "rationale": res.rationale,
                       "reason": res.reason, "snapshot_id": res.snapshot_id}})
        except (BrokenPipeError, ConnectionError):
            return
        except Exception as e:
            try:
                write({"stage": "error", "level": "error", "msg": str(e), "data": {}})
            except Exception:
                pass

    def _stream_run(self, prompt: str, want_web):
        """Run the pipeline, streaming each stage event as one NDJSON line."""
        write = self._ndjson_writer()
        cfg = self.cfg
        web_eff = cfg.allow_web if want_web is None else bool(want_web)
        broker = _build_broker(cfg, web=web_eff)
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


def _doctor_data(cfg: Config) -> dict:
    """Environment / readiness snapshot for the GUI Status button (same facts as the
    `ag doctor` CLI). Read-only; probes Ollama without hard-failing."""
    import os
    import urllib.request
    from . import archive, backup, bench
    from .evolve import gate_available
    from .model import has_oauth_profile, oauth_token_status

    has_key = bool(os.environ.get("ANTHROPIC_API_KEY")
                   or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    oauth = has_oauth_profile()
    reachable, models = False, []
    try:
        with urllib.request.urlopen(cfg.ollama_host.rstrip("/") + "/api/tags",
                                    timeout=1.5) as r:
            reachable = True
            models = [m.get("name", "?")
                      for m in json.loads(r.read().decode("utf-8")).get("models", [])]
    except Exception:
        pass
    if cfg.backend == "auto":
        eff = "anthropic" if (has_key or oauth) else "dry-run (stub)"
    else:
        eff = cfg.backend
    last = archive.adopted_history(limit=1)
    return {
        "backend": cfg.backend, "effective_backend": eff, "model": cfg.model,
        "ollama_model": cfg.ollama_model, "ollama_reachable": reachable,
        "ollama_models": models, "api_key": has_key, "oauth": oauth,
        "oauth_token": oauth_token_status() if oauth else "none",
        "autonomy": cfg.autonomy_level, "evolver_backend": cfg.evolver_backend,
        "fitness_gate": cfg.fitness_gate, "bench_tasks": len(bench.load_tasks()),
        "bench_mode": cfg.bench_mode, "bench_samples": cfg.bench_samples,
        "evolve_gate": "ready" if gate_available() else "unavailable (pip install pytest)",
        "snapshots": len(backup.list_snapshots()),
        "last_improvement": last[0] if last else None, "allow_web": cfg.allow_web,
    }


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
