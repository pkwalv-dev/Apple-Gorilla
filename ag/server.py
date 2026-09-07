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
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .config import Config
from .model import make_client, Canceller, Cancelled
from .pipeline import run as run_pipeline


class _Interrupted(Exception):
    """Raised inside a run's on_delta when the viewer disconnects (Stop / closed tab),
    so generation is halted upstream instead of running to completion unseen."""

_PAGE_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Apple-Gorilla</title>
__FONTS__
<style>__THEME__</style>
<style>
#improveout{margin-top:12px}
.result{padding:12px;border-radius:8px;background:rgba(127,127,127,.08);
  line-height:1.55;font-size:14px}
.result.ok{border-left:3px solid #3fb950}
.result.bad{border-left:3px solid #f85149}
table.hist{width:100%;border-collapse:collapse;margin-top:6px;font-size:13px}
table.hist td{padding:3px 6px;border-bottom:1px solid rgba(127,127,127,.15);
  vertical-align:top}
table.hist td.rat{color:#8b949e;font-style:italic}
#improve label.toggle{font-size:13px;opacity:.85;margin-left:auto}
#improve select{margin-left:6px}
/* live activity indicator on the pending reply bubble */
.body.pending{opacity:.85;font-style:italic}
.act-timer{opacity:.6;font-variant-numeric:tabular-nums;font-style:normal}
/* verbose live output (reasoning + streamed tokens) */
.verbose{margin-top:8px;font-size:12px}
.verbose summary{cursor:pointer;color:#8b949e}
.vbody{white-space:pre-wrap;max-height:260px;overflow:auto;margin-top:6px;padding:8px;
  border-radius:8px;background:rgba(127,127,127,.08);font-family:var(--mono),monospace;
  font-size:11px;line-height:1.45}
#stopbtn{background:#b91c1c;color:#fff}
</style></head><body>
<div class="wrap">
<header>
  <div class="logo">AG</div>
  <div class="brand"><h1>Apple-Gorilla<span class="ver" title="app version">v__VERSION__</span></h1>
    <div class="sub">self-improving prompt executor</div></div>
  <div id="status">ready</div>
</header>

<div id="update">
  <span class="g" id="updmsg"></span>
  <button class="yes" onclick="applyUpdate()">Yes, update</button>
  <button onclick="document.getElementById('update').style.display='none'">Dismiss</button>
</div>

<!-- where & how AG is running right now (populated from /whereami) -->
<div id="whereami" class="whereami" title="Where and how AG is running right now"></div>

<!-- when & how self-evolution is happening (populated from /doctor + /evolve/history) -->
<div id="evostatus" class="evostatus" title="When and how AG evolves itself">
  <span class="dot"></span><b>Evolve</b> loading status…
</div>

<!-- Claude sign-in (populated from /auth); lets `auto` use Claude instead of local -->
<div id="signin" class="whereami" title="Sign in so AG's auto backend uses Claude" hidden></div>

<div class="panel">
  <textarea id="p" placeholder="Ask Apple-Gorilla anything…"></textarea>
  <div class="controls">
    <button class="cmd primary" id="runbtn" onclick="go()">Run</button>
    <button class="cmd" id="stopbtn" onclick="stopRun()" hidden>Stop</button>
    <span class="cmd-note">executes a request</span>
    <span class="ctl-right">
      <label class="ctl" title="Stream the model's raw output — including its reasoning (Ollama's &lt;think&gt; blocks) — live as it is generated. Stop ends the response and returns control immediately; a local model may take a moment more to wind down in the background."><input type="checkbox" id="verbose"> show reasoning</label>
      <label class="ctl" title="Fast = ONE model call (no prompt-engineering, no self-review) — quick and best for iterating. Full = engineer the prompt, then self-critique and revise for higher quality (several calls, much slower). Context (web/profile/memory/history) applies in both.">mode
        <select id="mode" class="ctl-select" onchange="saveMode()">
          <option value="fast">fast · 1 call</option>
          <option value="full">full · review</option>
        </select>
      </label>
      <label class="ctl" title="Which model answers this run. Local models run offline via Ollama; the Claude cloud option appears when you're signed in. Bigger local models are smarter but slower — watch the activity timer on the reply.">model
        <select id="model" class="ctl-select" onchange="saveModel()">
          <option value="">loading…</option>
        </select>
      </label>
      <label class="ctl" title="Extended thinking — like the toggle in the Claude app. 'off' suppresses the model's step-by-step reasoning (fastest per call); 'on' forces it; 'auto' leaves the model to its default. Independent of the fast/full mode above.">thinking
        <select id="think" class="ctl-select" onchange="saveThink()">
          <option value="auto">auto</option>
          <option value="off">off · fast</option>
          <option value="on">on</option>
        </select>
      </label>
      <label class="ctl"><input type="checkbox" id="web" checked> use internet</label>
    </span>
  </div>
</div>

<h2>Conversation</h2>
<!-- context bar: lights up to show which sources feed the CURRENT answer -->
<div id="ctxbar" title="What AG is drawing on for the current answer — each lights up as it is used">
  <span class="ctxlbl">context in use:</span>
  <span class="chip" id="chip-history">Conversation <b class="cc" id="cc-history"></b></span>
  <span class="chip" id="chip-profile">Profile</span>
  <span class="chip" id="chip-memory">Memory <b class="cc" id="cc-memory"></b></span>
  <span class="chip" id="chip-web">Web <b class="cc" id="cc-web"></b></span>
</div>
<div class="panel chatwrap">
  <div class="chat-toolbar">
    <span class="ct-hint">earlier exchanges are kept here — scroll to revisit</span>
    <button class="linkbtn spacer" onclick="clearChat()">Clear conversation</button>
  </div>
  <div id="chat"></div>

<table id="tools" class="panel"></table>

<div class="panel" id="imagepanel">
  <span class="grouplabel">Image generation — local Stable Diffusion <span class="cmd-note" id="imgnote"></span></span>
  <textarea id="imgprompt" class="directive" rows="2"
    placeholder="Describe an image to generate. Needs a local Stable Diffusion server (Automatic1111/Forge) running with --api; prompts never leave your machine."></textarea>
  <div class="controls">
    <button class="cmd" id="imgbtn" onclick="genImage()">Generate image</button>
  </div>
  <div id="imgout"></div>
</div>

<div class="panel" id="improve">
  <div class="btngroup cmd-group">
    <span class="grouplabel">Commands — run &amp; modify AG</span>
    <textarea id="directive" class="directive" rows="2"
      placeholder="Optional — tell Evolve what to improve in plain text (e.g. "make answers more concise", "sharpen the web-search prompt"). This steers the next Evolve; safety &amp; fitness gates still apply."></textarea>
    <div class="controls">
      <button class="cmd" onclick="doBench()">Benchmark</button>
      <button class="cmd evolve" onclick="doEvolve()">Evolve</button>
      <label class="toggle">proposer
        <select id="proposer">
          <option value="">deploy backend</option>
          <option value="anthropic">Claude (anthropic)</option>
          <option value="ollama">Ollama (local)</option>
        </select>
      </label>
    </div>
  </div>
  <div class="btngroup view-group">
    <span class="grouplabel">Views — read-only, change nothing</span>
    <div class="controls">
      <button class="view" onclick="loadTools()">Tools &amp; friction</button>
      <button class="view" onclick="loadHistory()">History</button>
      <button class="view" onclick="loadDoctor()">Status</button>
    </div>
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

</div>
</div>

<script>
const $=id=>document.getElementById(id);
function escapeHtml(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function addEv(ev){
  const d=document.createElement('div');
  d.className='ev '+(ev.level||'info');
  d.innerHTML='<span class="tag">'+ev.stage+'</span>'+escapeHtml(ev.msg);
  $('log').appendChild(d); $('log').scrollTop=$('log').scrollHeight;
}
function setCard(id,v){ $(id).textContent=(v==null?'–':v);
  $(id+'b').style.width=((Number(v)||0)*10)+'%'; }

/* ---- conversation transcript (persisted locally) ---------------------- */
let CHAT=[];
function loadChat(){
  try{ CHAT=JSON.parse(localStorage.getItem('ag_chat')||'[]'); }catch(e){ CHAT=[]; }
  renderChat();
}
function saveChat(){
  try{ localStorage.setItem('ag_chat',JSON.stringify(CHAT.slice(-100))); }catch(e){}
}
function clearChat(){
  if(!CHAT.length||confirm('Clear the whole conversation?')){ CHAT=[]; saveChat(); renderChat(); }
}
function ctxFooter(c){
  if(!c) return '';
  const bits=[];
  if(c.history) bits.push(''+c.history+' prior turn'+(c.history>1?'s':''));
  if(c.profile) bits.push('profile');
  if(c.memory&&c.memory.length) bits.push(''+c.memory.length+' memory fact'+(c.memory.length>1?'s':''));
  if(c.web) bits.push(''+c.web+' web source'+(c.web>1?'s':''));
  if(c.saved&&c.saved.length) bits.push('remembered '+c.saved.length+' new fact'+(c.saved.length>1?'s':''));
  if(c.overall!=null) bits.push(''+c.overall+' overall');
  let h=bits.length? '<div class="ctx">'+bits.map(b=>'<span>'+escapeHtml(b)+'</span>').join('')+'</div>':'';
  if(c.memory&&c.memory.length){
    h+='<details class="memfacts"><summary>memory facts used</summary><ul>'
      +c.memory.map(f=>'<li>'+escapeHtml(f)+'</li>').join('')+'</ul></details>';
  }
  if(c.saved&&c.saved.length){
    h+='<details class="memfacts"><summary>saved to long-term memory</summary><ul>'
      +c.saved.map(f=>'<li>'+escapeHtml(f)+'</li>').join('')+'</ul></details>';
  }
  return h;
}
function bubble(m){
  const who=m.role==='user'?'you':'apple-gorilla';
  const t=m.ts?'<span class="ts">'+escapeHtml(m.ts)+'</span>':'';
  return '<div class="msg '+m.role+'"><div class="who">'+who+t+'</div>'
    +'<div class="body">'+escapeHtml(m.text)+'</div>'
    +(m.role==='ai'?ctxFooter(m.ctx):'')+'</div>';
}
function renderChat(){
  $('chat').innerHTML=CHAT.map(bubble).join('');
  $('chat').scrollTop=$('chat').scrollHeight;
}
function appendUser(text){
  CHAT.push({role:'user',text:text,ts:nowStr()}); renderChat(); saveChat();
}
function nowStr(){ const d=new Date();
  return d.toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'}); }
// live pending assistant bubble; returns the DOM node so we can fill it as we stream
function appendAssistant(){
  const el=document.createElement('div'); el.className='msg ai processing';
  el.innerHTML='<div class="who">apple-gorilla<span class="ts">'+nowStr()+'</span></div>'
    +'<div class="body pending"><span class="act-stage">…starting</span>'
    +'<span class="act-timer"></span></div>'
    +'<details class="verbose" hidden><summary>reasoning &amp; live output</summary>'
    +'<pre class="vbody"></pre></details>'
    +'<div class="live-ctx"></div>';
  $('chat').appendChild(el); $('chat').scrollTop=$('chat').scrollHeight;
  return el;
}

/* ---- live activity indicator: what the model is doing + how long ------- */
/* Makes a snag visible: the stage label shows the current step and the timer
   keeps ticking, so a stall (timer climbing, stage unchanged) is obvious. */
const STAGE_LABELS={conversation:'reading the conversation',
  web:'searching the web',optimize:'engineering the prompt',
  memory:'recalling memory',execute:'generating the answer',
  reason:'using tools',critique:'reviewing the answer',
  revise:'revising the answer',score:'scoring the answer'};
function fmtDur(ms){const s=Math.floor(ms/1000);
  return s>=60?(Math.floor(s/60)+':'+String(s%60).padStart(2,'0')):(s+'s');}
function startActivity(ai){
  ai._t0=Date.now();
  const tick=()=>{const el=ai.querySelector('.act-timer');
    if(el) el.textContent=' · '+fmtDur(Date.now()-ai._t0);};
  ai._timer=setInterval(tick,1000); tick();
}
function setStage(ai,ev){
  const lbl=STAGE_LABELS[ev.stage]; if(!lbl) return;
  const el=ai&&ai.querySelector('.act-stage');
  if(el){ el.textContent=lbl; ai._lastStage=Date.now(); }
}
function stopActivity(ai){ if(ai&&ai._timer){clearInterval(ai._timer); ai._timer=null;} }

/* ---- context-in-use indicator ---------------------------------------- */
let curCtx={history:0,profile:false,memory:[],saved:[],web:0};
function resetContext(){
  curCtx={history:0,profile:false,memory:[],saved:[],web:0};
  ['history','profile','memory','web'].forEach(n=>$('chip-'+n).classList.remove('active'));
  $('cc-history').textContent=''; $('cc-memory').textContent=''; $('cc-web').textContent='';
}
function applyContext(ev,ai){
  if(ev.stage==='conversation'){
    const m=/(\\d+)\\s+earlier/.exec(ev.msg||''); const n=m?+m[1]:0;
    curCtx.history=n; $('chip-history').classList.add('active');
    $('cc-history').textContent=n||'';
  }
  if(ev.stage==='optimize' && (/(profile)/i.test(ev.msg||'') || (ev.data&&ev.data.uses_profile))){
    curCtx.profile=true; $('chip-profile').classList.add('active');
  }
  if(ev.stage==='memory'){
    const f=(ev.data&&ev.data.facts)||[];
    const saved=(ev.data&&ev.data.saved)||[];
    $('chip-memory').classList.add('active');
    if(f.length){ curCtx.memory=f; $('cc-memory').textContent=f.length;
      if(ai){ const lc=ai.querySelector('.live-ctx');
        if(lc) lc.innerHTML='<div class="using">drawing on '+f.length
          +' remembered fact(s):</div><ul>'+f.map(x=>'<li>'+escapeHtml(x)+'</li>').join('')+'</ul>'; } }
    if(saved.length){ curCtx.saved=saved;
      // the save arrives AFTER the answer is committed — patch the finished bubble
      if(ai&&ai._committed&&ai._entry){
        ai._entry.ctx=ai._entry.ctx||{}; ai._entry.ctx.saved=saved; saveChat();
        ai.insertAdjacentHTML('beforeend',
          '<details class="memfacts" open><summary>saved '+saved.length
          +' fact(s) to long-term memory</summary><ul>'
          +saved.map(f=>'<li>'+escapeHtml(f)+'</li>').join('')+'</ul></details>');
      }
    }
  }
  if(ev.stage==='web'){
    let n=(ev.data&&ev.data.count); if(n==null){ const m=/(\\d+)\\s+result/.exec(ev.msg||''); n=m?+m[1]:null; }
    if(n!=null){ curCtx.web=n; $('chip-web').classList.add('active'); $('cc-web').textContent=n||''; }
  }
}

let CURRENT_ABORT=null, CURRENT_RUNID=null, RUN_STOPPED=false;
function setRunning(on){
  const rb=$('runbtn'), sb=$('stopbtn');
  if(rb){ rb.hidden=on; rb.disabled=on; }
  if(sb){ sb.hidden=!on; }
}
function stopRun(){
  $('status').textContent='stopping…'; RUN_STOPPED=true;
  // Signal the server to halt generation. We deliberately DON'T abort the fetch: we
  // keep reading so the server's stream loop reaches its cancel check and closes the
  // model connection from its own thread (a client abort can wedge the write on some
  // platforms). The server closes the stream right after it cancels.
  if(CURRENT_RUNID){ try{ fetch('/stop',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({run_id:CURRENT_RUNID})}); }catch(e){} }
}
async function go(){
  const p=$('p').value.trim(); if(!p)return;
  setRunning(true); $('status').textContent='running…';
  $('log').innerHTML=''; $('cards').style.display='none';
  ['acc','qual','spd','ovr'].forEach(x=>setCard(x,null));
  // prior turns become AG's working memory (the current prompt is sent separately)
  const hist=CHAT.slice(-20).map(m=>({role:m.role,text:m.text}));
  resetContext(); appendUser(p); const ai=appendAssistant(); startActivity(ai); $('p').value='';
  const ctrl=new AbortController(); CURRENT_ABORT=ctrl; RUN_STOPPED=false;
  const runId=(self.crypto&&crypto.randomUUID)?crypto.randomUUID():String(Date.now())+Math.random();
  CURRENT_RUNID=runId;
  const verbose=$('verbose')?$('verbose').checked:false;
  try{
    const r=await fetch('/run',{method:'POST',headers:{'Content-Type':'application/json'},
      signal:ctrl.signal,
      body:JSON.stringify({prompt:p,web:$('web').checked,history:hist,think:$('think').value,
        model:($('model')?$('model').value:''),mode:($('mode')?$('mode').value:''),
        verbose:verbose,run_id:runId})});
    const reader=r.body.getReader(), dec=new TextDecoder(); let buf='';
    while(true){
      const {value,done}=await reader.read(); if(done)break;
      buf+=dec.decode(value,{stream:true}); let nl;
      while((nl=buf.indexOf('\\n'))>=0){
        const line=buf.slice(0,nl).trim(); buf=buf.slice(nl+1);
        if(!line)continue; handle(JSON.parse(line),ai);
      }
    }
    // Stream ended without a final answer (e.g. dropped connection): say so plainly
    // rather than leaving the bubble stuck on the activity indicator forever.
    if(!ai._committed){ stopActivity(ai);
      const b=ai.querySelector('.body');
      if(b&&b.classList.contains('pending')){ b.className='body';
        b.textContent=RUN_STOPPED?'(stopped by you)'
          :'(the run ended without an answer — see the trace above)';
        ai.classList.remove('processing');
        $('status').textContent=RUN_STOPPED?'stopped':'ended'; } }
  }catch(e){
    stopActivity(ai);
    const b=ai.querySelector('.body'); b.className='body'; ai.classList.remove('processing');
    if(e&&e.name==='AbortError'){          // user hit Stop
      b.textContent='(stopped by you)'; $('status').textContent='stopped';
    } else {
      addEv({stage:'error',level:'error',msg:'request failed: '+e});
      $('status').textContent='error';
      b.textContent='(request failed: '+escapeHtml(''+e)+')';
    }
  }
  CURRENT_ABORT=null; CURRENT_RUNID=null; setRunning(false);
}
function handle(ev,ai){
  if(ev.stage==='delta'){   // live streamed model output (verbose mode)
    const det=ai.querySelector('.verbose');
    if(det){ det.hidden=false; det.open=true;
      const pre=det.querySelector('.vbody');
      if(pre){ pre.textContent+=((ev.data&&ev.data.text)||''); } }
    $('chat').scrollTop=$('chat').scrollHeight; return;
  }
  if(ev.stage==='progress'){ return; }   // heartbeat only (keeps Stop responsive)
  if(ev.stage==='error'){   // terminal pipeline error — make it visible, don't hang
    stopActivity(ai);
    const b=ai.querySelector('.body'); b.className='body';
    b.textContent=''+(ev.msg||'the run failed'); ai.classList.remove('processing');
    addEv(ev); $('status').textContent='error'; return;
  }
  if(ev.stage==='done'){
    stopActivity(ai);
    const d=ev.data||{};
    const b=ai.querySelector('.body'); b.className='body'; b.textContent=d.answer||'(no answer)';
    ai.classList.remove('processing');
    const lc=ai.querySelector('.live-ctx'); if(lc) lc.remove();
    const sc=d.scorecard||{};
    const ctx={history:curCtx.history,profile:curCtx.profile,memory:curCtx.memory,
               web:curCtx.web,saved:curCtx.saved,
               overall:(sc.overall!=null?sc.overall:null)};
    ai.insertAdjacentHTML('beforeend',ctxFooter(ctx));
    const entry={role:'ai',text:d.answer||'(no answer)',ctx:ctx,ts:nowStr()};
    CHAT.push(entry); ai._entry=entry; ai._committed=true; saveChat();
    $('chat').scrollTop=$('chat').scrollHeight;
    $('status').textContent='done · '+(d.iterations||0)+' iter · '+(d.elapsed_s||0)+'s'
      +(d.dry_run?' · dry-run':'');
    checkUpdate();   // one on-demand check AFTER the run — never a background poll
    return;
  }
  setStage(ai,ev);
  applyContext(ev,ai);
  addEv(ev);
  const sc=(ev.data&&ev.data.scorecard);
  if(sc){ $('cards').style.display='flex';
    setCard('acc',sc.accuracy); setCard('qual',sc.quality);
    setCard('spd',sc.speed); setCard('ovr',sc.overall); }
}

/* ---- where & how AG is running right now ------------------------------ */
async function renderWhere(){
  try{
    const w=await fetch('/whereami').then(r=>r.json());
    let h='<span class="dot ok"></span>running at <b>'+escapeHtml(w.url)+'</b>';
    h+=w.local_only?' <span class="tagpill local">local-only</span>'
      :' <span class="tagpill lan">also on LAN: '+escapeHtml(w.lan_url||'')+'</span>';
    h+=' · brain <b>'+escapeHtml(w.brain)+'</b>';
    h+=w.evolving?' · <span class="tagpill evolving">evolving now — runs still work</span>'
      :' · <span class="tagpill idle">can evolve while running</span>';
    $('whereami').innerHTML=h;
    const eb=document.querySelector('button.cmd.evolve');
    if(eb) eb.disabled=!!w.evolving;
    return w;
  }catch(e){ $('whereami').innerHTML=''; return null; }
}

/* ---- evolve status: when & how self-improvement runs ------------------ */
async function renderEvoStatus(){
  const es=$('evostatus');
  try{
    const [doc,hist]=await Promise.all([
      fetch('/doctor').then(r=>r.json()),
      fetch('/evolve/history').then(r=>r.json())
    ]);
    const last=(hist.history||[])[0];
    const ready=(doc.evolve_gate||'').indexOf('ready')===0;
    let h='<span class="dot '+(ready?'ok':'off')+'"></span><b>Evolve</b> ';
    h+='gate '+(ready?'<span class="g-ok">ready</span>':'<span class="g-off">'+escapeHtml(doc.evolve_gate||'?')+'</span>');
    h+=' · proposer <b>'+escapeHtml(doc.evolver_backend||doc.effective_backend||doc.backend||'?')+'</b>';
    h+=' · verified by <b>'+(doc.fitness_gate?'benchmark + tests':'tests')+'</b>';
    h+=' · '+doc.snapshots+' snapshot(s)';
    if(last){
      const v=last.adopted?'OK adopted':escapeHtml(last.verdict||'no change');
      const dl=(last.delta!=null?' Δ'+last.delta:'');
      h+=' — last run <b>'+escapeHtml(last.ts||'')+'</b>: '+v+dl;
    }else{ h+=' — <i>no evolve runs yet</i>'; }
    es.classList.remove('busy'); es.innerHTML=h;
  }catch(e){ es.classList.remove('busy');
    es.innerHTML='<span class="dot off"></span><b>Evolve</b> status unavailable'; }
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
// Step 1: ask AG to PROPOSE changes — nothing is applied. You then pick which to keep.
async function doEvolve(){
  $('log').innerHTML=''; $('improveout').innerHTML=''; $('status').textContent='proposing…';
  const directive=$('directive').value.trim();
  const es=$('evostatus'); es.classList.add('busy');
  es.innerHTML='<span class="dot spin"></span><b>Proposing changes…</b> '
    +(directive?'toward your request':'analysing AG');
  renderWhere();
  try{ await streamPost('/evolve/propose',{proposer:$('proposer').value,directive:directive}, ev=>{
    if(ev.stage==='done'){ renderProposal(ev.data||{});
      $('status').textContent='proposed'; renderEvoStatus(); renderWhere(); return; }
    if(ev.stage==='evolve'||ev.stage==='bench'){
      es.innerHTML='<span class="dot spin"></span><b>Proposing changes…</b> '+escapeHtml(ev.msg||''); }
    addEv(ev);
  }); }catch(e){ addEv({stage:'error',level:'error',msg:'propose failed: '+e});
    $('status').textContent='error'; renderEvoStatus(); renderWhere(); }
}
// Render the proposed changes as a checklist — YOU decide which to apply.
function renderProposal(d){
  const pts=d.patches||[];
  if(d.busy){ $('improveout').innerHTML='<div class="result">'+escapeHtml(d.reason||'evolve already running')+'</div>'; return; }
  if(!pts.length){ $('improveout').innerHTML='<div class="result">'+escapeHtml(d.reason||'no changes proposed')+'</div>'; return; }
  let h='<div class="proposal"><div class="phead"><b>Proposed changes — select the ones you want</b>'
    +'<span class="pnote">nothing is applied until you click Apply selected</span></div>';
  if(d.rationale) h+='<div class="prationale"><b>AG\\'s rationale:</b> '+escapeHtml(d.rationale)+'</div>';
  for(const p of pts){
    const dis=p.valid?'':'disabled';
    h+='<label class="pitem'+(p.valid?'':' invalid')+'">'
      +'<input type="checkbox" class="psel" value="'+escapeHtml(p.id)+'" '+(p.valid?'checked':'disabled')+'>'
      +'<span class="pmeta"><code>'+escapeHtml(p.path)+'</code> '
      +'<span class="pbytes">'+(p.bytes||0)+' B</span>'
      +(p.valid?'':'<span class="pbad">'+escapeHtml(p.error||'invalid')+'</span>')+'</span>';
    if(p.diff) h+='<details class="pdiff"><summary>view diff</summary><pre>'+escapeHtml(p.diff)+'</pre></details>';
    h+='</label>';
  }
  const anyValid=pts.some(p=>p.valid);
  h+='<div class="pactions">'
    +'<label class="toggle"><input type="checkbox" id="measurefit"> measure fitness impact (slow)</label>'
    +'<button class="cmd" onclick="applySelected()" '+(anyValid?'':'disabled')+'>OK Apply selected</button>'
    +'<button class="view" onclick="discardProposal()">Discard proposal</button>'
    +'<span class="pnote">safety tests still run on whatever you apply</span></div></div>';
  $('improveout').innerHTML=h;
}
function discardProposal(){ $('improveout').innerHTML=''; $('status').textContent='ready'; }
// Step 2: apply ONLY the changes you checked (safety tests still gate them).
async function applySelected(){
  const ids=[...document.querySelectorAll('.psel:checked')].map(c=>c.value);
  if(!ids.length){ $('status').textContent='select at least one change'; return; }
  const measure=!!($('measurefit')&&$('measurefit').checked);
  $('log').innerHTML=''; $('status').textContent='applying…';
  const es=$('evostatus'); es.classList.add('busy');
  es.innerHTML='<span class="dot spin"></span><b>Applying your selection…</b>';
  renderWhere();
  try{ await streamPost('/evolve/apply',{ids:ids,measure:measure}, ev=>{
    if(ev.stage==='done'){ const d=ev.data||{};
      if(d.busy){ $('improveout').innerHTML='<div class="result">'+escapeHtml(d.reason||'evolve already running')+'</div>';
        renderEvoStatus(); renderWhere(); return; }
      const cls=d.adopted?'ok':(d.rolled_back?'bad':'');
      let h='<div class="result '+cls+'"><b>'
        +(d.adopted?'OK Applied & kept':(d.rolled_back?'-> Reverted (safety)':'Not applied'))
        +'</b><br>'+escapeHtml(d.reason||'')+'<br>';
      if(d.delta!=null) h+='fitness '+d.incumbent+'→'+(d.candidate!=null?d.candidate:'?')
        +'/10 (Δ'+d.delta+', informational)<br>';
      if(d.changed&&d.changed.length) h+='files: '+escapeHtml(d.changed.join(', '))+'<br>';
      if(d.snapshot_id) h+='<span class="pnote">snapshot '+escapeHtml(d.snapshot_id)+' — revert with: ag rollback '+escapeHtml(d.snapshot_id)+'</span>';
      h+='</div>'; $('improveout').innerHTML=h;
      $('status').textContent='evolve: '+(d.adopted?'applied':(d.rolled_back?'reverted':'no change'));
      if(d.adopted) $('directive').value='';
      renderEvoStatus(); renderWhere();
      return; }
    if(ev.stage==='evolve'||ev.stage==='bench'){
      es.innerHTML='<span class="dot spin"></span><b>Applying your selection…</b> '+escapeHtml(ev.msg||''); }
    addEv(ev);
  }); }catch(e){ addEv({stage:'error',level:'error',msg:'apply failed: '+e});
    $('status').textContent='error'; renderEvoStatus(); renderWhere(); }
}
async function loadHistory(){
  const d=await (await fetch('/evolve/history')).json();
  if(!d.history||!d.history.length){
    $('improveout').innerHTML='<div class="result">No evolution history yet — click Evolve.</div>';
    return; }
  let h='<div class="result"><b>Fitness lineage</b> (newest first)<table class="hist">';
  for(const r of d.history){
    const mark=r.adopted?'OK adopted':'· '+(r.verdict||'n/a');
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
function saveThink(){ try{ localStorage.setItem('ag_think',$('think').value); }catch(e){} }
function restoreThink(){ try{ const v=localStorage.getItem('ag_think');
  if(v&&$('think')) $('think').value=v; }catch(e){} }
function saveModel(){ try{ localStorage.setItem('ag_model',$('model').value); }catch(e){} }
function saveMode(){ try{ localStorage.setItem('ag_mode',$('mode').value); }catch(e){} }
function restoreMode(){ try{ const v=localStorage.getItem('ag_mode');
  if(v&&$('mode')) $('mode').value=v; }catch(e){} }
async function loadModels(){
  const sel=$('model'); if(!sel) return;
  try{
    const d=await fetch('/models').then(r=>r.json());
    sel.innerHTML='';
    if(!d.options||!d.options.length){
      sel.innerHTML='<option value="">'+(d.ollama_reachable===false
        ?'(Ollama not reachable)':'(no models found)')+'</option>'; return; }
    for(const o of d.options){ const opt=document.createElement('option');
      opt.value=o.value; opt.textContent=o.label; sel.appendChild(opt); }
    let saved=null; try{ saved=localStorage.getItem('ag_model'); }catch(e){}
    const vals=d.options.map(o=>o.value);
    sel.value=(saved&&vals.includes(saved))?saved:(d.current||d.options[0].value);
  }catch(e){ sel.innerHTML='<option value="">(could not load models)</option>'; }
}
/* ---- Claude sign-in ---------------------------------------------------- */
async function loadAuth(){
  const el=$('signin'); if(!el) return;
  try{
    const a=await fetch('/auth').then(r=>r.json());
    el.hidden=false;
    if(a.signed_in){
      el.innerHTML='Signed in to Claude ('+escapeHtml(a.method)+') — auto uses <b>'
        +escapeHtml(a.model)+'</b>. <button class="linkbtn" onclick="logoutClaude()">sign out</button>';
    } else {
      el.innerHTML='Not signed in to Claude — running locally. '
        +'<a href="'+escapeHtml(a.console_url)+'" target="_blank" rel="noopener">Get an API key</a>, '
        +'paste it: <input id="apikey" type="password" placeholder="sk-ant-…" '
        +'style="width:210px;vertical-align:middle"> '
        +'<button class="cmd" onclick="saveKey()">Sign in</button>'
        +'<div class="ct-hint">On a Claude subscription? Install Claude Code and run its '
        +'login (or `ant auth login`) — AG reads that OAuth profile automatically.</div>';
    }
  }catch(e){}
}
async function saveKey(){
  const f=$('apikey'); const k=f?f.value.trim():''; if(!k) return;
  try{
    const r=await fetch('/login/key',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({key:k})}).then(r=>r.json());
    if(r.ok){ if(f) f.value=''; loadAuth(); loadModels(); renderWhere();
      $('status').textContent='signed in to Claude'; }
    else { alert('Sign in failed: '+(r.error||'unknown')); }
  }catch(e){ alert('Sign in failed: '+e); }
}
async function logoutClaude(){
  try{ await fetch('/logout',{method:'POST'}); }catch(e){}
  loadAuth(); loadModels(); renderWhere();
}
/* ---- local image generation ------------------------------------------ */
async function loadImageStatus(){
  const n=$('imgnote'); if(!n) return;
  try{
    const s=await fetch('/image/status').then(r=>r.json());
    if(!s.enabled){ n.textContent='· disabled in config'; }
    else if(s.reachable){ n.textContent='· ready at '+escapeHtml(s.host); }
    else { n.textContent='· no server at '+escapeHtml(s.host)+' (start Automatic1111/Forge with --api)'; }
  }catch(e){}
}
async function genImage(){
  const p=$('imgprompt')?$('imgprompt').value.trim():''; if(!p) return;
  const btn=$('imgbtn'), out=$('imgout');
  if(btn) btn.disabled=true;
  if(out) out.innerHTML='<div class="ct-hint">generating… (local diffusion can take a while)</div>';
  try{
    const r=await fetch('/image',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({prompt:p})}).then(r=>r.json());
    if(r.ok){ out.innerHTML='<img src="'+r.data_url+'" alt="'+escapeHtml(p)
      +'" style="max-width:100%;border-radius:10px;margin-top:8px">'
      +'<div class="ct-hint">saved to '+escapeHtml(r.path)+'</div>'; }
    else { out.innerHTML='<div class="result bad">'+escapeHtml(r.error||'failed')+'</div>'; }
  }catch(e){ out.innerHTML='<div class="result bad">request failed: '+escapeHtml(''+e)+'</div>'; }
  if(btn) btn.disabled=false;
}
// restore the transcript and status (where it runs + evolve) as soon as the page loads
restoreThink(); restoreMode(); loadModels(); loadChat(); renderWhere(); renderEvoStatus(); loadAuth(); loadImageStatus();
</script></body></html>"""

def _render_page() -> str:
    """Assemble the page from the evolvable theme (fonts + CSS) and the template."""
    from . import __version__
    from . import theme
    return (_PAGE_TEMPLATE
            .replace("__FONTS__", theme.FONT_LINK)
            .replace("__THEME__", theme.THEME_CSS)
            .replace("__VERSION__", __version__))

PAGE = _render_page()

def _clean_history(raw, *, max_turns: int = 40, max_len: int = 4000) -> list:
    """Sanitise conversation history from the client into [{role, text}] pairs.

    Untrusted input: coerce types, keep only known roles, bound count and size so a
    malformed or oversized payload can't wedge the pipeline. Content is treated as
    conversation data, never as instructions, by the prompt framing downstream.
    """
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw[-max_turns:]:
        if not isinstance(item, dict):
            continue
        role = "user" if str(item.get("role")) == "user" else "ai"
        text = str(item.get("text", "")).strip()
        if text:
            out.append({"role": role, "text": text[:max_len]})
    return out

def _build_broker(cfg: Config, *, web=None):
    web = cfg.allow_web if web is None else web
    if not (web or cfg.allow_local_tools):
        return None
    from .permissions import PermissionBroker
    broker = PermissionBroker(allow_external_tools=True)
    if web:
        broker.grant("network")
    if cfg.allow_local_tools:
        broker.grant("filesystem_read")   # read-only file/dir access
        broker.grant("spawn_agent")       # delegate a subtask to a sub-agent
        if getattr(cfg, "allow_code_exec", False):
            broker.grant("code_exec")     # arbitrary Python — opt-in only
    return broker

class _Handler(BaseHTTPRequestHandler):
    cfg: Config = Config()
    # Where the server is bound (filled in by serve()), for the GUI "where am I
    # running" indicator.
    bind_host: str = "127.0.0.1"
    bind_port: int = 8765
    # Evolve self-modifies source, so only one may run at a time. The server keeps
    # serving during an evolve (ThreadingHTTPServer), so this flag lets the GUI show
    # that a cycle is in progress and lets us reject overlapping evolves.
    _evolve_lock = threading.Lock()
    _evolving = threading.Event()
    # Cancellation registry: run_id -> True when the viewer asked to stop that run.
    # Checked every streamed chunk so Stop halts generation promptly and reliably,
    # rather than relying on a TCP write eventually failing.
    _cancel: dict = {}
    _cancel_lock = threading.Lock()
    # Last set of proposed self-edits, cached so the GUI can apply a chosen subset by
    # id (the browser never sends code back — AG applies exactly what it proposed).
    _proposal: dict = {}
    _proposal_directive: str = ""

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
        elif self.path == "/whereami":
            self._send(200, json.dumps(self._whereami()), "application/json")
        elif self.path == "/models":
            self._send(200, json.dumps(_models_data(self.cfg)), "application/json")
        elif self.path == "/auth":
            from .model import signin_status
            st = signin_status()
            st["model"] = self.cfg.model
            st["console_url"] = "https://console.anthropic.com/settings/keys"
            self._send(200, json.dumps(st), "application/json")
        elif self.path == "/image/status":
            from . import images
            cfg = self.cfg
            self._send(200, json.dumps({
                "enabled": bool(getattr(cfg, "allow_image_gen", False)),
                "reachable": images.sd_reachable(cfg) if getattr(
                    cfg, "allow_image_gen", False) else False,
                "host": cfg.sd_host}), "application/json")
        else:
            self._send(404, "not found", "text/plain")

    def _whereami(self) -> dict:
        """Where and how AG is running right now — for the GUI location indicator."""
        from .model import _has_anthropic_creds, has_oauth_profile
        cfg = self.cfg
        host, port = self.bind_host, self.bind_port
        local_only = host not in ("0.0.0.0", "::")
        if cfg.backend == "auto":
            if _has_anthropic_creds() or has_oauth_profile():
                brain = "Claude (" + cfg.model + ")"
            else:
                brain = ("Ollama (" + cfg.ollama_model + ")"
                         if getattr(cfg, "offline_backend", "") == "ollama"
                         else "dry-run (stub)")
        elif cfg.backend == "ollama":
            brain = "Ollama (" + cfg.ollama_model + ")"
        elif cfg.backend == "anthropic":
            brain = "Claude (" + cfg.model + ")"
        else:
            brain = "dry-run (stub)"
        try:
            hostname = socket.gethostname()
        except Exception:
            hostname = "?"
        return {
            "host": host, "port": port, "local_only": local_only,
            "hostname": hostname, "url": f"http://127.0.0.1:{port}",
            "lan_url": (f"http://{_lan_ip()}:{port}" if not local_only else None),
            "backend": cfg.backend, "brain": brain,
            "evolving": self._evolving.is_set(),
            "can_evolve_while_running": True,
        }

    def do_POST(self):
        if self.path == "/image":
            from . import images
            cfg = self.cfg
            try:
                n = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(n) or b"{}") if n else {}
                prompt = str(payload.get("prompt", "")).strip()
                if not prompt:
                    raise ValueError("empty prompt")
            except Exception as e:
                self._send(200, json.dumps({"ok": False, "error": str(e)}),
                           "application/json")
                return
            if not getattr(cfg, "allow_image_gen", False):
                self._send(200, json.dumps(
                    {"ok": False, "error": "image generation is disabled "
                     "(set allow_image_gen)"}), "application/json")
                return
            try:
                res = images.generate(prompt, cfg,
                                      negative_prompt=str(payload.get("negative", "")))
                self._send(200, json.dumps({
                    "ok": True, "data_url": res.data_url, "path": res.path,
                    "width": res.width, "height": res.height}), "application/json")
            except Exception as e:
                self._send(200, json.dumps({"ok": False, "error": str(e)}),
                           "application/json")
            return
        if self.path == "/stop":
            try:
                n = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(n) or b"{}") if n else {}
                rid = str(payload.get("run_id", ""))[:64]
            except Exception:
                rid = ""
            if rid:
                with _Handler._cancel_lock:
                    c = _Handler._cancel.get(rid)
                    if c is None:
                        _Handler._cancel[rid] = True   # sentinel: cancel on registration
                if hasattr(c, "cancel"):
                    c.cancel()                         # stop an already-running generation
            self._send(200, json.dumps({"ok": bool(rid)}), "application/json")
            return
        if self.path in ("/login/key", "/logout"):
            # Sign-in is a local action: only honour it from this machine, even when
            # bound to 0.0.0.0 for LAN viewing.
            if self.client_address and self.client_address[0] not in ("127.0.0.1", "::1"):
                self._send(403, json.dumps({"ok": False, "error": "sign-in is local-only"}),
                           "application/json")
                return
            from .model import save_api_key, clear_api_key, signin_status
            try:
                n = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(n) or b"{}") if n else {}
            except Exception:
                payload = {}
            if self.path == "/logout":
                clear_api_key()
                self._send(200, json.dumps({"ok": True, **signin_status()}),
                           "application/json")
                return
            key = str(payload.get("key", "")).strip()
            if len(key) < 10 or any(c.isspace() for c in key):
                self._send(200, json.dumps(
                    {"ok": False, "error": "that does not look like a valid key"}),
                    "application/json")
                return
            try:
                save_api_key(key)
                self._send(200, json.dumps({"ok": True, **signin_status()}),
                           "application/json")
            except Exception as e:
                self._send(200, json.dumps({"ok": False, "error": str(e)}),
                           "application/json")
            return
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
        if self.path in ("/evolve", "/evolve/propose", "/evolve/apply"):
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
            if self.path == "/evolve/apply":
                self._stream_apply(payload)
            elif self.path == "/evolve":
                self._stream_evolve(payload)   # legacy automatic path (not used by GUI)
            else:
                self._stream_propose(payload)
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
        history = _clean_history(payload.get("history"))
        think = str(payload.get("think", "auto")).lower()
        if think not in ("off", "auto", "on"):
            think = "auto"
        model = str(payload.get("model", "")).strip()[:100]
        mode = str(payload.get("mode", "")).lower().strip()
        verbose = bool(payload.get("verbose", False))
        run_id = str(payload.get("run_id", ""))[:64]
        self._stream_run(prompt, want_web, history, think, model, mode, verbose, run_id)

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
        """Run one gated self-improvement cycle, streaming its progress + verdict.

        Only one evolve runs at a time; the server keeps answering /run requests
        meanwhile (AG CAN evolve while running). Overlapping evolves are rejected.
        """
        from . import evolve as evolve_mod
        write = self._ndjson_writer()
        cfg = self.cfg
        # Reject a second concurrent evolve rather than corrupting a half-applied edit.
        if not self._evolve_lock.acquire(blocking=False):
            write({"stage": "done", "level": "info",
                   "msg": "an evolve cycle is already running — try again when it finishes",
                   "data": {"busy": True, "adopted": False, "rolled_back": False,
                            "reason": "evolve already in progress"}})
            return
        self._evolving.set()
        try:
            client = make_client(cfg)
            # Optional split backend: a stronger model proposes; deploy backend measures.
            evolver_client = None
            prop = str((payload or {}).get("proposer", "")).strip()
            directive = str((payload or {}).get("directive", "")).strip()
            if prop and prop != cfg.backend:
                try:
                    evolver_client = make_client(cfg, backend=prop)
                    write({"stage": "evolve", "level": "tool", "data": {},
                           "msg": f"proposer: {prop}  |  fitness measured on: {cfg.backend}"})
                except Exception as e:
                    write({"stage": "evolve", "level": "error", "data": {},
                           "msg": f"proposer '{prop}' unavailable ({e}); using deploy backend"})
            write({"stage": "evolve", "level": "tool",
                   "msg": ("starting self-improvement cycle"
                           + (" toward your request…" if directive else "…")),
                   "data": {}})
            res = evolve_mod.evolve(client, cfg, emit=write,
                                    evolver_client=evolver_client, directive=directive)
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
        finally:
            self._evolving.clear()
            self._evolve_lock.release()

    def _stream_propose(self, payload):
        """Generate candidate self-edits and stream them for the user to choose from.

        Nothing is applied here — the automatic adopt/reject decision is replaced by
        this propose→select→apply flow. The full patches are cached server-side so the
        browser only sends back the ids it selected.
        """
        from . import evolve as evolve_mod
        write = self._ndjson_writer()
        cfg = self.cfg
        if not self._evolve_lock.acquire(blocking=False):
            write({"stage": "done", "level": "info",
                   "msg": "an evolve cycle is already running — try again shortly",
                   "data": {"busy": True, "patches": []}})
            return
        try:
            client = make_client(cfg)
            evolver_client = None
            prop = str((payload or {}).get("proposer", "")).strip()
            directive = str((payload or {}).get("directive", "")).strip()
            if prop and prop != cfg.backend:
                try:
                    evolver_client = make_client(cfg, backend=prop)
                    write({"stage": "evolve", "level": "tool", "data": {},
                           "msg": f"proposer: {prop}"})
                except Exception as e:
                    write({"stage": "evolve", "level": "error", "data": {},
                           "msg": f"proposer '{prop}' unavailable ({e}); using deploy backend"})
            res = evolve_mod.propose(client, cfg, emit=write,
                                     evolver_client=evolver_client, directive=directive)
            _Handler._proposal = {p.id: p for p in res.patches}
            _Handler._proposal_directive = res.directive
            write({"stage": "done",
                   "level": "result" if res.attempted else "info",
                   "msg": res.reason, "data": res.as_dict()})
        except (BrokenPipeError, ConnectionError):
            return
        except Exception as e:
            try:
                write({"stage": "error", "level": "error", "msg": str(e), "data": {}})
            except Exception:
                pass
        finally:
            self._evolve_lock.release()

    def _stream_apply(self, payload):
        """Apply the user-selected subset of the last proposal (by id), streaming it.

        The human's selection IS the decision; only the safety test gate still applies.
        """
        from . import evolve as evolve_mod
        write = self._ndjson_writer()
        cfg = self.cfg
        ids = (payload or {}).get("ids") or []
        measure = bool((payload or {}).get("measure", False))
        cache = getattr(_Handler, "_proposal", {}) or {}
        selected = []
        for i in ids:
            p = cache.get(str(i))
            if p is not None and getattr(p, "valid", False):
                selected.append({"path": p.path, "new_content": p.new_content})
        if not selected:
            write({"stage": "done", "level": "info",
                   "msg": "nothing to apply — the proposal expired or held no valid "
                          "selection; click Evolve again",
                   "data": {"adopted": False, "rolled_back": False, "changed": [],
                            "reason": "no valid selection"}})
            return
        if not self._evolve_lock.acquire(blocking=False):
            write({"stage": "done", "level": "info",
                   "msg": "an evolve cycle is already running — try again shortly",
                   "data": {"busy": True, "adopted": False, "rolled_back": False}})
            return
        self._evolving.set()
        try:
            client = make_client(cfg)
            write({"stage": "evolve", "level": "tool", "data": {},
                   "msg": f"applying {len(selected)} selected change(s)"
                          + (" and measuring fitness…" if measure else "…")})
            res = evolve_mod.apply_selected(
                client, cfg, selected, emit=write, measure=measure,
                note=getattr(_Handler, "_proposal_directive", ""))
            write({"stage": "done",
                   "level": "result" if res.adopted else "info",
                   "msg": res.reason, "data": {
                       "adopted": res.adopted, "rolled_back": res.rolled_back,
                       "verdict": res.verdict, "incumbent": res.incumbent_fitness,
                       "candidate": res.candidate_fitness, "delta": res.fitness_delta,
                       "changed": res.changed, "rationale": res.rationale,
                       "reason": res.reason, "snapshot_id": res.snapshot_id}})
            if res.adopted:
                _Handler._proposal = {}   # consumed
        except (BrokenPipeError, ConnectionError):
            return
        except Exception as e:
            try:
                write({"stage": "error", "level": "error", "msg": str(e), "data": {}})
            except Exception:
                pass
        finally:
            self._evolving.clear()
            self._evolve_lock.release()

    def _parse_model_choice(self, choice: str):
        """Validate a "<backend>:<model>" selector value into per-run overrides.

        Returns a dict of Config overrides, or {} to use the configured default. We
        only honour a backend we actually support and, for the cloud option, only the
        configured Claude model — never an arbitrary string from the request. An
        Ollama model name is accepted as-is (the user may have just pulled it); if it
        isn't really available the run surfaces a clear error rather than pretending.
        """
        if not choice or ":" not in choice:
            return {}
        backend, _, model = choice.partition(":")
        backend, model = backend.lower().strip(), model.strip()
        if backend == "anthropic":
            return {"backend": "anthropic", "model": self.cfg.model}
        if backend == "ollama" and model:
            return {"backend": "ollama", "ollama_model": model}
        return {}

    def _stream_run(self, prompt: str, want_web, history=None, think="auto", model="",
                    mode="", verbose=False, run_id=""):
        """Run the pipeline, streaming each stage event as one NDJSON line.

        The model's output is always streamed internally via `on_delta`: when
        `verbose` is set the chunks are forwarded to the browser as 'delta' events so
        the viewer sees the reasoning/output live; otherwise a light heartbeat keeps
        the connection observable. Either way, if the viewer disconnects (Stop button /
        closed tab) the next write fails, we raise, and generation is halted upstream.
        """
        import dataclasses
        from .pipeline import capture_memory
        write = self._ndjson_writer()
        # Per-request overrides, without mutating the shared handler config.
        overrides = {"think": think}
        overrides.update(self._parse_model_choice(model))
        cfg = dataclasses.replace(self.cfg, **overrides)
        # Pipeline shape: explicit request mode wins, else the config default.
        fast = (mode == "fast") if mode in ("fast", "full") else None
        web_eff = cfg.allow_web if want_web is None else bool(want_web)
        broker = _build_broker(cfg, web=web_eff)

        # A Canceller lets /stop close the upstream model connection at any point (even
        # during prompt-eval), so Stop is prompt and reliable — not only once tokens flow.
        canceller = Canceller()
        if run_id:
            with _Handler._cancel_lock:
                pre = _Handler._cancel.get(run_id)
                _Handler._cancel[run_id] = canceller
            if pre is True:            # /stop arrived before we registered
                canceller.cancel()

        state = {"n": 0}

        def on_delta(text):
            state["n"] += 1
            try:
                if verbose:
                    write({"stage": "delta", "level": "token", "msg": "",
                           "data": {"text": text}})
                elif state["n"] % 16 == 0:
                    write({"stage": "progress", "level": "info", "msg": "",
                           "data": {"tokens": state["n"]}})
            except (BrokenPipeError, ConnectionError):
                raise _Interrupted()   # viewer closed the tab mid-stream

        try:
            client = make_client(cfg)
            rec = run_pipeline(client, cfg, prompt, web=web_eff, broker=broker,
                               emit=write, history=history, fast=fast,
                               on_delta=on_delta, cancel=canceller)
            write({"stage": "done", "level": "result", "msg": "done", "data": {
                "answer": rec.answer, "scorecard": rec.scorecard,
                "iterations": rec.iterations, "elapsed_s": rec.elapsed_s,
                "dry_run": rec.dry_run}})
            # Fill long-term memory AFTER the answer is on screen, so it never delays
            # the response. Best-effort; a "saved N fact(s)" event streams if it stores.
            try:
                from .pipeline import format_history
                convo = format_history(history,
                                       max_turns=getattr(cfg, "max_history_turns", 12))
                capture_memory(client, cfg, prompt, rec.answer,
                               conversation=convo, emit=write)
            except Exception:
                pass
        except (BrokenPipeError, ConnectionError, _Interrupted, Cancelled):
            return  # viewer stopped the run or navigated away mid-stream
        except Exception as e:
            try:
                write({"stage": "error", "level": "error", "msg": str(e), "data": {}})
            except Exception:
                pass
        finally:
            if run_id:
                with _Handler._cancel_lock:
                    _Handler._cancel.pop(run_id, None)

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
        if has_key or oauth:
            eff = "anthropic"
        elif getattr(cfg, "offline_backend", "") == "ollama":
            eff = "ollama"
        else:
            eff = "dry-run (stub)"
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

def _list_ollama_models(cfg: Config) -> list:
    """Names of models currently pulled in the local Ollama (empty if unreachable)."""
    import urllib.request
    try:
        with urllib.request.urlopen(cfg.ollama_host.rstrip("/") + "/api/tags",
                                    timeout=1.5) as r:
            return [m.get("name") for m
                    in json.loads(r.read().decode("utf-8")).get("models", [])
                    if m.get("name")]
    except Exception:
        return []

def _models_data(cfg: Config) -> dict:
    """Models the GUI selector can offer, and the current effective choice.

    Each option's value is "<backend>:<model>" so the /run handler knows both which
    backend to use and which model. Local (Ollama) models are always listed if the
    server is reachable; the cloud Claude model is offered only when creds/OAuth are
    present. `current` reflects what a run would use right now with no override."""
    from .model import _has_anthropic_creds, has_oauth_profile
    models = _list_ollama_models(cfg)
    cloud = bool(_has_anthropic_creds() or has_oauth_profile())
    options = []
    if cloud:
        options.append({"value": f"anthropic:{cfg.model}",
                        "label": f"{cfg.model} · Claude cloud"})
    options += [{"value": f"ollama:{m}", "label": f"{m} · local"} for m in models]

    if cfg.backend == "anthropic" or (cfg.backend == "auto" and cloud):
        current = f"anthropic:{cfg.model}"
    elif cfg.backend == "ollama" or (
            cfg.backend == "auto" and getattr(cfg, "offline_backend", "") == "ollama"):
        current = f"ollama:{cfg.ollama_model}"
    else:
        current = f"ollama:{cfg.ollama_model}"
    return {"options": options, "current": current, "cloud": cloud,
            "ollama_reachable": bool(models)}

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
    _Handler.bind_host = host
    _Handler.bind_port = port
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
