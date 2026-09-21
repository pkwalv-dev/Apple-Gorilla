"""The web interface as DATA: template, theme hooks, and the render function.

This module is deliberately separate from ag/server.py and listed in
config.evolvable_paths — the GUI is the one surface a human actually looks at,
so it should be improvable by the same keep-if-better loop as the prompts. Its
guard is NOT the fitness function (bench never renders a page); it is the test
suite's pins on the page's JS hooks — element ids, fetch endpoints, the SSE
reader — in tests/test_server.py. A candidate that breaks those pins fails
Gate 1 and rolls back, so the page can evolve in appearance but never in its
contract with the API below.
"""
from __future__ import annotations


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
tr.stalerow{opacity:.75}
.linkbtn.danger{color:#f85149}
#addhost{padding:5px 8px;border-radius:6px;border:1px solid rgba(127,127,127,.35);
  background:rgba(127,127,127,.06);color:inherit;font-size:13px}
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

<!-- ================= tab command bar ================= -->
<nav class="tabs" id="tabs">
  <button class="tab active" data-pane="chat" onclick="switchTab('chat')">&#9656; Chat</button>
  <button class="tab" data-pane="cluster" onclick="switchTab('cluster')">&#9741; Cluster <span class="tct" id="tc-cluster"></span></button>
  <button class="tab" data-pane="fleet" onclick="switchTab('fleet')">&#9670; Swarm <span class="tct" id="tc-fleet"></span></button>
  <button class="tab" data-pane="skills" onclick="switchTab('skills')">&#10022; Skills <span class="tct" id="tc-skills"></span></button>
  <button class="tab" data-pane="evolve" onclick="switchTab('evolve')">&#8635; Evolve</button>
  <button class="tab" data-pane="lora" onclick="switchTab('lora')">&#9881; LoRA</button>
  <button class="tab" data-pane="bundle" onclick="switchTab('bundle')">&#10697; Bundle</button>
  <button class="tab" data-pane="images" onclick="switchTab('images')">&#9638; Media</button>
</nav>

<!-- ================= CHAT ================= -->
<div class="tabpane active" id="pane-chat">

<!-- Row: the message column scrolls independently, the reasoning sidebar streams the
     model's thoughts and tool/trace in parallel, and the input below stays pinned. -->
<div id="chatmain">
<div id="chatscroll">
<!-- context bar: lights up to show which sources feed the CURRENT answer -->
<div id="ctxbar" title="What AG is drawing on for the current answer — each lights up as it is used">
  <span class="ctxlbl">context in use:</span>
  <span class="chip" id="chip-history">Conversation <b class="cc" id="cc-history"></b></span>
  <span class="chip" id="chip-profile">Profile</span>
  <span class="chip" id="chip-memory">Memory <b class="cc" id="cc-memory"></b></span>
  <span class="chip" id="chip-web">Web <b class="cc" id="cc-web"></b></span>
  <span class="chip" id="chip-skills" title="Lights up when AG uses tools or acquires a new skill this run">Skills <b class="cc" id="cc-skills"></b></span>
</div>
<div class="panel chatwrap">
  <div class="chat-toolbar">
    <span class="ct-hint">earlier exchanges are kept here — scroll to revisit</span>
    <button class="linkbtn spacer" onclick="clearChat()">Clear conversation</button>
  </div>
  <div id="chat"></div>
</div>

<div id="cards">
  <div class="card"><div class="n" id="spd">–</div><div class="l">speed (measured)</div>
    <div class="bar"><i id="spdb"></i></div></div>
</div>
</div><!-- /chatscroll -->

<aside id="reasonbar">
  <div class="rb-head">Reasoning &amp; trace</div>
  <pre id="reasonstream" title="the model's live reasoning (enable 'show reasoning')"></pre>
  <div id="log"></div>
</aside>
</div><!-- /chatmain -->

<div class="panel" id="inputpanel">
  <textarea id="p" placeholder="Ask Apple-Gorilla anything…  (Ctrl+Enter to send)"></textarea>
  <div class="controls">
    <button class="cmd primary" id="runbtn" onclick="go()">Run</button>
    <button class="cmd" id="stopbtn" onclick="stopRun()" hidden>Stop</button>
    <span class="cmd-note">Ctrl+Enter to send</span>
    <!-- Generated from ag/controls.py — one declaration drives the markup, the
         browser's save/restore/collect, and the server's config overrides. Add a
         capability there, not here. -->
    <span class="ctl-right">
__CONTROLS__
    </span>
    <!-- Extended controls, tucked to the side. Specific model selection here OVERRIDES
         the automatic primary/specialist routing for a run. -->
    <details class="advanced">
      <summary title="model override and other advanced controls">Advanced</summary>
      <div class="adv-body">
__MODEL_PICKER__
        <span class="adv-note">picking a model overrides automatic routing for this run</span>
      </div>
    </details>
  </div>
</div>
</div><!-- /pane-chat -->

<!-- ================= CLUSTER ================= -->
<div class="tabpane" id="pane-cluster">
  <div class="panel">
    <p class="lede"><b>Household cluster.</b> Your other devices, discovered on the LAN.
      Run <code>ag node</code> on a laptop or desktop and it appears here. A node is
      <b>seen</b> automatically, but AG will not <b>run</b> anything on it until you
      <b>Approve</b> it below — your approval is the credential. Approved, online nodes
      are used automatically for sub-agents, matched to their CPU/GPU/RAM.</p>
    <div class="fleet-bar">
      <button class="view" onclick="loadCluster()">Refresh</button>
      <label class="toggle" style="margin-left:6px">placement
        <select id="placement" onchange="setPlacement()">
          <option value="auto">auto (use other devices)</option>
          <option value="local">local only (this machine)</option>
        </select>
      </label>
      <span class="killbadge spacer" id="clbadge">nodes: —</span>
    </div>
    <table class="dtable" id="clustertbl" style="display:none">
      <thead><tr><th>device</th><th>address</th><th>CPU/RAM/GPU</th>
        <th class="n">score</th><th class="n">agents</th><th>status</th><th></th></tr></thead>
      <tbody id="clusterbody"></tbody>
    </table>
    <div id="clusterempty" class="emptyrow">No devices found yet. On another machine you
      own, run <code>python -m ag node</code> on the same Wi-Fi — it will show up here to
      approve.</div>
    <div class="fleet-bar" style="margin-top:10px">
      <input id="addhost" placeholder="host:port (e.g. 192.168.1.9:8767)"
        style="width:230px;vertical-align:middle">
      <button class="view" onclick="clusterAddStatic()">Add by address</button>
      <span class="ct-hint">for networks where broadcast is blocked</span>
    </div>
    <div class="fleet-bar" style="margin-top:14px;border-top:1px solid rgba(127,127,127,.15);padding-top:12px">
      <b>Mutual processing.</b>&nbsp;
      <label class="toggle">split
        <select id="benchiters" class="ctl-select">
          <option value="2000000">2M</option>
          <option value="5000000" selected>5M</option>
          <option value="20000000">20M</option>
          <option value="50000000">50M</option>
        </select>
      </label>
      <span class="ct-hint">iterations across every approved device, weighted by power</span>
      <button class="cmd" id="benchbtn" onclick="clusterBench()">Run distributed benchmark</button>
    </div>
    <div id="benchout"></div>
  </div>
</div>

<!-- ================= SWARM (fleet) ================= -->
<div class="tabpane" id="pane-fleet">
  <div class="panel">
    <p class="lede"><b>Agent swarm.</b> Every sub-agent AG spawns is registered here —
      its role, lineage, depth, <b>where it runs</b> (this machine or a cluster node) and
      live status. Kill a single agent, sweep <b>stale</b> ones, or hit the master
      <b>kill switch</b> to halt all spawning and stop running loops.</p>
    <div class="fleet-bar">
      <button class="view" onclick="loadFleet()">Refresh</button>
      <button class="view" onclick="fleetReap()">Reap stale</button>
      <button class="danger" onclick="fleetKill()">Engage kill switch</button>
      <button class="view" onclick="fleetRevive()">Clear kill switch</button>
      <span class="killbadge spacer" id="killbadge">kill: —</span>
    </div>
    <table class="dtable" id="fleettbl" style="display:none">
      <thead><tr><th>agent</th><th>role</th><th>location</th><th>parent</th>
        <th>depth</th><th>status</th><th class="n">skills</th><th></th></tr></thead>
      <tbody id="fleetbody"></tbody>
    </table>
    <div id="fleetempty" class="emptyrow">No sub-agents spawned yet. Delegation happens
      inside a run (the <code>delegate</code> tool) when tools are enabled.</div>
  </div>
</div>

<!-- ================= SKILLS ================= -->
<div class="tabpane" id="pane-skills">
  <div class="panel">
    <p class="lede"><b>Acquired skills.</b> Capabilities AG authored, tested, and
      registered — reused and inherited by sub-agents. Author one directly here, or turn
      on <b>self-extend</b> in Chat and just ask.</p>
    <textarea id="skspec" class="directive" rows="2"
      placeholder="Describe a capability to acquire, e.g. &quot;convert a CSV file to JSON&quot;. AG will author it, test it in isolation, and register it if the test passes."></textarea>
    <div class="controls" style="margin-top:0">
      <button class="cmd" id="skbtn" onclick="acquireSkill()">Acquire skill</button>
      <button class="view spacer" onclick="loadSkills()">Refresh list</button>
    </div>
    <div id="skacqout"></div>
    <div id="skillsout"></div>
  </div>
</div>

<!-- ================= EVOLVE ================= -->
<div class="tabpane" id="pane-evolve">
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
      <label class="ctl" style="margin-top:10px" title="Inject vetted reference code from your github_allowlist (config.evolve_github_refs) into the proposer's briefing so a self-edit can adapt proven implementations. Reference data only — evolve still edits only evolvable files and passes both gates."><input type="checkbox" id="evogithub"> use vetted GitHub references</label>
    </div>
    <div class="btngroup view-group">
      <span class="grouplabel">Views — read-only, change nothing</span>
      <div class="controls">
        <button class="view" onclick="loadTools()">Tools &amp; friction</button>
        <button class="view" onclick="loadHistory()">History</button>
        <button class="view" onclick="loadDoctor()">Status</button>
      </div>
    </div>
    <table id="tools" class="panel"></table>
    <div id="improveout"></div>
  </div>
</div>

<!-- ================= LORA ================= -->
<div class="tabpane" id="pane-lora">
  <div class="panel">
    <p class="lede"><b>LoRA fine-tuning.</b> Weight-level self-improvement of a
      <b>local</b> model (never Claude — no weight access): QLoRA a small base on a
      dataset distilled from AG's memory + a Claude teacher set. Needs an NVIDIA GPU and
      the training extras (<code>requirements-lora.txt</code>); heavy and explicit.</p>
    <div id="lorastatus" class="emptyrow">loading status…</div>
    <div class="controls" style="margin-top:6px">
      <label class="ctl">base model
        <select id="lorabase" class="ctl-select" onchange="loraSetBase()"><option>…</option></select>
      </label>
      <button class="help" data-help="The local model AG will fine-tune. Options are flagged for your GPU: 'fits' = comfortable, 'tight' = works with the memory-savers, 'won't fit' = inference-only. A LoRA adapter is bound to its base — it only works on this exact model.">?</button>
    </div>
    <div class="controls" style="margin-top:6px">
      <label class="ctl">epochs
        <input id="loraepochs" class="ctl-select" type="number" min="0.5" max="10" step="0.5"
               style="width:70px" onchange="loraSetOpts()">
      </label>
      <button class="help" data-help="How many passes over the dataset. More epochs learn the data harder (lower loss) but risk memorizing/overfitting on a small set. 1 is often too few; 3 is a solid default here.">?</button>
      <label class="toggle" style="margin-left:10px">
        <input type="checkbox" id="loraunsloth" onchange="loraSetOpts()"> use Unsloth
      </label>
      <button class="help" data-help="Unsloth trains ~2x faster in ~half the VRAM (needs a working C compiler). Off = the transformers+peft fallback, which is slower but needs no compiler. Turned off automatically if Unsloth isn't installed.">?</button>
    </div>
    <div class="controls" style="margin-top:6px">
      <button class="view" onclick="loadLora()">Refresh</button>
      <button class="cmd" onclick="loraBuild()">Build dataset</button>
      <button class="help" data-help="Assembles the training set from AG's own memory (high-scoring past runs + learned procedures) plus a Claude-generated teacher set. The teacher step calls the model, so it costs a few calls.">?</button>
      <button class="cmd" id="loratrain" onclick="loraTrain()">Train adapter</button>
      <button class="help" data-help="Runs one QLoRA fine-tune on the built dataset (needs a CUDA GPU + the training extras; uses Unsloth when installed). Heavy and long; runs in the background.">?</button>
    </div>
    <div id="loraout"></div>
  </div>
</div>

<!-- ================= BUNDLE ================= -->
<div class="tabpane" id="pane-bundle">
  <div class="panel">
    <p class="lede"><b>Portable bundle.</b> Pack AG and its learned state (profile,
      memory, skills, archive, config) into one archive for any Python 3.10+ host — the
      memory and skills travel with it. The check audits the portability constraints.</p>
    <div class="controls" style="margin-top:0">
      <button class="view" onclick="bundleCheck()">Check portability</button>
      <button class="cmd" onclick="bundleExport()">Export bundle</button>
    </div>
    <div id="bundleout"></div>
  </div>
</div>

<!-- ================= IMAGES ================= -->
<div class="tabpane" id="pane-images">
  <div class="panel" id="imagepanel">
    <span class="grouplabel">Media generation — on this machine's GPU <span class="cmd-note" id="imgnote"></span></span>
    <textarea id="imgprompt" class="directive" rows="2"
      placeholder="Describe an image or a clip to generate. Runs on your own GPU (ComfyUI, or an Automatic1111 server for images); the prompt never leaves your machine."></textarea>
    <div class="controls">
      <label class="ctl">make <select id="imgkind" class="ctl-select" onchange="syncMediaKind()">
        <option value="image">an image</option><option value="video">a video clip</option>
      </select></label>
      <label class="ctl" id="imgsecs-wrap" hidden>seconds <select id="imgsecs" class="ctl-select">
        <option value="2">2</option><option value="3" selected>3</option>
        <option value="5">5</option></select></label>
      <button class="cmd" id="imgbtn" onclick="genMedia()">Generate</button>
      <button class="cmd" id="imgstartbtn" onclick="startImageServer()" hidden>Start server</button>
    </div>
    <div id="imgout"></div>
  </div>
</div>

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
// Scroll the message region to the newest content. Only auto-follows when the reader
// is already near the bottom (or force=true, e.g. right after they send), so scrolling
// up to re-read an earlier turn is never yanked back down mid-stream.
function scrollChat(force){
  const s=$('chatscroll'); if(!s) return;
  const nearBottom = s.scrollHeight - s.scrollTop - s.clientHeight < 120;
  if(force||nearBottom) s.scrollTop=s.scrollHeight;
}
function ctxFooter(c){
  if(!c) return '';
  const bits=[];
  if(c.model) bits.push(escapeHtml(c.model));
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
// Per-answer actions: rate (teaches the routing doc), regenerate, and copy a distilled
// prompt for Claude. prompt+model travel on the row so the handlers can attribute.
function aiActions(prompt,model){
  const dp=escapeHtml(prompt||''), dm=escapeHtml(model||'');
  return '<div class="airow" data-prompt="'+dp+'" data-model="'+dm+'">'
    +'<button class="ico" title="good answer" onclick="rate(this,&#39;rating_up&#39;)">&#128077;</button>'
    +'<button class="ico" title="poor answer" onclick="rate(this,&#39;rating_down&#39;)">&#128078;</button>'
    +'<button class="ico" title="regenerate this answer" onclick="regen(this)">&#8635;</button>'
    +'<button class="ico wide" title="copy a distilled prompt to paste into Claude" onclick="toClaude(this)">Claude &#10697;</button>'
    +'</div>';
}
function bubble(m,prev){
  const who=m.role==='user'?'you':'apple-gorilla';
  const t=m.ts?'<span class="ts">'+escapeHtml(m.ts)+'</span>':'';
  return '<div class="msg '+m.role+'"><div class="who">'+who+t+'</div>'
    +'<div class="body">'+escapeHtml(m.text)+'</div>'
    +(m.role==='ai'?ctxFooter(m.ctx)+aiActions(prev,(m.ctx&&m.ctx.model)||''):'')+'</div>';
}
function renderChat(){
  $('chat').innerHTML=CHAT.map((m,i)=>bubble(m, i>0?CHAT[i-1].text:'')).join('');
  scrollChat(true);
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
    +'<div class="live-ctx"></div>';
  $('chat').appendChild(el); scrollChat(true);
  return el;
}

/* ---- live activity indicator: what the model is doing + how long ------- */
/* Makes a snag visible: the stage label shows the current step and the timer
   keeps ticking, so a stall (timer climbing, stage unchanged) is obvious. */
const STAGE_LABELS={conversation:'reading the conversation',
  web:'searching the web',memory:'recalling memory',
  execute:'generating the answer',reason:'using tools',
  acquire:'acquiring a skill',score:'scoring speed'};
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

/* ---- per-tab working-memory session id ------------------------------- */
const AG_SESSION=(function(){
  try{let s=sessionStorage.getItem('ag_session');
    if(!s){s='web-'+Date.now().toString(36)+'-'+Math.random().toString(36).slice(2,8);
      sessionStorage.setItem('ag_session',s);}
    return s;
  }catch(e){return 'web-'+Date.now().toString(36);}
})();

/* ---- context-in-use indicator ---------------------------------------- */
let curCtx={history:0,profile:false,memory:[],saved:[],web:0,skills:0};
function resetContext(){
  curCtx={history:0,profile:false,memory:[],saved:[],web:0,skills:0};
  ['history','profile','memory','web','skills'].forEach(n=>$('chip-'+n).classList.remove('active'));
  $('cc-history').textContent=''; $('cc-memory').textContent=''; $('cc-web').textContent='';
  $('cc-skills').textContent='';
}
function applyContext(ev,ai){
  if(ev.stage==='conversation'){
    const m=/(\\d+)/.exec(ev.msg||''); const n=m?+m[1]:0;
    curCtx.history=n||curCtx.history; $('chip-history').classList.add('active');
    $('cc-history').textContent=(ev.data&&ev.data.turns)||n||'';
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
  // 'reason' = a tool was used; 'acquire' = a skill was authored/registered this run.
  if(ev.stage==='reason' || ev.stage==='acquire'){
    curCtx.skills++; $('chip-skills').classList.add('active');
    $('cc-skills').textContent=curCtx.skills;
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
  if($('reasonstream')) $('reasonstream').textContent='';
  ['spd'].forEach(x=>setCard(x,null));
  // prior turns become AG's working memory (the current prompt is sent separately)
  const hist=CHAT.slice(-20).map(m=>({role:m.role,text:m.text}));
  resetContext(); appendUser(p); const ai=appendAssistant(); startActivity(ai); $('p').value='';
  const ctrl=new AbortController(); CURRENT_ABORT=ctrl; RUN_STOPPED=false;
  const runId=(self.crypto&&crypto.randomUUID)?crypto.randomUUID():String(Date.now())+Math.random();
  CURRENT_RUNID=runId;
  const verbose=$('verbose')?$('verbose').checked:false;
  try{
    // Every declared control goes along automatically — see AG_CONTROLS.
    const r=await fetch('/run',{method:'POST',headers:{'Content-Type':'application/json'},
      signal:ctrl.signal,
      body:JSON.stringify(Object.assign(ctlPayload(),
        {prompt:p,history:hist,model:($('model')?$('model').value:''),
         run_id:runId,session_id:AG_SESSION}))});
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
  if(ev.stage==='delta'){   // live streamed model reasoning -> the sidebar, in parallel
    const rs=$('reasonstream');
    if(rs){ rs.textContent+=((ev.data&&ev.data.text)||''); rs.scrollTop=rs.scrollHeight; }
    return;
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
               web:curCtx.web,saved:curCtx.saved,model:d.model_used||'',
               overall:(sc.overall!=null?sc.overall:null)};
    const prevPrompt=CHAT.length?CHAT[CHAT.length-1].text:'';
    ai.insertAdjacentHTML('beforeend',ctxFooter(ctx)+aiActions(prevPrompt,ctx.model));
    const entry={role:'ai',text:d.answer||'(no answer)',ctx:ctx,ts:nowStr()};
    CHAT.push(entry); ai._entry=entry; ai._committed=true; saveChat();
    scrollChat();
    $('status').textContent='done · '+(d.elapsed_s||0)+'s'+(d.dry_run?' · dry-run':'');
    checkUpdate();   // one on-demand check AFTER the run — never a background poll
    return;
  }
  setStage(ai,ev);
  applyContext(ev,ai);
  addEv(ev);
  const sc=(ev.data&&ev.data.scorecard);
  if(sc && sc.speed!=null){ $('cards').style.display='grid'; setCard('spd',sc.speed); }
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
  try{ await streamPost('/evolve/propose',{proposer:$('proposer').value,directive:directive,github:$('evogithub')?$('evogithub').checked:false}, ev=>{
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
/* ---- run controls ------------------------------------------------------
   Everything below is generic: it reads the declaration shipped from
   ag/controls.py, so a new control needs no JavaScript at all. */
const AG_CONTROLS=__CONTROLS_JSON__;
function ctlEl(c){ return $(c.id); }
function ctlGet(c){ const el=ctlEl(c); if(!el) return null;
  return c.kind==='toggle' ? el.checked : el.value; }
function ctlSet(c,v){ const el=ctlEl(c); if(!el) return;
  if(c.kind==='toggle') el.checked=!!v; else el.value=v; }
/* Live = this control can actually affect the run. A control whose prerequisite is
   off is disabled, cleared and dimmed, so the bar never offers a switch that does
   nothing. The server enforces the same rule; the page is not the authority. */
function ctlLive(c){
  if(!c.requires) return true;
  const dep=AG_CONTROLS.find(x=>x.id===c.requires); if(!dep) return true;
  const el=ctlEl(dep); if(!el) return true;
  return ctlLive(dep) && (dep.kind==='toggle' ? el.checked : !!el.value);
}
function syncCtls(){
  for(const c of AG_CONTROLS){
    const el=ctlEl(c); if(!el) continue;
    const live=ctlLive(c);
    el.disabled=!live;
    if(!live && c.kind==='toggle') el.checked=false;
    const wrap=$(c.id+'-wrap'); if(wrap) wrap.style.opacity=live?'1':'.45';
  }
  saveCtls();
}
function saveCtls(){ try{ for(const c of AG_CONTROLS){ const v=ctlGet(c);
  if(v===null) continue;
  localStorage.setItem('ag_ctl_'+c.id, c.kind==='toggle'?(v?'1':'0'):String(v));
}}catch(e){} }
function restoreCtls(){ try{ for(const c of AG_CONTROLS){
  if(!ctlEl(c)) continue;
  const raw=localStorage.getItem('ag_ctl_'+c.id);
  if(raw==null) ctlSet(c,c.default);
  else ctlSet(c, c.kind==='toggle' ? raw==='1' : raw);
}}catch(e){} syncCtls(); }
function ctlPayload(){ const out={};
  for(const c of AG_CONTROLS){ const v=ctlGet(c); if(v!==null) out[c.id]=v; }
  return out; }
function saveModel(){ try{ localStorage.setItem('ag_model',$('model').value); }catch(e){} }
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
      el.innerHTML='Signed in to Claude ('+escapeHtml(a.method)+') — available as <b>'
        +escapeHtml(a.model)+'</b>, used only when you pick it in the model menu (spends API tokens). '
        +'AG answers locally by default. <button class="linkbtn" onclick="logoutClaude()">sign out</button>';
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
  loadAuth(); loadModels(); renderWhere(); syncMediaKind();
}
/* ---- local image generation ------------------------------------------ */
async function loadImageStatus(){
  const n=$('imgnote'), sb=$('imgstartbtn'); if(!n) return;
  try{
    const s=await fetch('/image/status').then(r=>r.json());
    const k=$('imgkind'); if(k && !s.video){ k.value='image'; k.disabled=true; syncMediaKind(); }
    if(!s.enabled){ n.textContent='· disabled in config'; if(sb) sb.hidden=true; }
    else if(s.reachable){ n.textContent='· ready · '+escapeHtml(s.backend)+' at '+escapeHtml(s.host); if(sb) sb.hidden=true; }
    else { n.textContent='· no '+escapeHtml(s.backend)+' server at '+escapeHtml(s.host)
        +' — click Start server'
        +(s.models&&s.models.length?' (needs: '+escapeHtml(s.models.join(', '))+')':'');
      if(sb) sb.hidden=false; }
  }catch(e){}
}
async function startImageServer(){
  const n=$('imgnote'), sb=$('imgstartbtn');
  if(sb) sb.disabled=true; if(n) n.textContent='· starting image server (first start loads a model — up to a few minutes)…';
  try{
    const r=await fetch('/image/start',{method:'POST'}).then(r=>r.json());
    if(!r.ok && n) n.textContent='· '+escapeHtml(r.error||'could not start the server');
  }catch(e){ if(n) n.textContent='· start failed: '+escapeHtml(''+e); }
  if(sb) sb.disabled=false;
  loadImageStatus();
}
function mediaKind(){ return $('imgkind') ? $('imgkind').value : 'image'; }
function syncMediaKind(){
  const w=$('imgsecs-wrap'); if(w) w.hidden = mediaKind()!=='video';
}
async function genMedia(){
  const p=$('imgprompt')?$('imgprompt').value.trim():''; if(!p) return;
  const kind=mediaKind(), btn=$('imgbtn'), out=$('imgout');
  if(btn) btn.disabled=true;
  if(out) out.innerHTML='<div class="ct-hint">generating'
    +(kind==='video'?' a clip — this takes minutes, not seconds':'')
    +'… (if the server is not running AG will start it first — the first run loads a multi-GB model)</div>';
  try{
    const body={prompt:p};
    if(kind==='video' && $('imgsecs')) body.seconds=parseFloat($('imgsecs').value);
    const r=await fetch(kind==='video'?'/video':'/image',
      {method:'POST',headers:{'Content-Type':'application/json'},
       body:JSON.stringify(body)}).then(r=>r.json());
    if(r.ok && kind==='video'){
      out.innerHTML='<video controls style="max-width:100%;border-radius:10px;margin-top:8px" '
        +'src="/media?path='+encodeURIComponent(r.path)+'"></video>'
        +'<div class="ct-hint">'+r.seconds+'s · saved to '+escapeHtml(r.path)+'</div>'; }
    else if(r.ok){ out.innerHTML='<img src="'+r.data_url+'" alt="'+escapeHtml(p)
      +'" style="max-width:100%;border-radius:10px;margin-top:8px">'
      +'<div class="ct-hint">saved to '+escapeHtml(r.path)+'</div>'; }
    else { out.innerHTML='<div class="result bad">'+escapeHtml(r.error||'failed')+'</div>'; }
  }catch(e){ out.innerHTML='<div class="result bad">request failed: '+escapeHtml(''+e)+'</div>'; }
  if(btn) btn.disabled=false;
}
// restore the transcript and status (where it runs + evolve) as soon as the page loads
/* ---- help popovers (click to open; works on touch) ------------------- */
let HELP_POP=null;
function closeHelp(){ if(HELP_POP){ HELP_POP.remove(); HELP_POP=null; } }
document.addEventListener('click', function(e){
  const btn=e.target.closest && e.target.closest('.help');
  if(btn){
    e.preventDefault(); e.stopPropagation();
    const txt=btn.getAttribute('data-help')||'';
    if(HELP_POP && HELP_POP._for===btn){ closeHelp(); return; }
    closeHelp();
    const p=document.createElement('div'); p.className='help-pop'; p._for=btn;
    p.innerHTML='<span class="hx" onclick="closeHelp()">\\u2715</span>'+escapeHtml(txt);
    document.body.appendChild(p);
    const r=btn.getBoundingClientRect();
    let left=Math.min(r.left, window.innerWidth-p.offsetWidth-12);
    let top=r.bottom+6;
    if(top+p.offsetHeight>window.innerHeight-8) top=Math.max(8,r.top-p.offsetHeight-6);
    p.style.left=Math.max(8,left)+'px'; p.style.top=top+'px';
    HELP_POP=p; return;
  }
  if(HELP_POP && !e.target.closest('.help-pop')) closeHelp();
});
window.addEventListener('resize', closeHelp);

/* ---- tabs ------------------------------------------------------------- */
function switchTab(name){
  document.querySelectorAll('.tab').forEach(t=>t.classList.toggle('active',t.dataset.pane===name));
  document.querySelectorAll('.tabpane').forEach(p=>p.classList.toggle('active',p.id==='pane-'+name));
  try{ localStorage.setItem('ag_tab',name); }catch(e){}
  if(name==='cluster') loadCluster();
  if(name==='fleet') loadFleet();
  if(name==='skills') loadSkills();
  if(name==='lora') loadLora();
}
/* ---- lora ------------------------------------------------------------- */
let LORA_POLL=null;
let LORA_MERGE_READY=false;
async function loadLora(){
  try{
    const d=await (await fetch('/lora/status')).json();
    const f=d.feasibility||{};
    const tr=d.training||{};
    LORA_MERGE_READY=!!d.merge_ready;
    // base selector, annotated with fit for this GPU
    const sel=$('lorabase');
    if(sel){ sel.innerHTML=(d.bases||[]).map(b=>'<option value="'+escapeHtml(b.id)+'"'
      +(b.id===d.base?' selected':'')+'>'+escapeHtml(b.id)+' · '+b.params_b+'B · '+b.fit
      +'</option>').join(''); }
    const ep=$('loraepochs'); if(ep && d.epochs!=null) ep.value=d.epochs;
    const us=$('loraunsloth');
    if(us){ us.checked=!!d.unsloth_pref; us.disabled=!(f.unsloth); }
    $('lorastatus').className='kv';
    $('lorastatus').innerHTML=
      'GPU <b>'+escapeHtml(f.gpu||'?')+'</b> · VRAM <b>'+(f.vram_gb==null?'?':f.vram_gb+' GB')
      +'</b> · CUDA <b>'+(f.cuda?'yes':'no')+'</b> · Unsloth <b>'+(f.unsloth?'yes':'no')
      +'</b><br>dataset <b>'+(d.dataset_size||0)+'</b> example(s) · adapters <b>'
      +(d.adapters||[]).length+'</b> · merge→Ollama <b>'+(d.merge_ready?'ready':'not set up')+'</b>'
      +'<br>trainable now: <b>'+(f.ok?'YES':'no')+'</b>'
      +(f.missing_deps&&f.missing_deps.length?' · missing: '+escapeHtml(f.missing_deps.join(', ')):'')
      +((f.notes||[]).map(n=>'<br>&nbsp;• '+escapeHtml(n)).join(''))
      +(tr.running?'<br><b>LoRA job in progress…</b>':(tr.last?'<br>last: '+escapeHtml(tr.last):''));
    $('loratrain').disabled=!!tr.running;
    const ad=(d.adapters||[]);
    $('loraout').innerHTML = ad.length
      ? ad.map(a=>'<div class="skitem"><span class="sknm">'+escapeHtml(a.id)+'</span>'
          +'<span class="skds">base '+escapeHtml(a.base||'?')+' · '+(a.examples||'?')+' examples'
          +(a.unsloth?' · unsloth':'')+'</span>'
          +(LORA_MERGE_READY?'<button class="linkbtn" onclick="loraMerge(\\''+escapeHtml(a.id)
             +'\\')">merge → Ollama</button>':'')+'</div>').join('')
      : '<div class="emptyrow">No adapters yet. Build a dataset, then train.</div>';
    if(tr.running && !LORA_POLL){ LORA_POLL=setInterval(loadLora,4000); }
    if(!tr.running && LORA_POLL){ clearInterval(LORA_POLL); LORA_POLL=null; }
  }catch(e){ $('lorastatus').textContent='could not load LoRA status: '+e; }
}
async function loraSetBase(){
  const base=$('lorabase').value;
  try{ await fetch('/lora/set-base',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({base:base})}); }catch(e){}
  loadLora();
}
async function loraSetOpts(){
  const epochs=parseFloat($('loraepochs').value);
  const unsloth=$('loraunsloth').checked;
  try{ await fetch('/lora/set-opts',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({epochs:epochs,unsloth:unsloth})}); }catch(e){}
  loadLora();
}
async function loraMerge(adapter){
  if(!confirm('Merge "'+adapter+'" into its base, convert to GGUF, and register it with Ollama? This is heavy and long.')) return;
  try{
    const d=await (await fetch('/lora/merge',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({adapter:adapter})})).json();
    if(!d.started){ $('loraout').innerHTML='<div class="result bad">'+escapeHtml(d.reason||'could not start')+'</div>'; }
    else{ $('loraout').innerHTML='<div class="result">merge started — runs in the background; status updates above.</div>'; }
    loadLora();
  }catch(e){ $('loraout').innerHTML='<div class="result bad">merge failed to start: '+escapeHtml(''+e)+'</div>'; }
}
async function loraBuild(){
  $('loraout').innerHTML='<div class="result">building dataset (memory + teacher — the teacher set calls the model)…</div>';
  try{
    const d=await (await fetch('/lora/build',{method:'POST'})).json();
    $('loraout').innerHTML='<div class="result ok">dataset: '+(d.total||0)+' example(s) ('
      +(d.from_memory||0)+' from memory, '+(d.from_teacher||0)+' from teacher)</div>';
    loadLora();
  }catch(e){ $('loraout').innerHTML='<div class="result bad">build failed: '+escapeHtml(''+e)+'</div>'; }
}
async function loraTrain(){
  if(!confirm('Start a QLoRA training run? This is heavy and can take a long time.')) return;
  try{
    const d=await (await fetch('/lora/train',{method:'POST'})).json();
    if(!d.started){ $('loraout').innerHTML='<div class="result bad">'+escapeHtml(d.reason||'could not start')+'</div>'; }
    else{ $('loraout').innerHTML='<div class="result">training started — this runs in the background; status updates above.</div>'; }
    loadLora();
  }catch(e){ $('loraout').innerHTML='<div class="result bad">train failed to start: '+escapeHtml(''+e)+'</div>'; }
}
function restoreTab(){ let t='chat'; try{ t=localStorage.getItem('ag_tab')||'chat'; }catch(e){}
  if(!document.getElementById('pane-'+t)) t='chat'; switchTab(t); }

/* ---- fleet ------------------------------------------------------------ */
async function loadFleet(){
  try{
    const d=await (await fetch('/fleet')).json();
    const kb=$('killbadge');
    kb.textContent='kill: '+(d.kill_active?'ENGAGED':'off');
    kb.classList.toggle('on',!!d.kill_active);
    const agents=d.agents||[];
    $('tc-fleet').textContent=agents.length?('['+agents.length+']'):'';
    const tbl=$('fleettbl'), body=$('fleetbody'), empty=$('fleetempty');
    if(!agents.length){ tbl.style.display='none'; empty.style.display='block'; return; }
    empty.style.display='none'; tbl.style.display='table';
    body.innerHTML=agents.map(a=>{
      const loc=a.node&&a.node!=='local'?(a.location||a.node):'this machine';
      const st=a.stale?'stale':escapeHtml(a.status);
      const alive=a.status==='active'||a.status==='disabled';
      const A=escapeHtml(a.agent);
      let act='';
      if(a.status==='disabled') act='<button class="linkbtn" onclick="fleetAct(\\''+A+'\\',\\'enable\\')">enable</button> ';
      else if(alive) act='<button class="linkbtn" onclick="fleetAct(\\''+A+'\\',\\'disable\\')">disable</button> ';
      if(alive) act+='<button class="linkbtn danger" onclick="fleetKillAgent(\\''+A+'\\')">kill</button>';
      return '<tr'+(a.stale?' class="stalerow"':'')+'><td>'+A+'</td><td>'+escapeHtml(a.role)
        +'</td><td>'+escapeHtml(loc)+'</td><td>'+escapeHtml(a.parent)+'</td><td class="n">'+a.depth
        +'</td><td><span class="pill '+(a.stale?'degraded':escapeHtml(a.status))+'">'+st+'</span></td>'
        +'<td class="n">'+a.skills_acquired+'</td><td>'+act+'</td></tr>';
    }).join('');
  }catch(e){ $('fleetempty').textContent='could not load fleet: '+e; }
}
async function fleetAct(agent,action){
  try{ await fetch('/fleet/act',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({action:action,agent:agent})}); }catch(e){}
  loadFleet();
}
async function fleetKillAgent(agent){
  if(!confirm('Kill agent "'+agent+'" (and any sub-agents it spawned)? It stops at its next step.')) return;
  try{ await fetch('/fleet/act',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({action:'kill_agent',agent:agent})}); }catch(e){}
  loadFleet();
}
async function fleetReap(){
  try{ await fetch('/fleet/act',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({action:'reap'})}); }catch(e){}
  loadFleet();
}
async function fleetKill(){
  if(!confirm('Engage the kill switch? This halts ALL sub-agent spawning and stops running loops.')) return;
  try{ await fetch('/fleet/act',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({action:'kill'})}); }catch(e){}
  loadFleet();
}
async function fleetRevive(){
  try{ await fetch('/fleet/act',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({action:'revive'})}); }catch(e){}
  loadFleet();
}

/* ---- cluster ---------------------------------------------------------- */
async function loadCluster(){
  try{
    const d=await (await fetch('/cluster')).json();
    const badge=$('clbadge');
    if(!d.cluster_on){ badge.textContent='clustering off';
      $('clustertbl').style.display='none'; $('clusterempty').style.display='block';
      $('clusterempty').textContent='Clustering is disabled (config net_cluster=false).'; return; }
    if($('placement')) $('placement').value=d.placement||'auto';
    const nodes=d.nodes||[];
    const online=nodes.filter(n=>n.online).length;
    badge.textContent='nodes: '+online+' online / '+nodes.length+' known';
    $('tc-cluster').textContent=nodes.length?('['+nodes.length+']'):'';
    const tbl=$('clustertbl'), body=$('clusterbody'), empty=$('clusterempty');
    if(!nodes.length){ tbl.style.display='none'; empty.style.display='block'; return; }
    empty.style.display='none'; tbl.style.display='table';
    body.innerHTML=nodes.map(n=>{
      const c=n.caps||{};
      const priv=c.elevated?' <span class="tagpill lan" title="this node runs elevated — its agents have admin/root on that device">admin</span>':'';
      const mips=c.benchMips?(' · '+c.benchMips+' MIPS'):'';
      const specs=(c.cpu_count||'?')+' cpu · '+(c.ram_gb||'?')+' GB · '+escapeHtml(c.gpu||'?')+mips+priv;
      const me=n.self_node?' <span class="tagpill local">this device</span>':'';
      const N=escapeHtml(n.node_id);
      let act='';
      if(n.self_node){ act='—'; }
      else if(!n.approved){ act='<button class="linkbtn" onclick="clusterAct(\\''+N+'\\',\\'approve\\')">approve</button> '
        +'<button class="linkbtn danger" onclick="clusterAct(\\''+N+'\\',\\'forget\\')">forget</button>'; }
      else { act='<button class="linkbtn" onclick="clusterAct(\\''+N+'\\',\\'revoke\\')">revoke</button> '
        +'<button class="linkbtn danger" onclick="clusterAct(\\''+N+'\\',\\'forget\\')">forget</button>'; }
      return '<tr><td>'+escapeHtml(n.name||'?')+me+'</td><td>'+escapeHtml(n.host||'?')+':'+n.port
        +'</td><td>'+specs+'</td><td class="n">'+n.score+'</td><td class="n">'+(n.agents_here||0)
        +'</td><td><span class="pill '+(n.status==='online'?'available':(n.status==='pending'?'degraded':'unavailable'))
        +'">'+escapeHtml(n.status)+'</span></td><td>'+act+'</td></tr>';
    }).join('');
  }catch(e){ $('clusterempty').textContent='could not load cluster: '+e; }
}
async function clusterAct(node_id,action){
  if(action==='approve' && !confirm('Approve this device to run AG sub-agents? It will get this machine\\'s trust and full local tools for tasks you send it.')) return;
  try{ await fetch('/cluster/act',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({action:action,node_id:node_id})}); }catch(e){}
  loadCluster();
}
async function clusterAddStatic(){
  const v=($('addhost')?$('addhost').value.trim():''); if(!v||v.indexOf(':')<0) return;
  const host=v.substring(0,v.lastIndexOf(':')), port=parseInt(v.substring(v.lastIndexOf(':')+1),10);
  try{ await fetch('/cluster/act',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({action:'add',host:host,port:port})}); }catch(e){}
  if($('addhost')) $('addhost').value=''; loadCluster();
}
async function setPlacement(){
  const v=$('placement')?$('placement').value:'auto';
  try{ await fetch('/cluster/act',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({action:'set_placement',placement:v})}); }catch(e){}
}
async function clusterBench(){
  const btn=$('benchbtn'), out=$('benchout');
  const iters=parseInt($('benchiters')?$('benchiters').value:'5000000',10);
  if(btn){ btn.disabled=true; }
  out.innerHTML='<div class="ct-hint">splitting '+(iters/1e6).toFixed(0)
    +'M iterations across your devices and running them in parallel…</div>';
  try{
    const r=await fetch('/cluster/bench',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({iters:iters})}).then(r=>r.json());
    if(!r.ok){ out.innerHTML='<div class="result bad">'+escapeHtml(r.reason||'benchmark failed')+'</div>'; }
    else{
      let h='<div class="result '+(r.verified?'ok':'')+'"><b>'+r.aggregate_mips
        +' MIPS</b> aggregate across '+r.shards_done+' device(s) · '+(r.iterations/1e6).toFixed(1)
        +'M iterations · '+(r.verified?'verified':'UNVERIFIED')
        +(r.shards_failed?' · '+r.shards_failed+' failed':'')+'<table class="hist">';
      for(const p of (r.per_node||[])){
        h+='<tr><td>'+escapeHtml(p.name)+'</td><td>'+p.mips+' MIPS</td><td>'
          +(p.iterations/1e6).toFixed(1)+'M</td><td>'+(p.ok?(p.verified?'ok ✓':'ok'):'failed')+'</td></tr>';
      }
      h+='</table></div>'; out.innerHTML=h;
    }
  }catch(e){ out.innerHTML='<div class="result bad">request failed: '+escapeHtml(''+e)+'</div>'; }
  if(btn){ btn.disabled=false; }
  loadCluster();
}

/* ---- skills ----------------------------------------------------------- */
async function loadSkills(){
  try{
    const d=await (await fetch('/skills')).json();
    const sk=d.skills||[];
    $('tc-skills').textContent=sk.length?('['+sk.length+']'):'';
    const out=$('skillsout');
    if(!sk.length){ out.innerHTML='<div class="emptyrow">No skills acquired yet. '
      +'Author one above, or enable self-extend in Chat and ask for a capability.</div>'; return; }
    out.innerHTML=sk.map(s=>'<div class="skitem"><span class="sknm">'+escapeHtml(s.name)+'</span>'
      +'<span class="skds">'+escapeHtml(s.description)
      +(s.capabilities&&s.capabilities.length?' <span class="skcap">['+escapeHtml(s.capabilities.join(', '))+']</span>':'')
      +'</span><span class="pill '+(s.enabled?'ok':'disabled')+'">'+(s.enabled?'enabled':'disabled')+'</span>'
      +'<button class="linkbtn" onclick="skillAct(\\''+escapeHtml(s.name)+'\\',\\''+(s.enabled?'disable':'enable')+'\\')">'
      +(s.enabled?'disable':'enable')+'</button>'
      +'<button class="linkbtn" onclick="skillAct(\\''+escapeHtml(s.name)+'\\',\\'remove\\')">remove</button></div>').join('');
  }catch(e){ $('skillsout').innerHTML='<div class="emptyrow">could not load skills: '+escapeHtml(''+e)+'</div>'; }
}
async function skillAct(name,action){
  if(action==='remove' && !confirm('Remove skill "'+name+'"? This deletes it from the registry.')) return;
  try{ await fetch('/skills/act',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({action:action,name:name})}); }catch(e){}
  loadSkills();
}
async function acquireSkill(){
  const spec=$('skspec').value.trim(); if(!spec) return;
  const btn=$('skbtn'); btn.disabled=true;
  $('skacqout').innerHTML='<div class="result">authoring &amp; testing a skill for: '
    +escapeHtml(spec)+' … (this calls the model and may take a moment)</div>';
  try{
    const d=await (await fetch('/skills/acquire',{method:'POST',
      headers:{'Content-Type':'application/json'},body:JSON.stringify({spec:spec})})).json();
    $('skacqout').innerHTML='<div class="result '+(d.acquired?'ok':'bad')+'">'
      +escapeHtml(d.reason||(d.acquired?'acquired':'not acquired'))+'</div>';
    if(d.acquired){ $('skspec').value=''; loadSkills(); }
  }catch(e){ $('skacqout').innerHTML='<div class="result bad">acquire failed: '+escapeHtml(''+e)+'</div>'; }
  btn.disabled=false;
}

/* ---- bundle ----------------------------------------------------------- */
async function bundleCheck(){
  $('bundleout').innerHTML='<div class="result">running portability audit…</div>';
  try{
    const d=await (await fetch('/bundle/check')).json();
    const rows=(d.checks||[]).map(c=>'<div class="bcheck '+(c.ok?'ok':'bad')+'"><span class="led"></span>'
      +'<div><div>'+escapeHtml(c.name)+'</div>'+(c.detail?'<div class="bd">'+escapeHtml(c.detail)+'</div>':'')+'</div></div>').join('');
    $('bundleout').innerHTML=rows+'<div class="result '+(d.portable?'ok':'bad')+'">'
      +(d.portable?'Portable — all constraints satisfied.':'NOT fully portable — see failures above.')+'</div>';
  }catch(e){ $('bundleout').innerHTML='<div class="result bad">check failed: '+escapeHtml(''+e)+'</div>'; }
}
async function bundleExport(){
  $('bundleout').innerHTML='<div class="result">packing bundle…</div>';
  try{
    const d=await (await fetch('/bundle/export',{method:'POST'})).json();
    $('bundleout').innerHTML='<div class="result ok">Bundle written on the AG host:<br><code>'
      +escapeHtml(d.path||'')+'</code>'+(d.bytes?(' &middot; '+Math.round(d.bytes/1024)+' KB'):'')+'</div>';
  }catch(e){ $('bundleout').innerHTML='<div class="result bad">export failed: '+escapeHtml(''+e)+'</div>'; }
}
function saveEvoGithub(){ try{ localStorage.setItem('ag_evogithub',$('evogithub').checked?'1':'0'); }catch(e){} }
function restoreEvoGithub(){ try{ const v=localStorage.getItem('ag_evogithub');
  if($('evogithub')) $('evogithub').checked=(v==='1'); }catch(e){}
  if($('evogithub')) $('evogithub').addEventListener('change',saveEvoGithub); }

// Per-answer actions. Rating and escalation feed AG's model-routing doc; regenerate
// re-asks. prompt+model ride on the .airow so each handler can attribute correctly.
function _rowData(btn){ const r=btn.closest('.airow');
  return r?{row:r,prompt:r.getAttribute('data-prompt')||'',model:r.getAttribute('data-model')||''}:null; }
function _sendRate(prompt,model,signal){
  fetch('/rate',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({prompt:prompt,model:model,signal:signal})}).catch(()=>{}); }
function rate(btn,signal){ const d=_rowData(btn); if(!d) return;
  _sendRate(d.prompt,d.model,signal);
  // 👍 and 👎 are the first two icons; make them mutually exclusive and mark the choice.
  const icos=d.row.querySelectorAll('.ico');
  if(icos[0]) icos[0].classList.remove('on');
  if(icos[1]) icos[1].classList.remove('on');
  btn.classList.add('on'); }
function regen(btn){ const d=_rowData(btn); if(!d||!d.prompt) return;
  _sendRate(d.prompt,d.model,'rating_down');   // a regenerate = this wasn't good enough
  $('p').value=d.prompt; go(); }
function toClaude(btn){ const d=_rowData(btn); if(!d) return;
  const msg=btn.closest('.msg'); const bodyEl=msg?msg.querySelector('.body'):null;
  const answer=bodyEl?bodyEl.textContent:'';
  const distilled='I asked my local AI assistant a question and want your help solving it well.\\n\\n'
    +'=== PROBLEM ===\\n'+d.prompt+'\\n\\n'
    +'=== LOCAL ASSISTANT ('+(d.model||'local')+') ANSWERED ===\\n'+answer+'\\n\\n'
    +'=== WHAT I NEED ===\\nGive a correct, complete solution. If the local answer is wrong '
    +'or incomplete, fix it and explain what it missed.';
  const done=()=>{ const o=btn.innerHTML; btn.textContent='copied \\u2713';
    setTimeout(()=>{btn.innerHTML=o;},1500); };
  if(navigator.clipboard&&navigator.clipboard.writeText){
    navigator.clipboard.writeText(distilled).then(done).catch(()=>{ btn.textContent='copy failed'; });
  } else { try{ const t=document.createElement('textarea'); t.value=distilled;
    document.body.appendChild(t); t.select(); document.execCommand('copy'); t.remove(); done();
  }catch(e){ btn.textContent='copy failed'; } }
  _sendRate(d.prompt,d.model,'escalated');   // escalating = the local model fell short
}
restoreCtls(); restoreEvoGithub(); restoreTab(); loadModels(); loadChat(); renderWhere(); renderEvoStatus(); loadAuth(); loadImageStatus();
// Ctrl/Cmd+Enter sends from the chat box (a plain Enter still inserts a newline, so
// multi-line prompts are easy to write).
(function(){ const p=$('p'); if(!p) return;
  p.addEventListener('keydown', function(e){
    if((e.ctrlKey||e.metaKey) && e.key==='Enter'){ e.preventDefault();
      if(!$('runbtn').disabled) go(); } }); })();
</script></body></html>"""

_MODEL_PICKER = """      <label class="ctl" title="Which model answers this run. Local models run offline via Ollama; the Claude cloud option appears when you're signed in. Bigger local models are smarter but slower — watch the activity timer on the reply.">model
        <select id="model" class="ctl-select" onchange="saveModel()">
          <option value="">loading…</option>
        </select>
      </label>"""


def render_page() -> str:
    """Assemble the page from the evolvable theme (fonts + CSS), the declared run
    controls, and the template."""
    from . import __version__
    from . import controls, theme
    return (_PAGE_TEMPLATE
            .replace("__FONTS__", theme.FONT_LINK)
            .replace("__THEME__", theme.THEME_CSS)
            .replace("__CONTROLS__", controls.render_html())
            .replace("__MODEL_PICKER__", _MODEL_PICKER)
            .replace("__CONTROLS_JSON__", controls.spec_json())
            .replace("__VERSION__", __version__))
