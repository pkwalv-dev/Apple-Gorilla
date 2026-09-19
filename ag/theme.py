"""Presentation layer for the web app — fonts, palette, and layout.

This file is *evolvable* (listed in config.evolvable_paths), so AG can iterate its
own visual design within the test gate without touching server logic. Keep it to
CSS + a fonts <link>; the server injects THEME_CSS into the page. A bad edit that
breaks the page's required hooks (checked by tests/test_server.py) is rolled back.

Design language: a 2026 "technical control-room" — dark-first, data-forward, with a
monospace accent for labels/values/telemetry, hairline grids, LED-style status dots,
and a tabbed command surface for driving the agent, its swarm, its skills, and its
self-improvement. Inter for prose, JetBrains Mono for the instrument panels; robust
system fallbacks keep it sharp offline.
"""
from __future__ import annotations

FONT_LINK = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?'
    'family=Inter:wght@400;500;600;700;800&'
    'family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">'
)

THEME_CSS = """
:root{
  color-scheme:dark;
  --font:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
  --mono:'JetBrains Mono',ui-monospace,'SF Mono',Menlo,Consolas,'Liberation Mono',monospace;
  /* control-room dark */
  --bg:#070a10; --surface:#0d131c; --surface-2:#0a0f17; --elev:#111925;
  --text:#dbe4f0; --muted:#8b98ab; --faint:#5a6678; --border:#1b2534; --grid:#141c28;
  --accent:#22d3ee; --accent-2:#38bdf8; --accent-ink:#04121a;
  --live:#34d399; --tool:#38bdf8; --web:#22d3ee; --error:#fb7185; --success:#34d399;
  --warn:#fbbf24; --danger:#f43f5e;
  --shadow:0 1px 0 rgba(255,255,255,.02) inset,0 12px 40px rgba(0,0,0,.5);
  --radius:12px; --radius-sm:9px;
}
@media (prefers-color-scheme:light){:root{
  color-scheme:light;
  --bg:#eef1f7; --surface:#ffffff; --surface-2:#f5f8fc; --elev:#ffffff;
  --text:#0b1220; --muted:#55606f; --faint:#8894a3; --border:#dde3ed; --grid:#eaeff6;
  --accent:#0e7490; --accent-2:#0891b2; --accent-ink:#ffffff;
  --live:#059669; --tool:#2563eb; --web:#0891b2; --error:#dc2626; --success:#059669;
  --warn:#b45309; --danger:#e11d48;
  --shadow:0 1px 2px rgba(16,24,40,.06),0 10px 30px rgba(16,24,40,.08);
}}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{
  font-family:var(--font); font-size:15px; line-height:1.55; color:var(--text);
  background:
    linear-gradient(var(--grid) 1px,transparent 1px) 0 0/100% 34px,
    radial-gradient(900px 500px at 100% -8%, color-mix(in srgb,var(--accent) 10%,transparent), transparent 60%),
    var(--bg);
  margin:0; padding:22px 18px 60px; min-height:100vh;
  -webkit-font-smoothing:antialiased; text-rendering:optimizeLegibility;
}
.wrap{max-width:1040px;margin:0 auto}
header{display:flex;align-items:center;gap:14px;margin-bottom:16px}
.logo{font-family:var(--mono);font-weight:700;font-size:20px;line-height:1;color:var(--accent-ink);
  background:linear-gradient(180deg,var(--accent-2),var(--accent));padding:8px 10px;border-radius:10px;
  box-shadow:0 6px 18px color-mix(in srgb,var(--accent) 40%,transparent)}
.brand h1{font-size:1.35rem;font-weight:800;letter-spacing:-.02em;margin:0}
.brand .sub{font-family:var(--mono);font-size:.72rem;color:var(--muted);font-weight:500;
  margin-top:2px;letter-spacing:.02em}
#status{margin-left:auto;font-family:var(--mono);font-size:.72rem;font-weight:600;color:var(--muted);
  padding:6px 12px;border:1px solid var(--border);border-radius:999px;background:var(--surface);
  white-space:nowrap;text-transform:uppercase;letter-spacing:.06em}
.panel{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);
  box-shadow:var(--shadow);padding:16px;margin-bottom:16px}
textarea{width:100%;min-height:88px;font-family:var(--font);font-size:1rem;line-height:1.5;
  padding:13px 15px;border-radius:var(--radius-sm);border:1px solid var(--border);
  background:var(--surface-2);color:var(--text);resize:vertical;transition:border-color .15s,box-shadow .15s}
textarea::placeholder{color:var(--faint)}
textarea:focus{outline:none;border-color:var(--accent);
  box-shadow:0 0 0 4px color-mix(in srgb,var(--accent) 20%,transparent)}
.controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-top:12px}
button{font-family:var(--font);font-size:.92rem;font-weight:600;padding:10px 16px;cursor:pointer;
  border-radius:var(--radius-sm);border:1px solid var(--border);background:var(--surface-2);
  color:var(--text);transition:transform .05s,background .15s,border-color .15s,box-shadow .15s}
button:hover{border-color:var(--accent-2)}
button:active{transform:translateY(1px)}
button:disabled{opacity:.5;cursor:default}
button.primary{background:linear-gradient(180deg,var(--accent-2),var(--accent));
  color:var(--accent-ink);border-color:transparent;
  box-shadow:0 1px 0 rgba(255,255,255,.18) inset,0 6px 18px color-mix(in srgb,var(--accent) 40%,transparent)}
button.primary:hover{filter:brightness(1.06)}
button.danger{background:linear-gradient(180deg,#fb7185,var(--danger));color:#fff;border-color:transparent}
.cmd-note{font-family:var(--mono);font-size:.7rem;color:var(--faint);font-style:normal}
.toggle{display:inline-flex;gap:8px;align-items:center;font-size:.86rem;font-weight:500;
  color:var(--muted);margin-left:auto;user-select:none}
.toggle input{width:15px;height:15px;accent-color:var(--accent)}

/* --- run controls ---------------------------------------------------- */
.ctl-right{margin-left:auto;display:flex;gap:14px;align-items:center;flex-wrap:wrap;justify-content:flex-end}
/* An explicit display: wins over the hidden attribute, so say so once, globally —
   otherwise every hideable control needs its own inline style to disappear. */
[hidden]{display:none!important}
.ctl{display:inline-flex;gap:7px;align-items:center;font-family:var(--mono);font-size:.76rem;
  font-weight:500;color:var(--muted);user-select:none;letter-spacing:.01em}
.ctl input{width:15px;height:15px;accent-color:var(--accent)}
.ctl input:disabled{opacity:.4}
.ctl-select{font-family:var(--mono);font-size:.76rem;font-weight:600;color:var(--text);
  background:var(--surface-2);border:1px solid var(--border);border-radius:7px;
  padding:5px 8px;cursor:pointer}
.ctl-select:focus{outline:none;border-color:var(--accent);
  box-shadow:0 0 0 3px color-mix(in srgb,var(--accent) 18%,transparent)}
h2{display:flex;align-items:center;gap:8px;font-family:var(--mono);font-size:.7rem;font-weight:700;
  text-transform:uppercase;letter-spacing:.12em;color:var(--faint);margin:0 0 10px 2px}

/* --- tab command bar ------------------------------------------------- */
.tabs{display:flex;gap:4px;flex-wrap:wrap;margin:0 0 16px;padding:5px;border-radius:var(--radius);
  background:var(--surface-2);border:1px solid var(--border)}
.tab{font-family:var(--mono);font-size:.76rem;font-weight:600;letter-spacing:.04em;
  text-transform:uppercase;color:var(--muted);background:transparent;border:1px solid transparent;
  padding:8px 14px;border-radius:8px;cursor:pointer;display:inline-flex;gap:7px;align-items:center}
.tab:hover{color:var(--text);border-color:var(--border)}
.tab.active{color:var(--accent-ink);background:linear-gradient(180deg,var(--accent-2),var(--accent));
  border-color:transparent;box-shadow:0 4px 14px color-mix(in srgb,var(--accent) 34%,transparent)}
.tab .tct{font-size:.68rem;opacity:.8;font-variant-numeric:tabular-nums}
.tabpane{display:none;animation:fadein .18s ease}
.tabpane.active{display:block}
@keyframes fadein{from{opacity:0;transform:translateY(3px)}to{opacity:1;transform:none}}

/* --- live trace / terminal ------------------------------------------- */
#log{font-family:var(--mono);font-size:.78rem;line-height:1.5;max-height:340px;overflow:auto;
  border-radius:var(--radius-sm);background:#05080d;border:1px solid var(--border);padding:8px}
#log:empty::after{content:'> awaiting run — thoughts, tool & internet calls, errors stream here';
  color:var(--faint);font-family:var(--mono);font-size:.78rem;display:block;padding:12px}
.ev{display:flex;gap:10px;padding:4px 10px;border-left:2px solid var(--faint);
  border-radius:0 5px 5px 0;margin:2px 0;white-space:pre-wrap;word-break:break-word}
.ev:nth-child(even){background:rgba(127,127,127,.05)}
.ev .tag{flex:none;min-width:74px;color:var(--muted);font-weight:600;text-transform:uppercase;
  font-size:.66rem;letter-spacing:.05em;padding-top:2px}
.ev.tool{border-color:var(--tool)} .ev.web{border-color:var(--web)}
.ev.error{border-color:var(--error);color:var(--error)}
.ev.result{border-color:var(--success)}
.ev.token{border-color:var(--faint);color:var(--muted)}

/* --- scorecards ------------------------------------------------------ */
#cards{display:none;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:16px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-sm);
  padding:13px 15px;box-shadow:var(--shadow);position:relative;overflow:hidden}
.card .n{font-family:var(--mono);font-size:1.7rem;font-weight:700;letter-spacing:-.02em;
  font-variant-numeric:tabular-nums}
.card .l{font-family:var(--mono);font-size:.66rem;font-weight:600;text-transform:uppercase;
  letter-spacing:.08em;color:var(--faint);margin-top:2px}
.bar{height:6px;border-radius:999px;background:rgba(127,127,127,.14);margin-top:9px;overflow:hidden}
.bar>i{display:block;height:100%;border-radius:999px;
  background:linear-gradient(90deg,var(--accent-2),var(--accent));transition:width .5s cubic-bezier(.2,.8,.2,1)}
#answer{font-size:1.02rem;line-height:1.65;white-space:pre-wrap;word-break:break-word;min-height:44px;color:var(--text)}
#answer:empty::after{content:'—';color:var(--faint)}

/* --- data tables (tools / fleet) ------------------------------------- */
#tools,.dtable{display:none;width:100%;border-collapse:collapse;font-family:var(--mono);font-size:.8rem}
.dtable{display:table}
#tools th,.dtable th{text-align:left;font-size:.66rem;text-transform:uppercase;letter-spacing:.06em;
  color:var(--faint);padding:7px 10px;border-bottom:1px solid var(--border);font-weight:700}
#tools td,.dtable td{padding:8px 10px;border-bottom:1px solid var(--grid)}
#tools tr:last-child td{border-bottom:none;color:var(--muted)}
#tools td.n,.dtable td.n{text-align:right;font-variant-numeric:tabular-nums;font-weight:600}
.pill{display:inline-block;font-family:var(--mono);font-size:.66rem;font-weight:600;padding:2px 9px;
  border-radius:999px;border:1px solid var(--border);text-transform:uppercase;letter-spacing:.04em}
.pill.available,.pill.active,.pill.ok,.pill.done{color:var(--success);
  border-color:color-mix(in srgb,var(--success) 45%,transparent)}
.pill.degraded,.pill.disabled{color:var(--warn);border-color:color-mix(in srgb,var(--warn) 45%,transparent)}
.pill.unavailable{color:var(--faint)}

/* --- update banner --------------------------------------------------- */
#update{display:none;align-items:center;gap:12px;margin-bottom:16px;padding:13px 15px;
  border:1px solid color-mix(in srgb,var(--warn) 55%,var(--border));border-radius:var(--radius-sm);
  background:color-mix(in srgb,var(--warn) 12%,var(--surface))}
#update .g{flex:1;font-size:.9rem;font-weight:500}
#update .yes{background:var(--warn);color:#04121a;border-color:transparent}

.ver{font-family:var(--mono);font-size:.58rem;font-weight:700;letter-spacing:.03em;
  vertical-align:middle;margin-left:9px;padding:2px 8px;border-radius:999px;
  color:var(--accent);background:color-mix(in srgb,var(--accent) 16%,transparent)}

/* --- status strips (whereami / evostatus) ---------------------------- */
.whereami,.evostatus{display:flex;flex-wrap:wrap;align-items:center;gap:6px 8px;
  margin:0 0 12px;padding:9px 14px;border-radius:var(--radius-sm);
  background:var(--surface);border:1px solid var(--border);box-shadow:var(--shadow);
  font-family:var(--mono);font-size:.75rem;color:var(--muted);line-height:1.5}
.whereami b,.evostatus b{color:var(--text);font-weight:600}
.evostatus i{color:var(--faint);font-style:normal}
.evostatus .g-ok{color:var(--success);font-weight:600}
.evostatus .g-off{color:var(--warn);font-weight:600}
.dot{width:8px;height:8px;border-radius:999px;background:var(--faint);flex:none;margin-right:4px;
  box-shadow:0 0 0 0 transparent}
.dot.ok{background:var(--live);box-shadow:0 0 8px color-mix(in srgb,var(--live) 70%,transparent)}
.dot.off{background:var(--warn)}
.evostatus.busy{border-color:var(--accent);background:color-mix(in srgb,var(--accent) 8%,var(--surface))}
.evostatus .dot.spin{background:var(--accent);animation:agspin 1s linear infinite;
  box-shadow:0 0 0 3px color-mix(in srgb,var(--accent) 22%,transparent)}
@keyframes agspin{to{transform:rotate(360deg)}}
.tagpill{font-family:var(--mono);font-size:.64rem;font-weight:700;padding:2px 8px;border-radius:999px;
  border:1px solid var(--border);text-transform:uppercase;letter-spacing:.04em}
.tagpill.local{color:var(--success);border-color:color-mix(in srgb,var(--success) 45%,transparent)}
.tagpill.lan{color:var(--warn);border-color:color-mix(in srgb,var(--warn) 45%,transparent)}
.tagpill.idle{color:var(--faint)}
.tagpill.evolving{color:var(--accent);font-weight:700;
  border-color:color-mix(in srgb,var(--accent) 55%,transparent);
  background:color-mix(in srgb,var(--accent) 12%,var(--surface))}

/* --- command / view button groups ------------------------------------ */
.btngroup{border-radius:var(--radius-sm);padding:12px 14px}
.grouplabel{display:block;font-family:var(--mono);font-size:.64rem;font-weight:700;text-transform:uppercase;
  letter-spacing:.1em;color:var(--faint);margin:0 0 8px 2px}
.cmd-group{background:color-mix(in srgb,var(--accent) 6%,transparent);
  border:1px solid color-mix(in srgb,var(--accent) 22%,var(--border))}
.view-group{margin-top:12px;background:var(--surface-2);border:1px dashed var(--border)}
.cmd-group .grouplabel{color:var(--accent)}
button.cmd{background:linear-gradient(180deg,var(--accent-2),var(--accent));
  color:var(--accent-ink);border-color:transparent;
  box-shadow:0 1px 0 rgba(255,255,255,.15) inset,0 6px 18px color-mix(in srgb,var(--accent) 32%,transparent)}
button.cmd:hover{filter:brightness(1.07);border-color:transparent}
button.cmd.evolve{background:linear-gradient(180deg,#c084fc,#a855f7)}
button.view{background:transparent;color:var(--muted);border:1px dashed var(--border);box-shadow:none}
button.view:hover{color:var(--text);border-style:solid;border-color:var(--accent-2)}

/* --- context-in-use chips ------------------------------------------- */
#ctxbar{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin:0 2px 12px}
.ctxlbl{font-family:var(--mono);font-size:.66rem;font-weight:600;text-transform:uppercase;
  letter-spacing:.08em;color:var(--faint);margin-right:2px}
.chip{display:inline-flex;align-items:center;gap:6px;font-family:var(--mono);font-size:.7rem;
  font-weight:600;padding:5px 11px;border-radius:999px;border:1px solid var(--border);color:var(--faint);
  background:var(--surface-2);opacity:.5;transition:opacity .2s,color .2s,border-color .2s,box-shadow .2s}
.chip .cc{font-variant-numeric:tabular-nums}
.chip.active{opacity:1;color:var(--accent);
  border-color:color-mix(in srgb,var(--accent) 55%,transparent);
  background:color-mix(in srgb,var(--accent) 12%,var(--surface));
  box-shadow:0 0 0 3px color-mix(in srgb,var(--accent) 14%,transparent)}

/* --- conversation ---------------------------------------------------- */
.chatwrap{padding:12px}
.chat-toolbar{display:flex;align-items:center;gap:10px;margin-bottom:8px}
.ct-hint{font-family:var(--mono);font-size:.7rem;color:var(--faint)}
.spacer{margin-left:auto}
.linkbtn{background:none;border:none;color:var(--muted);font-size:.76rem;font-weight:600;
  padding:4px 6px;cursor:pointer;border-radius:6px}
.linkbtn:hover{color:var(--accent);background:color-mix(in srgb,var(--accent) 10%,transparent)}
#chat{display:flex;flex-direction:column;gap:14px;max-height:520px;overflow:auto;padding:6px}
#chat:empty::after{content:'Conversation appears here — send a prompt above to begin.';
  color:var(--faint);font-size:.88rem;display:block;padding:22px;text-align:center}
.msg{display:flex;flex-direction:column;gap:4px;max-width:88%}
.msg.user{align-self:flex-end;align-items:flex-end}
.msg.ai{align-self:flex-start;align-items:flex-start}
.msg .who{font-family:var(--mono);font-size:.62rem;font-weight:700;text-transform:uppercase;
  letter-spacing:.06em;color:var(--faint);display:flex;gap:8px;align-items:center}
.msg .who .ts{font-weight:500;text-transform:none;letter-spacing:0;opacity:.8}
.msg .body{padding:11px 14px;border-radius:13px;line-height:1.6;white-space:pre-wrap;
  word-break:break-word;font-size:.96rem}
.msg.user .body{background:linear-gradient(180deg,var(--accent-2),var(--accent));
  color:var(--accent-ink);border-bottom-right-radius:4px}
.msg.ai .body{background:var(--surface-2);border:1px solid var(--border);border-bottom-left-radius:4px}
.msg.ai .body.pending{color:var(--muted);font-style:italic;opacity:.85}
.msg.ai.processing{outline:2px solid color-mix(in srgb,var(--accent) 45%,transparent);
  outline-offset:4px;border-radius:13px}
.msg .ctx{display:flex;flex-wrap:wrap;gap:6px;margin-top:2px}
.msg .ctx span{font-family:var(--mono);font-size:.66rem;color:var(--muted);padding:1px 8px;
  border-radius:999px;border:1px solid var(--border);background:var(--surface)}
.msg .live-ctx{font-size:.72rem;color:var(--muted);margin-top:2px}
.msg .live-ctx .using{font-weight:600;color:var(--accent)}
.msg .live-ctx ul,.memfacts ul{margin:4px 0 0;padding-left:18px}
.msg .live-ctx li,.memfacts li{margin:1px 0}
.memfacts{margin-top:4px;font-size:.72rem;color:var(--muted)}
.memfacts summary{cursor:pointer;color:var(--accent);font-weight:600}
.act-timer{opacity:.6;font-variant-numeric:tabular-nums;font-style:normal}
.verbose{margin-top:8px;font-size:12px}
.verbose summary{cursor:pointer;color:var(--muted);font-family:var(--mono);font-size:.7rem}
.vbody{white-space:pre-wrap;max-height:260px;overflow:auto;margin-top:6px;padding:8px;
  border-radius:8px;background:#05080d;font-family:var(--mono);font-size:11px;line-height:1.45}

/* --- evolve directive + proposal ------------------------------------- */
.directive{width:100%;min-height:52px;font-family:var(--font);font-size:.9rem;line-height:1.5;
  padding:10px 12px;border-radius:var(--radius-sm);
  border:1px dashed color-mix(in srgb,var(--accent) 40%,var(--border));
  background:var(--surface-2);color:var(--text);resize:vertical;margin-bottom:10px;
  transition:border-color .15s,box-shadow .15s}
.directive::placeholder{color:var(--faint)}
.directive:focus{outline:none;border-style:solid;border-color:var(--accent);
  box-shadow:0 0 0 4px color-mix(in srgb,var(--accent) 18%,transparent)}
.proposal{border:1px solid var(--border);border-radius:var(--radius-sm);background:var(--surface);padding:14px}
.phead{display:flex;flex-wrap:wrap;align-items:baseline;gap:8px 12px;margin-bottom:8px}
.phead b{font-size:.94rem}
.pnote{font-size:.7rem;color:var(--faint);font-style:italic}
.prationale{font-size:.82rem;color:var(--muted);line-height:1.5;margin-bottom:10px;
  padding:8px 10px;border-radius:8px;background:var(--surface-2)}
.pitem{display:flex;flex-wrap:wrap;align-items:center;gap:8px;padding:9px 10px;
  border:1px solid var(--border);border-radius:8px;margin:6px 0;cursor:pointer}
.pitem:hover{border-color:var(--accent-2)}
.pitem.invalid{opacity:.6;cursor:not-allowed}
.pitem .psel{width:17px;height:17px;accent-color:var(--accent);flex:none}
.pmeta{display:flex;flex-wrap:wrap;align-items:center;gap:8px;flex:1;font-size:.84rem}
.pmeta code{font-family:var(--mono);font-size:.8rem;color:var(--text);
  background:color-mix(in srgb,var(--accent) 10%,transparent);padding:1px 7px;border-radius:6px}
.pbytes{font-size:.7rem;color:var(--faint);font-variant-numeric:tabular-nums}
.pbad{font-size:.7rem;color:var(--error);font-weight:600}
.pdiff{flex-basis:100%;margin:2px 0 0 25px;font-size:.76rem}
.pdiff summary{cursor:pointer;color:var(--accent);font-weight:600}
.pdiff pre{font-family:var(--mono);font-size:.76rem;line-height:1.45;overflow-x:auto;max-height:280px;
  overflow-y:auto;padding:10px;border-radius:8px;background:#05080d;border:1px solid var(--border);
  margin:6px 0 0;white-space:pre}
.pactions{display:flex;flex-wrap:wrap;align-items:center;gap:10px;margin-top:12px;padding-top:12px;
  border-top:1px solid var(--border)}
.pactions .toggle{margin-left:0}
#improveout{margin-top:12px}
.result{padding:12px;border-radius:8px;background:var(--surface-2);line-height:1.55;font-size:14px}
.result.ok{border-left:3px solid var(--success)}
.result.bad{border-left:3px solid var(--error)}
table.hist{width:100%;border-collapse:collapse;margin-top:6px;font-size:13px;font-family:var(--mono)}
table.hist td{padding:3px 6px;border-bottom:1px solid var(--grid);vertical-align:top}
table.hist td.rat{color:var(--muted);font-style:italic;font-family:var(--font)}
#improve label.toggle{font-size:13px;opacity:.85;margin-left:auto}
#improve select{margin-left:6px}
#stopbtn{background:linear-gradient(180deg,#fb7185,var(--danger));color:#fff;border-color:transparent}

/* --- panel intro line ------------------------------------------------ */
.lede{font-size:.86rem;color:var(--muted);line-height:1.5;margin:0 0 12px}
.lede b{color:var(--text)}
.emptyrow{color:var(--faint);font-family:var(--mono);font-size:.8rem;padding:12px 2px}
.kv{font-family:var(--mono);font-size:.78rem;color:var(--muted)}
.kv b{color:var(--text)}

/* --- fleet ----------------------------------------------------------- */
.fleet-bar{display:flex;flex-wrap:wrap;align-items:center;gap:10px;margin-bottom:12px}
.killbadge{font-family:var(--mono);font-size:.68rem;font-weight:700;text-transform:uppercase;
  letter-spacing:.05em;padding:4px 10px;border-radius:999px;border:1px solid var(--border);color:var(--faint)}
.killbadge.on{color:#fff;background:var(--danger);border-color:transparent;
  box-shadow:0 0 0 3px color-mix(in srgb,var(--danger) 30%,transparent)}

/* --- bundle checks --------------------------------------------------- */
.bcheck{display:flex;align-items:flex-start;gap:10px;padding:9px 10px;border:1px solid var(--border);
  border-radius:8px;margin:6px 0;font-size:.86rem}
.bcheck .led{width:9px;height:9px;border-radius:999px;flex:none;margin-top:5px;background:var(--faint)}
.bcheck.ok .led{background:var(--live);box-shadow:0 0 8px color-mix(in srgb,var(--live) 70%,transparent)}
.bcheck.bad .led{background:var(--danger)}
.bcheck .bd{color:var(--muted);font-family:var(--mono);font-size:.74rem;margin-top:2px}

/* --- help icons (click-to-open, mobile-friendly) --------------------- */
.help{display:inline-flex;align-items:center;justify-content:center;width:16px;height:16px;
  border-radius:999px;border:1px solid var(--border);background:var(--surface-2);
  color:var(--muted);font-family:var(--mono);font-size:.62rem;font-weight:700;line-height:1;
  cursor:pointer;margin-left:5px;padding:0;flex:none;vertical-align:middle;user-select:none}
.help:hover,.help:focus{color:var(--accent);border-color:var(--accent-2);outline:none}
.help-pop{position:fixed;z-index:60;max-width:290px;background:var(--elev);
  border:1px solid color-mix(in srgb,var(--accent) 35%,var(--border));border-radius:10px;
  box-shadow:var(--shadow);padding:11px 13px 12px;font-size:.82rem;line-height:1.5;color:var(--text)}
.help-pop .hx{float:right;margin:-2px -3px 0 10px;cursor:pointer;color:var(--faint);
  font-weight:700;font-family:var(--mono)}
.help-pop .hx:hover{color:var(--text)}

/* --- skills registry ------------------------------------------------- */
.skitem{display:flex;flex-wrap:wrap;align-items:center;gap:10px;padding:10px 12px;
  border:1px solid var(--border);border-radius:8px;margin:6px 0}
.skitem .sknm{font-family:var(--mono);font-weight:700;color:var(--text)}
.skitem .skds{flex:1;font-size:.84rem;color:var(--muted);min-width:180px}
.skitem .skcap{font-family:var(--mono);font-size:.66rem;color:var(--faint)}
"""
