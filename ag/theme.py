"""Presentation layer for the web app — fonts, palette, and layout.

This file is *evolvable* (listed in config.evolvable_paths), so AG can iterate its
own visual design within the test gate without touching server logic. Keep it to
CSS + a fonts <link>; the server injects THEME_CSS into the page. A bad edit that
breaks the page's required hooks (checked by tests/test_server.py) is rolled back.

Design goals: high visibility (strong contrast, generous type), a professional
typeface (Inter for UI, JetBrains Mono for logs) with robust system fallbacks, and
a clean light/dark palette that adapts to the viewer's OS setting.
"""
from __future__ import annotations

# Google Fonts load when the browser is online; the fallback stacks below keep the
# UI professional offline. No other external assets are used.
FONT_LINK = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?'
    'family=Inter:wght@400;500;600;700;800&'
    'family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">'
)

THEME_CSS = """
:root{
  color-scheme:light dark;
  --font:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
  --mono:'JetBrains Mono',ui-monospace,'SF Mono',Menlo,Consolas,'Liberation Mono',monospace;
  --bg:#eef1f7; --surface:#ffffff; --surface-2:#f7f9fc; --elev:#ffffff;
  --text:#0b1220; --muted:#57616f; --faint:#8a94a3; --border:#e2e7f0;
  --accent:#4f46e5; --accent-2:#6366f1; --accent-ink:#ffffff;
  --tool:#2563eb; --web:#0891b2; --error:#dc2626; --success:#16a34a; --warn:#c2410c;
  --shadow:0 1px 2px rgba(16,24,40,.06),0 10px 30px rgba(16,24,40,.08);
  --radius:14px; --radius-sm:10px;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#080b12; --surface:#111725; --surface-2:#0d1220; --elev:#161d2e;
  --text:#eef2f8; --muted:#9aa6b8; --faint:#6b7688; --border:#212a3d;
  --accent:#6366f1; --accent-2:#818cf8; --accent-ink:#0b0f1a;
  --tool:#60a5fa; --web:#22d3ee; --error:#f87171; --success:#4ade80; --warn:#fb923c;
  --shadow:0 1px 2px rgba(0,0,0,.5),0 14px 40px rgba(0,0,0,.45);
}}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{
  font-family:var(--font); font-size:16px; line-height:1.55; color:var(--text);
  background:
    radial-gradient(1200px 600px at 100% -10%, color-mix(in srgb,var(--accent) 12%,transparent), transparent 60%),
    var(--bg);
  margin:0; padding:28px 20px 64px; min-height:100vh;
  -webkit-font-smoothing:antialiased; text-rendering:optimizeLegibility;
}
.wrap{max-width:920px;margin:0 auto}
header{display:flex;align-items:center;gap:14px;margin-bottom:22px}
.logo{font-size:34px;line-height:1;filter:drop-shadow(0 2px 6px rgba(0,0,0,.18))}
.brand h1{font-size:1.5rem;font-weight:800;letter-spacing:-.02em;margin:0}
.brand .sub{font-size:.82rem;color:var(--muted);font-weight:500;margin-top:1px}
#status{margin-left:auto;font-size:.8rem;font-weight:600;color:var(--muted);
  padding:6px 12px;border:1px solid var(--border);border-radius:999px;background:var(--surface);
  white-space:nowrap}
.panel{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);
  box-shadow:var(--shadow);padding:18px;margin-bottom:18px}
textarea{width:100%;min-height:96px;font-family:var(--font);font-size:1.02rem;line-height:1.5;
  padding:14px 16px;border-radius:var(--radius-sm);border:1px solid var(--border);
  background:var(--surface-2);color:var(--text);resize:vertical;transition:border-color .15s,box-shadow .15s}
textarea::placeholder{color:var(--faint)}
textarea:focus{outline:none;border-color:var(--accent);
  box-shadow:0 0 0 4px color-mix(in srgb,var(--accent) 22%,transparent)}
.controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-top:14px}
button{font-family:var(--font);font-size:.95rem;font-weight:600;padding:11px 18px;cursor:pointer;
  border-radius:var(--radius-sm);border:1px solid var(--border);background:var(--surface);
  color:var(--text);transition:transform .05s,background .15s,border-color .15s,box-shadow .15s}
button:hover{border-color:var(--accent-2)}
button:active{transform:translateY(1px)}
button:disabled{opacity:.55;cursor:default}
button.primary{background:linear-gradient(180deg,var(--accent-2),var(--accent));
  color:var(--accent-ink);border-color:transparent;
  box-shadow:0 1px 0 rgba(255,255,255,.15) inset,0 6px 18px color-mix(in srgb,var(--accent) 40%,transparent)}
button.primary:hover{filter:brightness(1.06)}
.toggle{display:inline-flex;gap:8px;align-items:center;font-size:.9rem;font-weight:500;
  color:var(--muted);margin-left:auto;user-select:none}
.toggle input{width:16px;height:16px;accent-color:var(--accent)}
/* right-aligned run controls (thinking selector + internet toggle) */
.ctl-right{margin-left:auto;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
.ctl{display:inline-flex;gap:7px;align-items:center;font-size:.9rem;font-weight:500;
  color:var(--muted);user-select:none}
.ctl input{width:16px;height:16px;accent-color:var(--accent)}
.ctl-select{font-family:var(--font);font-size:.86rem;font-weight:600;color:var(--text);
  background:var(--surface-2);border:1px solid var(--border);border-radius:8px;
  padding:5px 8px;cursor:pointer}
.ctl-select:focus{outline:none;border-color:var(--accent);
  box-shadow:0 0 0 3px color-mix(in srgb,var(--accent) 18%,transparent)}
h2{display:flex;align-items:center;gap:8px;font-size:.76rem;font-weight:700;text-transform:uppercase;
  letter-spacing:.09em;color:var(--faint);margin:0 0 10px 2px}
#log{font-family:var(--mono);font-size:.82rem;line-height:1.5;max-height:340px;overflow:auto;
  border-radius:var(--radius-sm);background:var(--surface-2);border:1px solid var(--border);padding:8px}
#log:empty::after{content:'The run trace appears here — thoughts, tool & internet calls, errors.';
  color:var(--faint);font-family:var(--font);font-size:.85rem;display:block;padding:14px}
.ev{display:flex;gap:10px;padding:5px 10px;border-left:3px solid var(--faint);
  border-radius:0 6px 6px 0;margin:3px 0;white-space:pre-wrap;word-break:break-word}
.ev:nth-child(even){background:color-mix(in srgb,var(--text) 3%,transparent)}
.ev .tag{flex:none;min-width:78px;color:var(--muted);font-weight:500;text-transform:uppercase;
  font-size:.7rem;letter-spacing:.04em;padding-top:1px}
.ev.tool{border-color:var(--tool)} .ev.web{border-color:var(--web)}
.ev.error{border-color:var(--error);color:var(--error)}
.ev.result{border-color:var(--success)}
#cards{display:none;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:18px}
@media(max-width:560px){#cards{grid-template-columns:repeat(2,1fr)}}
.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-sm);
  padding:14px 16px;box-shadow:var(--shadow)}
.card .n{font-size:1.9rem;font-weight:800;letter-spacing:-.02em;font-variant-numeric:tabular-nums}
.card .l{font-size:.72rem;font-weight:600;text-transform:uppercase;letter-spacing:.06em;color:var(--faint);margin-top:2px}
.bar{height:7px;border-radius:999px;background:color-mix(in srgb,var(--text) 10%,transparent);
  margin-top:10px;overflow:hidden}
.bar>i{display:block;height:100%;border-radius:999px;
  background:linear-gradient(90deg,var(--accent-2),var(--accent));transition:width .5s cubic-bezier(.2,.8,.2,1)}
#answer{font-size:1.04rem;line-height:1.65;white-space:pre-wrap;word-break:break-word;min-height:44px;color:var(--text)}
#answer:empty::after{content:'—';color:var(--faint)}
#tools{display:none;width:100%;border-collapse:collapse;font-size:.86rem}
#tools th{text-align:left;font-size:.7rem;text-transform:uppercase;letter-spacing:.05em;
  color:var(--faint);padding:6px 10px;border-bottom:1px solid var(--border)}
#tools td{padding:8px 10px;border-bottom:1px solid var(--border)}
#tools tr:last-child td{border-bottom:none;color:var(--muted)}
#tools td.n{text-align:right;font-variant-numeric:tabular-nums;font-weight:600}
.pill{display:inline-block;font-size:.7rem;font-weight:600;padding:2px 9px;border-radius:999px;
  border:1px solid var(--border);text-transform:capitalize}
.pill.available{color:var(--success);border-color:color-mix(in srgb,var(--success) 45%,transparent)}
.pill.degraded{color:var(--warn);border-color:color-mix(in srgb,var(--warn) 45%,transparent)}
.pill.unavailable{color:var(--faint)}
#update{display:none;align-items:center;gap:12px;margin-bottom:18px;padding:14px 16px;
  border:1px solid color-mix(in srgb,var(--warn) 55%,var(--border));border-radius:var(--radius-sm);
  background:color-mix(in srgb,var(--warn) 12%,var(--surface))}
#update .g{flex:1;font-size:.92rem;font-weight:500}
#update .yes{background:var(--warn);color:#fff;border-color:transparent}

/* --- version badge in the header ------------------------------------- */
.ver{font-family:var(--mono);font-size:.6rem;font-weight:700;letter-spacing:.03em;
  vertical-align:middle;margin-left:9px;padding:2px 8px;border-radius:999px;
  color:var(--accent);background:color-mix(in srgb,var(--accent) 14%,transparent)}

/* --- evolve status strip: when & how AG evolves itself --------------- */
.evostatus{display:flex;flex-wrap:wrap;align-items:center;gap:6px 8px;
  margin:-4px 0 18px;padding:10px 14px;border-radius:var(--radius-sm);
  background:var(--surface);border:1px solid var(--border);box-shadow:var(--shadow);
  font-size:.8rem;color:var(--muted);line-height:1.5}
.evostatus b{color:var(--text);font-weight:600}
.evostatus i{color:var(--faint)}
.evostatus .g-ok{color:var(--success);font-weight:600}
.evostatus .g-off{color:var(--warn);font-weight:600}
.evostatus .dot{width:8px;height:8px;border-radius:999px;background:var(--faint);
  flex:none;margin-right:4px}
.evostatus .dot.ok{background:var(--success)} .evostatus .dot.off{background:var(--warn)}
.evostatus.busy{border-color:var(--accent);
  background:color-mix(in srgb,var(--accent) 8%,var(--surface))}
.evostatus .dot.spin{background:var(--accent);animation:agspin 1s linear infinite;
  box-shadow:0 0 0 3px color-mix(in srgb,var(--accent) 22%,transparent)}
@keyframes agspin{to{transform:rotate(360deg)}}

/* --- command vs. view buttons: two visually distinct kinds ----------- */
.btngroup{border-radius:var(--radius-sm);padding:12px 14px}
.grouplabel{display:block;font-size:.68rem;font-weight:700;text-transform:uppercase;
  letter-spacing:.08em;color:var(--faint);margin:0 0 8px 2px}
.cmd-group{background:color-mix(in srgb,var(--accent) 6%,transparent);
  border:1px solid color-mix(in srgb,var(--accent) 22%,var(--border))}
.view-group{margin-top:12px;background:var(--surface-2);border:1px dashed var(--border)}
.cmd-group .grouplabel{color:var(--accent)}
button.cmd{background:linear-gradient(180deg,var(--accent-2),var(--accent));
  color:var(--accent-ink);border-color:transparent;
  box-shadow:0 1px 0 rgba(255,255,255,.15) inset,
    0 6px 18px color-mix(in srgb,var(--accent) 32%,transparent)}
button.cmd:hover{filter:brightness(1.07);border-color:transparent}
button.cmd.evolve{background:linear-gradient(180deg,#a855f7,#7c3aed)}
button.view{background:transparent;color:var(--muted);border:1px dashed var(--border);
  box-shadow:none}
button.view:hover{color:var(--text);border-style:solid;border-color:var(--accent-2)}
.cmd-note{font-size:.74rem;color:var(--faint);font-style:italic}

/* --- context-in-use chips ------------------------------------------- */
#ctxbar{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin:0 2px 10px}
.ctxlbl{font-size:.72rem;font-weight:600;text-transform:uppercase;letter-spacing:.05em;
  color:var(--faint);margin-right:2px}
.chip{display:inline-flex;align-items:center;gap:6px;font-size:.76rem;font-weight:600;
  padding:5px 11px;border-radius:999px;border:1px solid var(--border);color:var(--faint);
  background:var(--surface-2);opacity:.55;transition:opacity .2s,color .2s,
    border-color .2s,box-shadow .2s}
.chip .cc{font-variant-numeric:tabular-nums}
.chip.active{opacity:1;color:var(--accent);
  border-color:color-mix(in srgb,var(--accent) 55%,transparent);
  background:color-mix(in srgb,var(--accent) 12%,var(--surface));
  box-shadow:0 0 0 3px color-mix(in srgb,var(--accent) 14%,transparent)}

/* --- conversation transcript ---------------------------------------- */
.chatwrap{padding:12px}
.chat-toolbar{display:flex;align-items:center;gap:10px;margin-bottom:8px}
.ct-hint{font-size:.74rem;color:var(--faint)}
.spacer{margin-left:auto}
.linkbtn{background:none;border:none;color:var(--muted);font-size:.78rem;font-weight:600;
  padding:4px 6px;cursor:pointer;border-radius:6px}
.linkbtn:hover{color:var(--accent);background:color-mix(in srgb,var(--accent) 10%,transparent)}
#chat{display:flex;flex-direction:column;gap:14px;max-height:460px;overflow:auto;padding:6px}
#chat:empty::after{content:'Your conversation appears here — ask something above to begin.';
  color:var(--faint);font-size:.9rem;display:block;padding:22px;text-align:center}
.msg{display:flex;flex-direction:column;gap:4px;max-width:88%}
.msg.user{align-self:flex-end;align-items:flex-end}
.msg.ai{align-self:flex-start;align-items:flex-start}
.msg .who{font-size:.66rem;font-weight:700;text-transform:uppercase;letter-spacing:.05em;
  color:var(--faint);display:flex;gap:8px;align-items:center}
.msg .who .ts{font-weight:500;text-transform:none;letter-spacing:0;opacity:.8}
.msg .body{padding:11px 14px;border-radius:14px;line-height:1.6;white-space:pre-wrap;
  word-break:break-word;font-size:.98rem}
.msg.user .body{background:linear-gradient(180deg,var(--accent-2),var(--accent));
  color:var(--accent-ink);border-bottom-right-radius:4px}
.msg.ai .body{background:var(--surface-2);border:1px solid var(--border);
  border-bottom-left-radius:4px}
.msg.ai .body.pending{color:var(--muted);font-style:italic}
.msg.ai.processing{outline:2px solid color-mix(in srgb,var(--accent) 45%,transparent);
  outline-offset:4px;border-radius:14px}
.msg .ctx{display:flex;flex-wrap:wrap;gap:6px;margin-top:2px}
.msg .ctx span{font-size:.7rem;color:var(--muted);padding:1px 8px;border-radius:999px;
  border:1px solid var(--border);background:var(--surface)}
.msg .live-ctx{font-size:.74rem;color:var(--muted);margin-top:2px}
.msg .live-ctx .using{font-weight:600;color:var(--accent)}
.msg .live-ctx ul,.memfacts ul{margin:4px 0 0;padding-left:18px}
.msg .live-ctx li,.memfacts li{margin:1px 0}
.memfacts{margin-top:4px;font-size:.74rem;color:var(--muted)}
.memfacts summary{cursor:pointer;color:var(--accent);font-weight:600}

/* --- where AG is running + how it evolves ---------------------------- */
.whereami{display:flex;flex-wrap:wrap;align-items:center;gap:6px 8px;
  margin:0 0 12px;padding:9px 14px;border-radius:var(--radius-sm);
  background:var(--surface);border:1px solid var(--border);box-shadow:var(--shadow);
  font-size:.8rem;color:var(--muted);line-height:1.5}
.whereami b{color:var(--text);font-weight:600}
.whereami .dot{width:8px;height:8px;border-radius:999px;background:var(--faint);
  flex:none;margin-right:2px}
.whereami .dot.ok{background:var(--success)}
.tagpill{font-size:.68rem;font-weight:700;padding:2px 8px;border-radius:999px;
  border:1px solid var(--border)}
.tagpill.local{color:var(--success);
  border-color:color-mix(in srgb,var(--success) 45%,transparent)}
.tagpill.lan{color:var(--warn);
  border-color:color-mix(in srgb,var(--warn) 45%,transparent)}
.tagpill.idle{color:var(--faint)}
.tagpill.evolving{color:var(--accent);font-weight:700;
  border-color:color-mix(in srgb,var(--accent) 55%,transparent);
  background:color-mix(in srgb,var(--accent) 12%,var(--surface))}

/* --- evolve directive box ------------------------------------------- */
.directive{width:100%;min-height:52px;font-family:var(--font);font-size:.9rem;
  line-height:1.5;padding:10px 12px;border-radius:var(--radius-sm);
  border:1px dashed color-mix(in srgb,var(--accent) 40%,var(--border));
  background:var(--surface);color:var(--text);resize:vertical;margin-bottom:10px;
  transition:border-color .15s,box-shadow .15s}
.directive::placeholder{color:var(--faint)}
.directive:focus{outline:none;border-style:solid;border-color:var(--accent);
  box-shadow:0 0 0 4px color-mix(in srgb,var(--accent) 18%,transparent)}

/* --- proposed-changes checklist (human-in-the-loop evolve) ----------- */
.proposal{border:1px solid var(--border);border-radius:var(--radius-sm);
  background:var(--surface);padding:14px}
.phead{display:flex;flex-wrap:wrap;align-items:baseline;gap:8px 12px;margin-bottom:8px}
.phead b{font-size:.95rem}
.pnote{font-size:.72rem;color:var(--faint);font-style:italic}
.prationale{font-size:.84rem;color:var(--muted);line-height:1.5;margin-bottom:10px;
  padding:8px 10px;border-radius:8px;background:var(--surface-2)}
.pitem{display:flex;flex-wrap:wrap;align-items:center;gap:8px;padding:9px 10px;
  border:1px solid var(--border);border-radius:8px;margin:6px 0;cursor:pointer}
.pitem:hover{border-color:var(--accent-2)}
.pitem.invalid{opacity:.6;cursor:not-allowed}
.pitem .psel{width:17px;height:17px;accent-color:var(--accent);flex:none}
.pmeta{display:flex;flex-wrap:wrap;align-items:center;gap:8px;flex:1;font-size:.86rem}
.pmeta code{font-family:var(--mono);font-size:.82rem;color:var(--text);
  background:color-mix(in srgb,var(--accent) 10%,transparent);padding:1px 7px;border-radius:6px}
.pbytes{font-size:.72rem;color:var(--faint);font-variant-numeric:tabular-nums}
.pbad{font-size:.72rem;color:var(--error);font-weight:600}
.pdiff{flex-basis:100%;margin:2px 0 0 25px;font-size:.76rem}
.pdiff summary{cursor:pointer;color:var(--accent);font-weight:600}
.pdiff pre{font-family:var(--mono);font-size:.76rem;line-height:1.45;overflow-x:auto;
  max-height:280px;overflow-y:auto;padding:10px;border-radius:8px;
  background:var(--surface-2);border:1px solid var(--border);margin:6px 0 0;white-space:pre}
.pactions{display:flex;flex-wrap:wrap;align-items:center;gap:10px;margin-top:12px;
  padding-top:12px;border-top:1px solid var(--border)}
.pactions .toggle{margin-left:0}
"""
