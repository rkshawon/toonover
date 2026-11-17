"""Minimal web app: paste a link or upload a clip, pick a style, download the cartoon.

Run:  toonover serve   then open http://127.0.0.1:8000
"""
from __future__ import annotations

import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from ..config import OUTPUT, WORK
from ..convert import convert_redraw, convert_video
from ..redraw.comfy import presets as redraw_presets
from ..stylize import ALL_HELP

app = FastAPI(title="Toonover")
POOL = ThreadPoolExecutor(max_workers=1)   # one render at a time on a normal PC
JOBS: dict[str, dict] = {}
LOCK = threading.Lock()


def _run(job_id: str, source: str, opts: dict):
    def prog(stage, f):
        with LOCK:
            JOBS[job_id].update(stage=stage, progress=round(f, 3))

    def log(msg):
        with LOCK:
            JOBS[job_id]["log"].append(msg)

    try:
        fn = convert_redraw if opts["style"] in redraw_presets() else convert_video
        out = fn(source, out_path=OUTPUT / f"toonover_{job_id}.mp4", log=log, progress=prog,
                            job_id=job_id, **opts)
        with LOCK:
            JOBS[job_id].update(status="done", output=str(out))
    except Exception as e:
        with LOCK:
            JOBS[job_id].update(status="error", error=str(e), trace=traceback.format_exc()[-2000:])


@app.post("/api/jobs")
async def create_job(link: str = Form(""), title: str = Form(""), style: str = Form("toon"),
                     channel: str = Form(""), file: UploadFile | None = File(None)):
    job_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    if file is not None and file.filename:
        d = WORK / job_id
        d.mkdir(parents=True, exist_ok=True)
        src = d / f"upload_{file.filename.replace('/', '_').replace(chr(92), '_')}"
        src.write_bytes(await file.read())
        source = str(src)
    elif link.strip():
        source = link.strip()
    else:
        raise HTTPException(400, "Give a link or upload a video")
    if style not in ALL_HELP:
        style = "toon"
    opts = dict(style=style, title=title.strip(), channel=channel.strip())
    with LOCK:
        JOBS[job_id] = dict(id=job_id, status="running", stage="queued", progress=0.0, log=[], source=source)
    POOL.submit(_run, job_id, source, opts)
    return {"id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    with LOCK:
        j = JOBS.get(job_id)
        if not j:
            raise HTTPException(404)
        return {k: v for k, v in j.items() if k != "trace"}


@app.get("/api/jobs/{job_id}/video")
def job_video(job_id: str):
    j = JOBS.get(job_id)
    if not j or j.get("status") != "done":
        raise HTTPException(404)
    return FileResponse(j["output"], media_type="video/mp4", filename=f"toonover_{job_id}.mp4")


@app.get("/api/styles")
def styles():
    return [{"id": k, "help": v} for k, v in ALL_HELP.items()]


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Toonover</title>
<style>
 :root{--bg:#0f1220;--card:#181c2e;--ink:#eef0ff;--mut:#9aa0c3;--acc:#ffcc33}
 *{box-sizing:border-box} body{margin:0;font-family:system-ui,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--ink)}
 main{max-width:760px;margin:40px auto;padding:0 18px}
 h1{font-size:40px;margin:0 0 4px;letter-spacing:-1px} h1 span{color:var(--acc)}
 p.sub{color:var(--mut);margin:0 0 24px}
 .card{background:var(--card);border-radius:16px;padding:20px;margin-bottom:18px}
 label{display:block;font-size:13px;color:var(--mut);margin:12px 0 6px}
 input,select{width:100%;padding:12px;border-radius:10px;border:1px solid #2c3352;background:#0f1324;color:var(--ink);font-size:15px}
 .row{display:grid;grid-template-columns:1fr 1fr;gap:12px}
 button{margin-top:18px;width:100%;padding:14px;border:0;border-radius:12px;background:var(--acc);color:#1a1400;font-weight:800;font-size:16px;cursor:pointer}
 button:disabled{opacity:.5}
 .bar{height:10px;background:#0f1324;border-radius:6px;overflow:hidden;margin-top:10px}
 .bar>div{height:100%;width:0;background:var(--acc);transition:width .3s}
 video{width:100%;max-height:70vh;border-radius:12px;background:#000;margin-top:12px}
 .mut{color:var(--mut);font-size:13px} a{color:var(--acc)}
 .chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:6px}
 .chip{padding:6px 10px;border-radius:999px;background:#0f1324;border:1px solid #2c3352;cursor:pointer;font-size:13px;user-select:none}
 .chip.on{background:var(--acc);color:#1a1400;border-color:var(--acc)}
</style></head><body><main>
<h1>Toon<span>over</span></h1>
<p class="sub">Paste a meme link or upload a clip. Get the same video, same people, as a cartoon.</p>
<form class="card" id="f">
  <label>Video link (YouTube, Facebook, Instagram, TikTok)</label>
  <input name="link" placeholder="https://www.youtube.com/shorts/...">
  <label>…or upload a file</label>
  <input type="file" name="file" accept="video/*">
  <div class="row">
    <div><label>Title bar (optional)</label><input name="title" placeholder="e.g. Samay was Rich"></div>
    <div><label>Bottom bar (optional)</label><input name="channel" placeholder="e.g. Toonover"></div>
  </div>
  <label>Style</label>
  <select name="style" id="style"></select>
  <p class="mut" id="stylehelp"></p>
  <button id="go">Make cartoon</button>
  <p class="mut">Only use clips you have the right to use.</p>
</form>
<div class="card" id="status" hidden>
  <div id="stage">Starting…</div>
  <div class="bar"><div id="pb"></div></div>
  <div id="result"></div>
</div>
</main>
<script>
fetch('/api/styles').then(r=>r.json()).then(ss=>{
  const sel=document.getElementById('style'), help=document.getElementById('stylehelp');
  ss.forEach(x=>{const o=document.createElement('option');o.value=x.id;o.textContent=x.id+' - '+x.help;sel.appendChild(o)});
});
const names={queued:'Waiting',download:'Downloading',prepare:'Preparing video',cartoonize:'Drawing cartoon',redraw:'Redrawing keyframes on GPU',done:'Done'};
document.getElementById('f').onsubmit=async e=>{
  e.preventDefault(); const go=document.getElementById('go'); go.disabled=true;
  const st=document.getElementById('status'); st.hidden=false; document.getElementById('result').innerHTML='';
  const r=await fetch('/api/jobs',{method:'POST',body:new FormData(e.target)});
  if(!r.ok){document.getElementById('stage').textContent='Error: '+(await r.text());go.disabled=false;return}
  const {id}=await r.json();
  const t=setInterval(async()=>{
    const j=await (await fetch('/api/jobs/'+id)).json();
    document.getElementById('stage').textContent=(names[j.stage]||j.stage)+' '+Math.round(j.progress*100)+'%';
    document.getElementById('pb').style.width=(j.progress*100)+'%';
    if(j.status==='done'){clearInterval(t);go.disabled=false;
      document.getElementById('result').innerHTML='<video controls autoplay src="/api/jobs/'+id+'/video"></video><p><a href="/api/jobs/'+id+'/video" download>Download MP4</a></p>';}
    if(j.status==='error'){clearInterval(t);go.disabled=false;document.getElementById('stage').textContent='Error: '+j.error;}
  },1000);
};
</script></body></html>"""
