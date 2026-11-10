"""Local queued image API. Run with uvicorn image_generator.app:app."""
import json
import logging
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv
from . import engine

load_dotenv()
logger = logging.getLogger(__name__)
DATA = Path(os.getenv('IMAGE_OUTPUT_DIR', 'generated')).resolve()
DATA.mkdir(parents=True, exist_ok=True)
app = FastAPI(title='AI Image Studio', version='1.0.0')
executor = ThreadPoolExecutor(max_workers=1)
lock = threading.Lock()
jobs = {}

class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    negative_prompt: str = Field(default='', max_length=2000)
    steps: int = Field(default=25, ge=1, le=60)
    guidance: float = Field(default=7.5, ge=0, le=20)
    seed: int = Field(default=42, ge=0, le=2147483647)
    width: int = Field(default=512, ge=128, le=768)
    height: int = Field(default=512, ge=128, le=768)

    @field_validator('width', 'height')
    @classmethod
    def divisible_by_eight(cls, value):
        if value % 8:
            raise ValueError('Dimensions must be divisible by 8')
        return value

    @field_validator('prompt')
    @classmethod
    def nonblank(cls, value):
        if not value.strip(): raise ValueError('Enter a prompt')
        return value.strip()

def execute(job_id, parameters):
    with lock: jobs[job_id]['status'] = 'running'
    try:
        image = engine.generate(parameters)
        image_temp = DATA / f'{job_id}.png.tmp'
        image.save(image_temp, format='PNG')
        image_temp.replace(DATA / f'{job_id}.png')
        with lock:
            record = dict(jobs[job_id], status='completed', image_url=f'/images/{job_id}', completed_at=datetime.now(timezone.utc).isoformat())
        metadata_temp = DATA / f'{job_id}.json.tmp'
        metadata_temp.write_text(json.dumps(record, indent=2), encoding='utf-8')
        metadata_temp.replace(DATA / f'{job_id}.json')
        with lock:
            jobs[job_id] = record
    except Exception as exc:
        logger.exception('Image generation failed for %s', job_id)
        message = str(exc) if isinstance(exc, ValueError) else 'Generation failed. Check the server log, model access and available memory.'
        with lock: jobs[job_id].update(status='failed', error=message)

@app.get('/', response_class=HTMLResponse)
def home(): return Path(__file__).with_name('index.html').read_text(encoding='utf-8')

@app.get('/health')
def health(): return {'status': 'ok'}

@app.get('/api/config')
def config():
    values = engine.settings()
    return {'model': values['model'], 'adapter_loaded': bool(values['adapter']), 'device': values['device'], 'test_model': 'tiny-' in values['model']}

@app.post('/api/jobs', status_code=202)
def submit(request: GenerateRequest):
    job_id = uuid.uuid4().hex
    with lock:
        if sum(j['status'] in ('queued', 'running') for j in jobs.values()) >= 4:
            raise HTTPException(429, 'Queue is full. Wait for a generation to finish.')
        # Bound in-memory history; completed records remain on disk.
        if len(jobs) > 100:
            for key in list(jobs):
                if jobs[key]['status'] in ('completed', 'failed'):
                    del jobs[key]
                    if len(jobs) <= 80: break
        jobs[job_id] = {'id': job_id, 'status': 'queued', 'parameters': request.model_dump(), 'model': engine.settings()['model'], 'adapter': bool(engine.settings()['adapter']), 'created_at': datetime.now(timezone.utc).isoformat()}
    executor.submit(execute, job_id, request.model_dump())
    return {'id': job_id, 'status': 'queued'}

def valid_id(job_id):
    if len(job_id) != 32 or any(c not in '0123456789abcdef' for c in job_id):
        raise HTTPException(404, 'Image not found')

@app.get('/api/jobs/{job_id}')
def status(job_id: str):
    valid_id(job_id)
    with lock:
        if job_id in jobs: return dict(jobs[job_id])
    path = DATA / f'{job_id}.json'
    if path.is_file(): return json.loads(path.read_text())
    raise HTTPException(404, 'Job not found')

@app.get('/api/gallery')
def gallery():
    records = []
    for path in sorted(DATA.glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True)[:30]:
        try: records.append(json.loads(path.read_text()))
        except (ValueError, OSError): logger.warning('Skipping invalid gallery record %s', path.name)
    return records

@app.get('/images/{job_id}')
def image(job_id: str, download: bool = False):
    valid_id(job_id)
    path = DATA / f'{job_id}.png'
    if not path.is_file(): raise HTTPException(404, 'Image not found')
    return FileResponse(path, media_type='image/png', filename=f'{job_id}.png' if download else None)
