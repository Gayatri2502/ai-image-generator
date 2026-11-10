import time
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from image_generator import app as module

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'DATA', tmp_path)
    with module.lock: module.jobs.clear()
    return TestClient(module.app)

def wait(client, job_id):
    for _ in range(100):
        record = client.get('/api/jobs/' + job_id).json()
        if record['status'] in ('failed','completed'): return record
        time.sleep(.01)
    pytest.fail('Worker did not finish')

def test_generate_download_and_reload(client, monkeypatch):
    monkeypatch.setattr(module.engine, 'generate', lambda p: Image.new('RGB', (p['width'],p['height']), 'red'))
    response = client.post('/api/jobs', json={'prompt':'test circle', 'width':128,'height':128})
    assert response.status_code == 202
    record = wait(client, response.json()['id'])
    assert record['status'] == 'completed'
    assert client.get(record['image_url']).headers['content-type'] == 'image/png'
    assert 'attachment' in client.get(record['image_url'] + '?download=true').headers['content-disposition']
    with module.lock: module.jobs.clear()
    assert client.get('/api/jobs/'+record['id']).json()['status'] == 'completed'
    assert len(client.get('/api/gallery').json()) == 1

def test_errors_and_validation(client, monkeypatch):
    def failure(p): raise ValueError('Test failure')
    monkeypatch.setattr(module.engine,'generate',failure)
    for data in ({'prompt':' '},{'prompt':'ok','width':129},{'prompt':'ok','steps':0}):
        assert client.post('/api/jobs',json=data).status_code == 422
    record = wait(client,client.post('/api/jobs',json={'prompt':'test'}).json()['id'])
    assert record['status'] == 'failed' and record['error'] == 'Test failure'
    assert client.get('/images/not-a-real-id').status_code == 404

def test_dataset_paths(tmp_path):
    import json
    from image_generator.train import load_records
    (tmp_path/'metadata.jsonl').write_text(json.dumps({'file_name':'../secret.png','text':'x'}))
    with pytest.raises(ValueError): load_records(tmp_path)
