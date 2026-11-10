"""Optional real-model integration: RUN_MODEL_TESTS=1 python -m pytest -q."""
import os
import pytest

@pytest.mark.skipif(os.getenv('RUN_MODEL_TESTS') != '1', reason='Downloads the official tiny test model')
def test_train_export_reload_and_generate(tmp_path, monkeypatch):
    from image_generator.sample_data import create
    from image_generator.train import train
    from image_generator import engine
    from safetensors.torch import load_file
    model = 'hf-internal-testing/tiny-stable-diffusion-torch'
    dataset = tmp_path / 'data'; create(dataset)
    adapter = tmp_path / 'adapter'
    metrics = train(model, dataset, adapter, steps=2, resolution=32, rank=2)
    assert len(metrics['losses']) == 2 and metrics['trainable_parameters'] > 0
    weights = load_file(adapter / 'pytorch_lora_weights.safetensors')
    assert any('.lora.up.weight' in k and v.abs().sum() > 0 for k,v in weights.items())
    monkeypatch.setenv('BASE_MODEL', model)
    monkeypatch.setenv('MODEL_VARIANT', '')
    monkeypatch.setenv('LORA_PATH', str(adapter))
    monkeypatch.setenv('DEVICE', 'cpu')
    engine.pipeline.cache_clear()
    try:
        image = engine.generate({'prompt':'a circle','negative_prompt':'','steps':2,'guidance':7.5,'seed':42,'width':128,'height':128})
        assert image.size == (128,128)
    finally:
        engine.pipeline.cache_clear()
