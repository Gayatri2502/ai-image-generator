import os
import threading
from functools import lru_cache
from pathlib import Path


def settings():
    return {'model': os.getenv('BASE_MODEL', 'stable-diffusion-v1-5/stable-diffusion-v1-5'),
            'adapter': os.getenv('LORA_PATH', ''), 'device': os.getenv('DEVICE', 'auto'), 'variant': os.getenv('MODEL_VARIANT', '')}


@lru_cache(maxsize=1)
def pipeline():
    import torch
    from diffusers import StableDiffusionPipeline
    config = settings()
    device = config['device']
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if device not in ('cuda', 'mps', 'cpu'):
        raise ValueError('DEVICE must be auto, cpu, mps or cuda')
    torch.set_num_threads(int(os.getenv('TORCH_NUM_THREADS', '4')))
    options = {'variant': config['variant']} if config['variant'] else {}
    pipe = StableDiffusionPipeline.from_pretrained(config['model'], torch_dtype=torch.float16 if device == 'cuda' else torch.float32, **options)
    if config['adapter']:
        adapter = Path(config['adapter']).expanduser().resolve()
        if not adapter.exists():
            raise ValueError('Configured LORA_PATH does not exist')
        # File adapters and directories are both supported, always using safetensors.
        if adapter.is_file():
            pipe.load_lora_weights(str(adapter.parent), weight_name=adapter.name, use_safetensors=True)
        else:
            pipe.load_lora_weights(str(adapter), use_safetensors=True)
    pipe.enable_attention_slicing()
    pipe.to(device)
    return pipe, device


_generation_lock = threading.Lock()

def generate(parameters: dict):
    import torch
    with _generation_lock:
        pipe, device = pipeline()
        generator = torch.Generator(device='cuda' if device == 'cuda' else 'cpu').manual_seed(parameters['seed'])
        result = pipe(prompt=parameters['prompt'], negative_prompt=parameters['negative_prompt'] or None,
                      num_inference_steps=parameters['steps'], guidance_scale=parameters['guidance'],
                      width=parameters['width'], height=parameters['height'], generator=generator)
        if result.nsfw_content_detected and any(result.nsfw_content_detected):
            raise ValueError('The model safety checker filtered this image. Try another prompt.')
        return result.images[0]
