"""Local prompt-to-image app with optional Diffusers LoRA adapter."""
import os
from functools import lru_cache
from pathlib import Path
import gradio as gr
import torch
from diffusers import StableDiffusionPipeline

@lru_cache(maxsize=1)
def load_pipeline():
    model = os.getenv('BASE_MODEL', 'stable-diffusion-v1-5/stable-diffusion-v1-5')
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    dtype = torch.float16 if device in ('cuda', 'mps') else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype=dtype)
    adapter = os.getenv('LORA_PATH')
    if adapter:
        path = Path(adapter).expanduser()
        if not path.exists(): raise ValueError(f'LoRA path does not exist: {path}')
        pipe.load_lora_weights(str(path))
    return pipe.to(device), device

def generate(prompt, negative_prompt, steps, guidance, seed):
    if not prompt.strip(): raise gr.Error('Enter a prompt.')
    pipe, device = load_pipeline()
    generator = torch.Generator(device='cpu' if device == 'mps' else device).manual_seed(int(seed))
    image = pipe(prompt=prompt, negative_prompt=negative_prompt or None, num_inference_steps=int(steps), guidance_scale=float(guidance), generator=generator).images[0]
    return image

with gr.Blocks(title='AI Image Generator') as demo:
    gr.Markdown('# AI Image Generator\nGenerate locally with Stable Diffusion and an optional LoRA adapter.')
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label='Prompt', lines=3)
            negative = gr.Textbox(label='Negative prompt')
            steps = gr.Slider(10, 50, value=25, step=1, label='Inference steps')
            guidance = gr.Slider(1, 15, value=7.5, label='Guidance scale')
            seed = gr.Number(value=42, precision=0, label='Seed')
            button = gr.Button('Generate', variant='primary')
        output = gr.Image(label='Generated image', type='pil')
    button.click(generate, inputs=[prompt, negative, steps, guidance, seed], outputs=output)

if __name__ == '__main__': demo.queue(default_concurrency_limit=1).launch(server_name='127.0.0.1')
