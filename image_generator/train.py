"""Train a Stable Diffusion UNet LoRA from a captioned JSONL image dataset."""
import argparse
import json
import random
from pathlib import Path


def load_records(directory):
    directory = Path(directory).resolve()
    records = []
    for line in (directory / 'metadata.jsonl').read_text(encoding='utf-8').splitlines():
        if not line.strip(): continue
        item = json.loads(line)
        path = (directory / item['file_name']).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise ValueError('Each file_name must reference an image inside the dataset directory')
        if not isinstance(item.get('text'), str) or not item['text'].strip():
            raise ValueError('Each record needs a nonempty text caption')
        records.append((path, item['text']))
    if not records: raise ValueError('Dataset is empty')
    return records


def train(model, dataset, output, steps=100, resolution=512, rank=4, learning_rate=1e-4, seed=42, device='cpu', variant=None):
    import torch
    import torch.nn.functional as F
    from PIL import Image, ImageOps
    from diffusers import StableDiffusionPipeline, DDPMScheduler
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
    records = load_records(dataset)
    if steps < 1 or resolution < 32 or resolution % 8 or not 1 <= rank <= 64:
        raise ValueError('Use positive steps, resolution divisible by 8, and rank between 1 and 64')
    random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(4)
    # Float32 keeps CPU/MPS training portable; larger CUDA workloads can use accelerate separately.
    options = {'variant': variant} if variant else {}
    pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype=torch.float32, **options).to(device)
    pipe.vae.requires_grad_(False); pipe.text_encoder.requires_grad_(False); pipe.unet.requires_grad_(False)
    pipe.unet.add_adapter(LoraConfig(r=rank, lora_alpha=rank, init_lora_weights='gaussian', target_modules=['to_k', 'to_q', 'to_v', 'to_out.0']))
    pipe.unet.train()
    parameters = [p for p in pipe.unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    scheduler = DDPMScheduler.from_pretrained(model, subfolder='scheduler')
    losses = []
    for step in range(steps):
        path, caption = random.choice(records)
        with Image.open(path) as image:
            image = ImageOps.fit(image.convert('RGB'), (resolution, resolution))
            pixels = torch.tensor(bytearray(image.tobytes()), dtype=torch.float32).reshape(resolution, resolution, 3).permute(2, 0, 1).unsqueeze(0).to(device) / 127.5 - 1
        with torch.no_grad():
            latents = pipe.vae.encode(pixels).latent_dist.sample() * pipe.vae.config.scaling_factor
            tokens = pipe.tokenizer(caption, max_length=pipe.tokenizer.model_max_length, padding='max_length', truncation=True, return_tensors='pt').input_ids.to(device)
            hidden = pipe.text_encoder(tokens)[0]
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (1,), device=device).long()
        noisy = scheduler.add_noise(latents, noise, timesteps)
        predicted = pipe.unet(noisy, timesteps, encoder_hidden_states=hidden).sample
        kind = scheduler.config.prediction_type
        if kind == 'epsilon': target = noise
        elif kind == 'v_prediction': target = scheduler.get_velocity(latents, noise, timesteps)
        else: raise ValueError('Unsupported noise scheduler prediction type: ' + kind)
        loss = F.mse_loss(predicted.float(), target.float())
        if not torch.isfinite(loss): raise ValueError('Training loss is nonfinite; check data and settings')
        loss.backward(); torch.nn.utils.clip_grad_norm_(parameters, 1.0); optimizer.step(); optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu())); print(f'Step {step+1}/{steps}: loss={losses[-1]:.6f}', flush=True)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    weights = convert_state_dict_to_diffusers(get_peft_model_state_dict(pipe.unet))
    StableDiffusionPipeline.save_lora_weights(str(output), unet_lora_layers=weights, safe_serialization=True)
    metrics = {'base_model': model, 'steps': steps, 'resolution': resolution, 'rank': rank, 'learning_rate': learning_rate, 'seed': seed, 'samples': len(records), 'losses': losses, 'variant': variant, 'trainable_parameters': sum(p.numel() for p in parameters)}
    (output / 'training.json').write_text(json.dumps(metrics, indent=2))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='stable-diffusion-v1-5/stable-diffusion-v1-5')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--variant', default=None, help='Optional model weight variant, e.g. fp16')
    parser.add_argument('--output', default='adapters/domain')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--rank', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['cpu','mps','cuda'], default='cpu')
    args = parser.parse_args()
    train(args.model, args.dataset, args.output, args.steps, args.resolution, args.rank, args.learning_rate, args.seed, args.device, args.variant)

if __name__ == '__main__': main()
