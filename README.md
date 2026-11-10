# AI Image Studio

A local application for prompt-based Stable Diffusion generation and captioned-image LoRA training. The browser interface includes generation settings, a bounded job queue, status/error feedback, a persistent gallery, and PNG downloads. The training CLI saves a portable safetensors adapter and a JSON training record.

## Start

Use Python **3.10+** (tested with 3.12):

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python app.py
```

Open **http://127.0.0.1:8003/**. Alternatively, `./run.sh` installs dependencies into `.venv` and starts the application; use `PYTHON_BIN=/path/to/python3.12 ./run.sh` on machines with an older system Python. API docs are available at `/docs`.

The default model is `stable-diffusion-v1-5/stable-diffusion-v1-5`. The first generation downloads several gigabytes of weights; a compatible GPU is recommended. This workspace has CPU inference available, so generation is slower. Images and settings are saved under `generated/`. Generated images stay on this machine. Hugging Face receives requests for model files during setup/loading. Review the base model's license and usage conditions.

Run one server process/worker: the job queue is in memory, while completed images and metadata persist on disk. Pending jobs are lost on restart. The app binds to localhost and is intended for one local user. Configure authentication and per-user storage before deploying publicly.

## Configuration

- `BASE_MODEL`: compatible Stable Diffusion model ID or local Diffusers directory.
- `MODEL_VARIANT`: optional weight variant such as `fp16` for the default model. CPU/MPS inference still casts to float32 for compatibility.
- `DEVICE`: `auto`, `cpu`, `mps` or `cuda`. Auto detects CUDA, then Metal, then CPU.
- `LORA_PATH`: local adapter directory or `.safetensors` file matching the base model. Restart after changing it.
- `IMAGE_OUTPUT_DIR`: gallery directory (default `generated`).
- `HF_HOME`: optional Hugging Face cache location.

## Train a domain LoRA

Prepare a directory of training images with a UTF-8 `metadata.jsonl` file. Each line contains an image filename and its caption:

```json
{"file_name":"product-001.png","text":"a photo of a ceramic teapot in studio light"}
```

Use licensed images representing the intended domain and captions describing them accurately. Then run:

```sh
python -m image_generator.train --dataset training-data \
  --output adapters/domain --variant fp16 --steps 500 --resolution 512 --rank 4 --device cuda
```

The command freezes the VAE, text encoder and base UNet; trains attention LoRA weights using a diffusion noise-prediction objective; clips gradients; and saves `pytorch_lora_weights.safetensors` plus `training.json` (losses, seed, model and settings). It uses float32, batch size one, and a fixed learning rate. Lower the resolution or use a larger GPU if memory is insufficient. Choose the number of steps based on held-out image quality; training loss alone is not a quality metric. This is LoRA training; QLoRA is not implemented.

To use the result, set `LORA_PATH=adapters/domain` in `.env` and restart. Keep `BASE_MODEL` consistent with the training model. The UI reports when an adapter is configured.

## Small end-to-end training check

A small official test model and original geometric images verify training/export/loading without downloading the full artwork model:

```sh
python -m image_generator.sample_data training-data/circles
python -m image_generator.train \
  --model hf-internal-testing/tiny-stable-diffusion-torch \
  --dataset training-data/circles --output adapters/smoke \
  --steps 3 --resolution 32 --rank 2
BASE_MODEL=hf-internal-testing/tiny-stable-diffusion-torch \
  MODEL_VARIANT= LORA_PATH=adapters/smoke python app.py
```

The tiny model produces noise and exists solely to verify integration. The UI labels it accordingly. It cannot establish useful domain adaptation. A meaningful domain-specific model requires an appropriate dataset, a longer training run and held-out evaluation.

## Tests

```sh
pip install -r requirements-dev.txt
python -m pytest -q
```

Offline tests cover job completion, PNG download, persistent history, validation, generation failures, and dataset path validation. CI runs these without model downloads. Set `RUN_MODEL_TESTS=1` to include the real tiny-model training/export/loading test. The small-model integration was also exercised locally through a three-step training run, adapter export/reload, and image generation. The full-model browser generation check is recorded in the workspace's `RUNNING.md`.

Reference implementations: [Diffusers LoRA training](https://huggingface.co/docs/diffusers/training/lora), [LoRA loading](https://huggingface.co/docs/diffusers/api/loaders/lora), and the [SD1.5 model card](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5).
