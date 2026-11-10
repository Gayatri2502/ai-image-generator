# AI Image Generator

A local Gradio interface for prompt-based Stable Diffusion generation with an optional LoRA adapter. The model and adapter are loaded on the first request; outputs are generated on the local machine.

## Run

Use Python 3.10+, install `pip install -r requirements.txt`, and run `python app.py`. Open the local URL printed by Gradio. A compatible GPU is strongly recommended; the initial model download can be several gigabytes. Set `BASE_MODEL` to a compatible Stable Diffusion model ID or local directory. If a compatible LoRA has been trained, set `LORA_PATH` to its local directory or weight file before starting the app. Check the model license and any gated access requirements before downloading.

Example: `LORA_PATH=/path/to/adapter python app.py`

## Fine-tuning

No fine-tuned adapter or training dataset is included yet. A domain, properly licensed captioned training images, and GPU resources are needed. Use the official [Diffusers text-to-image LoRA training guide](https://huggingface.co/docs/diffusers/training/lora) to create an adapter compatible with the selected base model; point `LORA_PATH` at the resulting weights. QLoRA is not implemented. The application has not been tested with an actual downloaded model or adapter in this workspace.
