---
tags:
  - robotics
  - smolvla
  - lerobot
  - imitation-learning
  - isaac-sim
  - leisaac
  - so101
license: apache-2.0
library_name: lerobot
pipeline_tag: robotics
datasets:
  - LightwheelAI/leisaac-pick-orange
base_model: lerobot/smolvla_base
---

# SmolVLA SO101 PickOrange

Fine-tuned [SmolVLA](https://huggingface.co/lerobot/smolvla_base) policy for the SO101 robot arm performing an orange-picking task in [LeIsaac](https://github.com/LightwheelAI/leisaac) (Isaac Sim).

## Task

Pick three oranges from the table and place them on the plate, then reset the arm to rest state. Evaluated in the `LeIsaac-SO101-PickOrange-v0` Isaac Sim environment.

## Architecture

SmolVLA is a Vision-Language-Action model that combines a frozen vision encoder with a language model backbone and a lightweight action expert head.

### Inference pipeline

```
2 camera images (480x640)
  -> resize to 512x512 with padding
  -> patchify into 16x16 patches (1024 tokens per image, 2048 total)
  -> 12-layer ViT vision encoder (bf16)
  -> project to LM hidden space
  -> 16-layer SmolLM2 backbone + 16-layer Expert (interleaved cross-attention)
  -> decode 50 action tokens -> 6D joint positions
```

### Vision Encoder (ViT)

| Property | Value |
|----------|-------|
| Architecture | Vision Transformer (SigLIP-derived) |
| Hidden size | 768 |
| Layers | 12 |
| Attention heads | 12 |
| Patch size | 16x16 |
| Input resolution | 512x512 |
| Tokens per image | 1024 (32x32 patches) |
| Precision | bfloat16 |
| Status (training) | Frozen |

### Text/LM Backbone (SmolLM2)

| Property | Value |
|----------|-------|
| Architecture | SmolLM2 (Llama-based) |
| Hidden size | 960 |
| Layers | 16 (truncated from 32) |
| Attention heads | 15 |
| Intermediate size | 2560 |
| Vocab size | 49,280 |

### Action Expert Head

| Property | Value |
|----------|-------|
| Layers | 16 (matches truncated VLM) |
| Hidden size | 720 (0.75x VLM hidden) |
| Attention mode | Cross-attention (interleaved with VLM layers) |
| Output | 50 action chunks x 6D |
| Trainable params | 100M |

### Full Model Summary

| Component | Params | Trainable | Precision |
|-----------|:------:|:---------:|:---------:|
| Vision encoder (ViT) | ~86M | Frozen | bf16 |
| LM backbone (SmolLM2) | ~264M | Frozen | bf16 |
| Action expert head | ~100M | Yes | bf16 |
| **Total** | **450M** | **100M** | **bf16** |

## Branches

| Branch | Training | Batch size | Final loss |
|--------|----------|:-:|:-:|
| `main` | multi-rank, 30k steps | 56 | 0.019 |
| `single-rank` | single-rank, 30k steps | 64 | 0.008 |

## Training

| Parameter | Value |
|-----------|-------|
| Dataset | [LightwheelAI/leisaac-pick-orange](https://huggingface.co/datasets/LightwheelAI/leisaac-pick-orange) (sim-collected) |
| Episodes | 60 |
| Frames | 36,293 |
| Steps | 30,000 |
| Batch size | 56 effective (main) / 64 (single-rank) |
| Learning rate | 1e-4 (cosine decay with 1k warmup) |
| Optimizer | AdamW (betas=0.9/0.95, eps=1e-8, weight_decay=1e-10) |
| Scheduler | Cosine decay, 1000 warmup steps, decay to 2.5e-6 |
| Grad clip norm | 10 |
| VLM layers | 16 (truncated from 32) |
| Vision encoder | Frozen |
| Training mode | Expert-only (train_expert_only=true) |
| Framework | [LeRobot](https://github.com/huggingface/lerobot) v0.4.5 |

## Usage

### Serve the policy

```bash
python -m lerobot.scripts.serve \
  --policy.type=smolvla \
  --policy.pretrained_path=edge-inference/smolvla-so101-pick-orange \
  --policy.vlm_model_name=HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  --port=8080
```

### Evaluate in Isaac Sim (requires LeIsaac + Isaac Sim)

```bash
python scripts/evaluation/policy_inference.py \
  --task=LeIsaac-SO101-PickOrange-v0 \
  --policy_type=lerobot-smolvla \
  --policy_host=localhost \
  --policy_port=8080 \
  --policy_checkpoint_path=edge-inference/smolvla-so101-pick-orange \
  --policy_action_horizon=50 \
  --policy_language_instruction="Pick up the orange and place it on the plate" \
  --eval_rounds=10 \
  --device=cuda \
  --enable_cameras
```

### Use the single-rank branch (lower loss)

```python
from huggingface_hub import snapshot_download
snapshot_download(
    "edge-inference/smolvla-so101-pick-orange",
    revision="single-rank",
    local_dir="./checkpoint-single-rank"
)
```

## Dataset

The training data was collected via teleoperation inside the LeIsaac simulation (Isaac Sim), meaning there is zero visual domain gap between training and evaluation environments.

## Files

- `model.safetensors` -- Model weights (1.2 GB)
- `config.json` -- Policy architecture config
- `train_config.json` -- Full training configuration (reproducible)
- `policy_preprocessor*.json/safetensors` -- Input normalization (state mean/std)
- `policy_postprocessor*.json/safetensors` -- Output denormalization (action mean/std)
