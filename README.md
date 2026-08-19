# Custom NLA

A from-scratch Natural Language Autoencoder, trained with pure RL — no
supervised examples, no externally-generated labels.

## Architecture

- **Base model** (frozen) — `Qwen2.5-1.5B`. Source of activation vectors, never
  trained.
- **AV, Activation Verbalizer** — `Qwen2.5-1.5B` + LoRA. Takes an activation
  vector, injects it at a placeholder token (`<OVERHERE>`), generates a
  natural-language explanation of it.
- **AR, Activation Reconstructor** — `Qwen2.5-1.5B` + LoRA + a linear value
  head. Takes an explanation, outputs a reconstructed activation vector from
  the last token's hidden state.

AV and AR train alternately: AV by GRPO, rewarded by how well AR can
reconstruct the activation from its explanation (plus a paraphrase-robustness
term); AR by supervised cosine/MSE loss against the real activation.

## Files

| file | |
|---|---|
| `activations.py` | extracts activations from the frozen base model |
| `av_model.py` | the AV — injection, generation, log-prob extraction for GRPO |
| `ar.py` | the AR — stripped LM head, value head |
| `training.py` | the full training loop |
| `curriculum.py` | the visible-fraction curriculum — how much of the source text the AV can see, ladders down from 100% to 0% as reconstruction improves |
| `checkpoints.py` | save/load both models' LoRA weights and optimizers |
| `metrics.py` | FVE (fraction of variance explained) computation |
| `dataset.py` | streams FineWeb for training prompts |
| `run.py` | entry point |
| `prompts/` | prompt templates — AV, paraphrasing, semantic scoring |
| `curriculum_plan.md` | the current training-curriculum design |
| `notes.md` | earlier design notes |

## Running it

```bash
pip install -r requirements.txt
python run.py
```

## Status

Training runs end-to-end on a GPU pod, checkpointing every 100 steps. Currently
in a bootstrapping deadlock: the AR needs good AV explanations to learn
accurate reconstruction, and the AV needs accurate AR reconstruction to get a
useful reward signal, so neither improves much without the other already
being good.
