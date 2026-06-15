# Custom NLA Project

## What this is
A from-scratch implementation of a Natural Language Autoencoder (NLA) trained with pure RL — no supervised examples. Based on the kitft paper concept but with a completely different training approach.

## Architecture
- **Base model** (frozen): Qwen/Qwen2.5-1.5B — source of activation vectors, never trained
- **AV model** (Activation Verbalizer): Qwen/Qwen2.5-1.5B + LoRA — takes an activation vector, injects it as a token embedding at a special placeholder token `<OVERHERE>`, generates a natural language explanation
- **AR model** (Activation Reconstructor): Qwen/Qwen2.5-1.5B + LoRA + linear value head — takes an explanation, strips lm_head and final layernorm, outputs a reconstructed activation vector from the last token's hidden state

## Training Loop (pure RL, no supervised labels)

### Per Training Step
1. Sample a batch of diverse user-style prompts (wide range of topics to produce diverse activations)
2. Extract activation vectors from frozen base model at a chosen layer
3. AV generates X explanations per activation (GRPO group)
4. Paraphrase model generates Y paraphrases per explanation
5. AR scores each explanation and paraphrase against the original activation

### AV Reward (GRPO)
- score = (cosine_sim(AR(explanation), activation) * high_weight)
         + (mean cosine_sim(AR(paraphrases), activation) * lower_weight)
         + (readability_score * low_weight)
- Normalize within group: advantage = (score - mean) / std
- Backpass AV using policy gradient scaled by advantage

### AR Training (Supervised, after every X AV steps)
- loss = MSE/cosine(AR(explanation), original_activation)
         + MSE/cosine(AR(paraphrase), original_activation)
- Both original and paraphrase should reconstruct the same target vector
- Train AR on both original explanations and paraphrases for robustness

### Alternating Training
- Train AV for X steps, then train AR for Y steps, alternate
- AR follows AV's lead — trains on whatever language AV is currently producing
- Ratio to be tuned empirically based on cosine similarity curves

## Collapse Detection
- Print explanations during training — should look like real semantic descriptions
- Cross-test: score explanation A against activation B — should be low
- Paraphrase score vs direct score gap — high direct + low paraphrase = private code collapse
- Entropy of explanations — different activations should produce visibly different explanations

## Files
- `activations.py` — base model activation extraction (Activations class). Done.
- `av.py` — AV model definition. Done, including fixed `extract_log_probs` (now uses `inputs_embeds` with activation injected for correct conditioning, concatenates prompt embeddings + generated embeddings for teacher-forcing).
- `ar.py` — AR model definition. Done.
- `training.py` — full end-to-end training loop draft. Mostly done but needs work before running (see below).
- `notes.md` — detailed training loop notes.
- `extractLogProbs.py` — empty, superseded by method in av.py.

## Current Status
Core model files (activations, AV, AR) are complete. Training loop is drafted end-to-end. Three things remain before first training run:

### 1. Prompt (AV injection prompt)
The prompt passed to AV's `av_forward_pass` needs to be designed well — it must contain `<OVERHERE>` and instruct the model to convert the injected activation into a natural language explanation. Currently hardcoded as a rough draft in training.py.

### 2. Logging, printing, and checkpointing
- Print explanations each step so collapse can be detected visually
- Print AV loss, AR loss, cosine similarity scores each step
- Save model checkpoints periodically (AV LoRA weights, AR LoRA weights + value head)
- Collapse detection checks (cross-test, paraphrase-gap, entropy)

### 3. Dataset
Currently just 2 hardcoded prompts. Need a real diverse dataset of user-style prompts covering a wide range of topics — diversity is critical so the base model produces varied activations and the AV learns to distinguish them.

## Claude's role
Educator and advisor only. Never write code. Explain concepts, give hints, point to docs, explain the *why* behind things. Always explain the why behind things, not just the what. Jared writes all the code himself.
