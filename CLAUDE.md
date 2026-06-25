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
- `activations.py` — base model activation extraction (Activations class). Done. Uses AutoModelForCausalLM (not AutoModel) so the same model handles activation extraction AND paraphrase generation AND semantic scoring.
- `av_model.py` — AV model definition. Done. extract_log_probs uses inputs_embeds with activation injected for correct conditioning, concatenates prompt embeddings + generated embeddings for teacher-forcing.
- `ar.py` — AR model definition. Done. lm_head and final norm stripped (at correct PEFT depth: self.model.model.lm_head, self.model.model.model.norm), value_head in bfloat16.
- `training.py` — full end-to-end training loop. Running on GPU.
- `checkpoints.py` — saves/loads AV LoRA, AR LoRA, AR value head, both optimizers. Keeps last N checkpoints.
- `run.py` — entry point. batch_size=4, GRPO_size=4, total_steps=5000.
- `prompts/` — av_prompt.py, paraphrase_prompt.py, semantic_meaning_prompt.py, ai_questions.py (73 diverse prompts).

## Current Status
Training is running end-to-end on RunPod/Lambda A100. Core bugs are fixed. Currently investigating and addressing training instability.

### What's working
- Full training loop runs without crashes
- AR loss decreasing steadily (1.0 → ~0.22 over ~75 steps)
- GRPO_mean (cosine similarity) increasing steadily
- Checkpointing every 100 steps
- All models loaded in bfloat16 to fit in GPU memory

### Known training dynamics issue: AR mode collapse
AR learns to output roughly the same vector for all inputs (the "average activation direction") because all 73 training prompts produce similar activation clusters. This causes GRPO_std → 0 (all 4 GRPO explanations score identically), which kills AV's reward signal and causes AV gradient norms to explode (1000-8000+).

### Fix implemented: cross-activation discrimination penalty
Added to AR loss: penalizes AR when pairwise cosine similarities between its outputs don't match pairwise cosine similarities between original activations. Forces AR to produce different outputs for different activations rather than collapsing to one mean vector.
- Uses top_k GRPO + paraphrase reconstructions per activation as representatives
- Compares [batch_size, batch_size] pairwise similarity matrix of AR outputs vs original activations
- Penalty scaled dynamically to 75% of mean_cos_sim magnitude
- Added to existing reconstruction loss (1 - mean_cos_sim)

### Next steps to investigate
- Whether discrimination penalty fixes the GRPO_std collapse
- Whether dataset diversity needs to be increased (replace 73 hand-crafted prompts with lmsys/lmsys-chat-1m or allenai/WildChat — real user conversations with natural diversity including typos, nonsensical prompts, statements, etc.)
- Monitor GRPO_std across training — if it stays above near-zero, collapse is fixed

### Key diagnostics to watch
- `GRPO_std`: should stay meaningfully above 1e-8 — if it collapses, AR is still outputting same vector
- `AV Gradient_norm`: should stay in hundreds range, not thousands/tens-of-thousands
- `AR_loss`: should decrease steadily without oscillating
- `GRPO_mean` / `Paraphrase_mean`: should increase steadily toward ~1.0
- Printed explanations: different activations should produce visibly different explanations

## Claude's role
Educator and advisor only. Never write code. Explain concepts, give hints, point to docs, explain the *why* behind things. Always explain the why behind things, not just the what. Jared writes all the code himself.
