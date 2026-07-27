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
Training runs end-to-end on RunPod/Lambda A100. Core bugs fixed. Currently stuck in a bootstrapping deadlock — training is not progressing meaningfully.

### What's working
- Full training loop runs without crashes
- Checkpointing every 100 steps
- All models loaded in bfloat16 to fit in GPU memory
- GRPO_std is non-zero (discrimination penalty working — AR outputs different vectors for different activations)
- AV gradient norms stable (0.4-0.7 range) after fixing `.sum` → `.mean` in `extract_log_probs`

### Fixed bugs
- **`.sum` → `.mean` in `av_model.py` `extract_log_probs`**: `.sum(dim=-1)` over 150 tokens caused artificially large loss magnitudes and exploding AV gradient norms (900-8950). Changed to `.mean(dim=-1)` to normalize per token. Gradient norms dropped to stable range immediately.

### Cross-activation discrimination penalty (implemented, working partially)
Added to AR loss: penalizes AR when pairwise cosine similarities between its outputs don't match pairwise cosine similarities between original activations. Prevents AR from collapsing to outputting the same vector for all inputs.
- Uses top_k GRPO + paraphrase reconstructions per activation as representatives
- Compares [batch_size, batch_size] pairwise similarity matrix of AR outputs vs original activations
- Penalty scaled dynamically to 75% of mean_cos_sim magnitude
- Result: GRPO_std non-zero, but GRPO_mean still stuck at 0.20-0.24 after 300 steps

### Core unresolved problem: bootstrapping deadlock
**The fundamental issue:** AR needs good AV explanations to learn accurate reconstruction. AV needs accurate AR reconstruction to get a useful reward signal. Neither can improve without the other being already good.

Concretely at step ~300:
- AR_loss stuck at 0.93-0.95 — AR is barely reconstructing anything useful
- GRPO_mean stuck at 0.20-0.24 — AV cosine similarities not climbing
- AV gradient norms 0.4-0.7 — model barely updating its weights

**Why GRPO_std being non-zero doesn't mean the reward is informative:** GRPO_std is non-zero simply because AV uses stochastic sampling (do_sample=True), so the 4 outputs are never identical tokens. Different tokens → different AR hidden states → different cosine similarities. But the *differences* don't reliably reflect explanation quality — AR isn't good enough to consistently score better explanations higher. So AV gets noise dressed up as signal and can't learn which directions to improve.

### Reward hacking episode (steps ~260-280, now resolved)
AV discovered that echoing the av_prompt wording scored well on semantic meaning scorer (word overlap). Got reinforced, all 4 GRPO samples produced similar prompt-echoing text, GRPO_std collapsed, gradients fell to near-zero. Appeared to recover by step 290-300 with coherent explanations returning.

### Proposed solution: AR warm-start
Pretrain AR on (C4 text → base model activation) pairs before RL begins. This gives AR baseline reconstruction ability so its cosine similarity scores are informative from the start of RL — breaking the bootstrapping deadlock.
- C4 dataset already used for training prompts (allenai/c4, streaming, split="train")
- Warm-start would be supervised: AR sees real text, produces reconstruction vector, loss = cosine distance to actual activation that text produced in the base model
- After warm-start, AR can give AV a meaningful learning signal from step 1 of RL

### Key diagnostics to watch
- `GRPO_std`: should stay meaningfully above 1e-8 — if it collapses, AR is outputting same vector for everything
- `AV Gradient_norm`: should be in 1-50 range — below 0.5 means barely learning, above 100 means instability
- `AR_loss`: should decrease toward 0 — stuck at 0.93-0.95 is a sign of bootstrapping deadlock
- `GRPO_mean` / `Paraphrase_mean`: should increase steadily toward ~1.0 — flat means no learning
- Printed explanations: different activations should produce visibly different explanations, not generic "internal state" boilerplate

## Claude's role
Educator and advisor only. Never write code. Explain concepts, give hints, point to docs, explain the *why* behind things. Always explain the why behind things, not just the what. Jared writes all the code himself.
