# Training Loop

## Per Training Step

1. Sample a batch of sentences from the dataset
2. Extract activation vectors from the frozen base model (layer X) for each sentence
3. AV generates X explanations per activation (GRPO group)
4. Paraphrase model generates Y paraphrases per explanation
5. AR scores each explanation and each paraphrase against the original activation vector

## AV Reward (GRPO)
For each explanation in the group:
- score = (cosine_sim(AR(explanation), original_activation) * high_weight)
         + (mean cosine_sim(AR(paraphrases), original_activation) * lower_weight)
         + (readability_score * low_weight)

Normalize scores within the group (GRPO advantage = (score - mean) / std)
Backpass AV using policy gradient scaled by advantage

## AR Training (Supervised, after every X AV steps)
For each explanation and its paraphrases:
- loss = MSE/cosine(AR(explanation), original_activation)
         + MSE/cosine(AR(paraphrase), original_activation)

Both original and paraphrase should reconstruct the same target activation vector.
Backpass AR directly against original activation.

## Collapse Detection
- Print explanations during training — should look like real semantic descriptions
- Cross-test: score explanation A against activation B — should be low
- Paraphrase score vs direct score gap — if direct is high but paraphrase is low, AV is using a private code
- Entropy of explanations — different activations should produce different explanations

## Key Design Decisions
- AV: GRPO (no ground truth explanation exists)
- AR: supervised loss directly against original activation vector
- Train AV for X steps, then train AR for Y steps, alternate
- Weight original explanation score higher than paraphrase scores in reward
- Readability reward weighted low — treat as regularizer not main signal
- AR trained on both original explanations and paraphrases for robustness
