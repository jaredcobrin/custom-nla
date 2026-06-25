from activations import Activations
from av_model import AV
from ar import AR
import torch
import itertools
import checkpoints
import dataset


def setup():
    
    # LOAD DEVICE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

    # 1: EXTRACT ACTIVATIONS

    activation_model = Activations(model_id="Qwen/Qwen2.5-1.5B")
    activation_model.tokenizer.padding_side = "left"
    activation_model.model.to(device)


    # 1.1 token ids of paraphrase_prompt, semantic meaning prompt.
    #paraphrase_prompt_token_ids = activation_model.tokenizer(paraphrase_prompt, return_tensors="pt")
    #semantic_meaning_prompt_token_ids = activation_model.tokenizer(semantic_meaning_prompt, return_tensors="pt")

    # 2: Initialize AR & AV
    av = AV(model_id="Qwen/Qwen2.5-1.5B", 
            r_value=16, 
            target_module=["q_proj", "k_proj", "o_proj", "v_proj"], 
            lora_alpha=32)
    av.model.to(device)

    ar = AR(model_id="Qwen/Qwen2.5-1.5B", 
            r_value=16,  
            target_module=["q_proj", "k_proj", "o_proj", "v_proj"], 
            lora_alpha=32)
    ar.model.to(device)
    ar.value_head.to(device)


    # 2.1 CREATE OPTIMIZERS
    av_optimizer = torch.optim.AdamW(av.model.parameters(), lr=1e-4)
    ar_parameters = list(itertools.chain(ar.model.parameters(), ar.value_head.parameters()))
    ar_optimizer = torch.optim.AdamW(ar_parameters, lr=1e-4)
    
    # 2.2 Get Dataset:
    buffer_dataset = dataset.get_dataset_buffer()
    
    
    return av, ar, activation_model, av_optimizer, ar_optimizer, ar_parameters, buffer_dataset



    # pt2: TRAINING LOOP: Forward & Backpass
def train(buffer_dataset, paraphrase_prompt: str, av_prompt: str, semantic_meaning_prompt: str, activation_model, av, ar, av_optimizer, ar_optimizer, total_steps, ar_parameters, batch_size, GRPO_size):    
    
    # LOAD DEVICE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    # 0: load checkpoint: 
    start_step = checkpoints.load_checkpoints(av, ar, av_optimizer, ar_optimizer)
    try:
        for i in range(start_step, total_steps):

            print(f"--------------------STEP : {i} ----------------------------------")

            # 0.0 random sample
            base_model_inputs = dataset.sample_dataset(buffer_dataset, batch_size, min_window_size=20, max_window_size=200)
            # 0: EXTRACT ACTIVATIONS
            activations_ai_prompts = activation_model.extract_activations(layer=20, prompts=base_model_inputs)
            # 1: av original explanation/GRPO expl, 1.5: extract log_probs
            GRPO_expl, GRPO_log_probs, out_expl = av.av_forward_pass(base_activations=activations_ai_prompts, 
                                                prompt=av_prompt, 
                                                batch_size=batch_size,  
                                                GRPO_size=GRPO_size, 
                                                temperature=1.0, 
                                                do_sample=True)
            if (i % 10 == 0):
                print(f"GRPO_Explanation 0: {GRPO_expl[0]}")
                print(f"GRPO_Explanation 1: {GRPO_expl[1]}")

            # 2: make paraphrased explanations -> loop through with strings instead of cat with token_ids, to avoid EOS padding
            
            paraphrase_prompt_expl = []
            for expl in GRPO_expl:
                paraphrase_prompt_expl.append(paraphrase_prompt + expl + "\nParaphrase: ")
            paraphrase_prompt_token_ids = activation_model.tokenizer(paraphrase_prompt_expl, return_tensors="pt", padding=True).to(device)
            paraphrase_expl_tokens = activation_model.model.generate(input_ids=paraphrase_prompt_token_ids["input_ids"], attention_mask=paraphrase_prompt_token_ids["attention_mask"], max_new_tokens=150, num_return_sequences=1, do_sample=True, temperature=0.7)
            paraphrase_expl_only_tokens = paraphrase_expl_tokens[:, len(paraphrase_prompt_token_ids["input_ids"][0]):]
            paraphrase_expl = activation_model.tokenizer.batch_decode(paraphrase_expl_only_tokens, skip_special_tokens=True)
            if (i % 10 == 0):
                print(f"Paraphrase_Explanation: {paraphrase_expl[0]}")
                print(f"Paraphrase_Explanation: {paraphrase_expl[1]}")
            # 3: run though ar 
            
            GRPO_ar_activations = ar.forward_pass(explanations=GRPO_expl, batch_size=batch_size, GRPO_size=GRPO_size)
            paraphrase_ar_activations = ar.forward_pass(explanations=paraphrase_expl, batch_size=batch_size, GRPO_size=GRPO_size)

            # 4: compute reward functions
            GRPO_ar_activations = GRPO_ar_activations.reshape(batch_size, GRPO_size, 1536)
            paraphrase_ar_activations = paraphrase_ar_activations.reshape(batch_size, GRPO_size, 1536)


            # 4.1 implement cosine similarity of GRPO
            cosine_sim_GRPO = torch.nn.functional.cosine_similarity(GRPO_ar_activations.detach(), activations_ai_prompts.unsqueeze(1), dim=-1)
            print(f"cosine_sim_GRPO: {cosine_sim_GRPO}")
            # 4.2 implement cosine similarity of paraphrase
            cosine_sim_paraphrase = torch.nn.functional.cosine_similarity(paraphrase_ar_activations.detach(), activations_ai_prompts.unsqueeze(1), dim=-1)
            print(f"cosine_sim_paraphrase: {cosine_sim_paraphrase}")

            # 4.3 implement semantic meaning
            semantic_meaning_prompt_expl = []
            index_tensor = torch.arange(batch_size*GRPO_size).to(device)
            for expl in GRPO_expl:
                semantic_meaning_prompt_expl.append(expl + semantic_meaning_prompt + " Rating:")
            semantic_meaning_prompt_token_ids = activation_model.tokenizer(semantic_meaning_prompt_expl, return_tensors="pt", padding=True).to(device)
            sum_mask = semantic_meaning_prompt_token_ids["attention_mask"].sum(dim=1) - 1
            logits = activation_model.model(input_ids = semantic_meaning_prompt_token_ids["input_ids"], attention_mask = semantic_meaning_prompt_token_ids["attention_mask"]).logits[index_tensor, sum_mask, :]
            onefive = torch.tensor([[1], [2], [3], [4], [5]], dtype=torch.bfloat16).to(device)
            one_to_five = activation_model.tokenizer.convert_tokens_to_ids(["1", "2", "3", "4", "5"])
            one_to_five_logits = logits[:, one_to_five]
            softmax_one_to_five = torch.softmax(one_to_five_logits, dim=-1)
            weighted_sum = softmax_one_to_five @ onefive
            semantic_meaning_scores = weighted_sum.squeeze(dim=-1)
            semantic_meaning_scores = semantic_meaning_scores.reshape(batch_size, GRPO_size)
                
            # 4.4 calculate final reward
                
            # 4.4.1 normalizing GRPO, paraphrasing, semantic meaning
                
            # normalizing  GRPO
            GRPO_mean = torch.mean(cosine_sim_GRPO, dim=-1)
            GRPO_std = torch.std(cosine_sim_GRPO, dim=-1) + 1e-8
            print(f"GRPO_std: {GRPO_std}")
            GRPO_mean = GRPO_mean.unsqueeze(-1)
            GRPO_std = GRPO_std.unsqueeze(-1)
            normalized_GRPO_scores = (cosine_sim_GRPO - GRPO_mean) / GRPO_std
            print(f"normalized_GRPO_scores: {normalized_GRPO_scores}")

            # normalizing paraphrasing
            paraphrase_mean = torch.mean(cosine_sim_paraphrase, dim=-1)
            paraphrase_std = torch.std(cosine_sim_paraphrase, dim=-1) + 1e-8
            print(f"paraphrase_std: {paraphrase_std}")
            paraphrase_mean = paraphrase_mean.unsqueeze(-1)
            paraphrase_std = paraphrase_std.unsqueeze(-1)
            normalized_paraphrase_scores = (cosine_sim_paraphrase - paraphrase_mean) / paraphrase_std
            print(f"normalized_paraphrase_scores: {normalized_paraphrase_scores}")
            
            # normalizing semantic meaning
            semantic_meaning_mean = torch.mean(semantic_meaning_scores, dim=-1)
            semantic_meaning_std = torch.std(semantic_meaning_scores, dim=-1) + 1e-8
            print(f"semantic_meaning_std: {semantic_meaning_std}")
            semantic_meaning_mean = semantic_meaning_mean.unsqueeze(-1)
            semantic_meaning_std = semantic_meaning_std.unsqueeze(-1)
            normalized_semantic_meaning_scores = (semantic_meaning_scores - semantic_meaning_mean) / semantic_meaning_std
            print(f"normalized_semantic_meaning_scores: {normalized_semantic_meaning_scores}")

            # 4.4.2 scale GRPO, paraphrasing, semantic meaning
            scaled_GRPO_scores = normalized_GRPO_scores * 0.5
            scaled_paraphrase_scores = normalized_paraphrase_scores * 0.3
            scaled_semantic_meaning_scores = normalized_semantic_meaning_scores * 0.2

            # 4.4.3 Reward Score
            rewards = scaled_GRPO_scores + scaled_paraphrase_scores + scaled_semantic_meaning_scores

            # 4.4.4 Normalized Reward Score
            rewards_mean = torch.mean(rewards, dim=-1)
            rewards_std = torch.std(rewards, dim=-1) + 1e-8
            print(f"rewards_std: {rewards_std}")
            rewards_mean = rewards_mean.unsqueeze(-1)
            rewards_std = rewards_std.unsqueeze(-1)
            normalized_rewards = (rewards - rewards_mean) / rewards_std
            print(f"normalized_rewards: {normalized_rewards}")



            reshaped_normalized_rewards = normalized_rewards.reshape(GRPO_size*batch_size)
            # 4.4.2 scale GRPO, paraphrasing, semantic meaning
            # 4.5 calculate loss with log_probs
            av_loss = -(GRPO_log_probs * reshaped_normalized_rewards).mean()
            print(f"AV_loss: {av_loss.item()}")
            print(f"GRPO_mean: {GRPO_mean.mean().item()}")
            print(f"Paraphrase_mean: {paraphrase_mean.mean().item()}")
            print(f" GRPO_mean - paraphrase_mean: {(GRPO_mean - paraphrase_mean).mean().item()}")
            print(f"Semantic_meaning_mean: {semantic_meaning_mean.mean().item()}")
            
            # 5: backprop on AV & AR

            # 5.1 backprop on AV
            av_optimizer.zero_grad()
            av_loss.backward()
            av_grad_norm = torch.nn.utils.clip_grad_norm_(av.model.parameters(), max_norm=5.0)
            print(f"AV Gradient_norm: {av_grad_norm.item()}")
            av_optimizer.step()

            # 5.2 backprop on AR

            # 5.2.1 RECALCULATION OF SIMILARITIES (DUE TO .detach() on ar stuff for av)
            cosine_sim_GRPO = torch.nn.functional.cosine_similarity(GRPO_ar_activations, activations_ai_prompts.unsqueeze(1), dim=-1)
            cosine_sim_paraphrase = torch.nn.functional.cosine_similarity(paraphrase_ar_activations, activations_ai_prompts.unsqueeze(1), dim=-1)

            # 5.2.2 calculate loss, using top half best normalized scores
            # from top best normalized scores:
            # loss is calculated from these scores respected original and paraphrased explanations.
            # loss is a sum of all these original and paraphrased explanations.
            top_k_normalized_rewards = torch.topk(normalized_rewards, k=(GRPO_size//2), dim=1).indices
            top_k_cos_GRPO = torch.gather(cosine_sim_GRPO, dim=1, index=top_k_normalized_rewards)
            top_k_cos_paraphase = torch.gather(cosine_sim_paraphrase, dim=1, index=top_k_normalized_rewards)
            mean_cos_sim = torch.cat([top_k_cos_GRPO, top_k_cos_paraphase], dim=0).mean()



            # 5.2.3 Cos-Sim difference.
            top_k_GRPO = torch.gather(GRPO_ar_activations, dim=1, index=top_k_normalized_rewards.unsqueeze(-1).expand(-1, -1, 1536))
            top_k_paraphase = torch.gather(paraphrase_ar_activations, dim=1, index=top_k_normalized_rewards.unsqueeze(-1).expand(-1, -1, 1536))
            top_k_GRPO_paraphase = torch.cat([top_k_GRPO, top_k_paraphase], dim=1)

            # Compute mean of activations
            top_k_mean_GRPO_paraphase = torch.mean(top_k_GRPO_paraphase, dim=1)
            
            # Compute distances of cos sim
            cos_sim_distance = torch.nn.functional.cosine_similarity(top_k_mean_GRPO_paraphase.unsqueeze(0), 
                                                                     top_k_mean_GRPO_paraphase.unsqueeze(1), dim=-1)
            cos_sim_distance_original = torch.nn.functional.cosine_similarity(activations_ai_prompts.unsqueeze(0), 
                                                                     activations_ai_prompts.unsqueeze(1), dim=-1)

            cos_sim_dist = cos_sim_distance * (1 - torch.eye(4).to(device))
            cos_sim_dist_orig = cos_sim_distance_original * (1 - torch.eye(4).to(device))
            penalty = ((cos_sim_dist - cos_sim_dist_orig)**2).sum()

            scale = (0.75 * mean_cos_sim.detach()) / (penalty.detach() + 1e-8)
            ar_loss = (1 - mean_cos_sim) + (scale * penalty)
            print(f"AR_loss: {ar_loss.item()}")
            # 5.2.2 backprop 
            ar_optimizer.zero_grad()
            ar_loss.backward()
            ar_grad_norm = torch.nn.utils.clip_grad_norm_(ar_parameters, max_norm=5.0)
            print(f"AR Gradient_norm: {ar_grad_norm.item()}")
            ar_optimizer.step()


            if i%100 == 0:
                checkpoints.save_checkpoints(i, av, ar, av_optimizer, ar_optimizer, 10)
    except KeyboardInterrupt:
        checkpoints.save_checkpoints(i, av, ar, av_optimizer, ar_optimizer, 10)



        #indexed_normal = GRPO_ar_activations[top_k_normalized_rewards]
        #indexed_paraphrase = paraphrase_ar_activations[top_k_normalized_rewards]
        #combined_activations = torch.cat(indexed_normal, indexed_paraphrase, dim=0)






