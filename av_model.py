from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from peft import LoraConfig, get_peft_model
import activations

model_id = "Qwen/Qwen2.5-1.5B"
target_modules=["q_proj", "k_proj", "o_proj", "v_proj"]
class AV:

    def __init__(self, model_id: str, r_value: int, target_module: list[str], lora_alpha: int):
        self.model_id = model_id
        self.r_value = r_value
        self.target_module = target_module
        self.lora_alpha = lora_alpha
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.padding_side = "left"
        self.special_tokens = {
            "additional_special_tokens": ["<OVERHERE>"]
            }
        self.tokenizer.add_special_tokens(self.special_tokens)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=torch.bfloat16)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.config = LoraConfig(
            r=self.r_value,
            target_modules=self.target_module,
            lora_alpha=self.lora_alpha
        )
        self.model = get_peft_model(self.model, self.config)
        self.model.print_trainable_parameters()
        self.special_token_id = self.tokenizer.convert_tokens_to_ids("<OVERHERE>")
        
    def av_forward_pass(self, base_activations, prompts: list[str], batch_size: int, GRPO_size: int, temperature: float, do_sample: bool): # GRPO must be multiple of batch_size
        # LOAD DEVICE
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # prompts: one string per batch row (each with its own curriculum-visible
        # context filled in), so tokenize as a batch with left-padding.
        prompt_token_ids = self.tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        prompt_token_embeddings = self.model.get_input_embeddings()(prompt_token_ids["input_ids"])

        # special token position varies per row now (context length differs), so
        # find each row's column index rather than assuming one shared position.
        special_token_positions = (prompt_token_ids["input_ids"] == self.special_token_id).nonzero()
        row_indices = special_token_positions[:, 0]
        col_indices = special_token_positions[:, 1]

        # replace token placeholder token with base_activations
        scaled_activations = base_activations*(self.target_norm()/self.activation_norm(base_activations))
        prompt_token_embeddings[row_indices, col_indices] = scaled_activations.to(prompt_token_embeddings.dtype)
        # run model # might need to repeat(8,1,1)
        expl_output = self.model.generate(inputs_embeds=prompt_token_embeddings, attention_mask=prompt_token_ids["attention_mask"], num_return_sequences=GRPO_size, do_sample=do_sample, temperature=temperature, max_new_tokens=150)
        print(expl_output.shape)
        #out_expl = expl_output[:, len(prompt_token_ids["input_ids"][0]):]

        batch_explanation = self.tokenizer.batch_decode(expl_output, skip_special_tokens=True)
        log_probs = self.extract_log_probs(expl_output, prompt_token_ids, prompt_token_embeddings, GRPO_size)
        explanations = list(batch_explanation)
        return explanations, log_probs, expl_output

    def activation_norm(self, base_activations: list[float]):
        normed_base_activations = base_activations.norm(dim=-1)
        normed_base_activations = normed_base_activations.unsqueeze(dim=-1)
        return normed_base_activations
    def target_norm(self):
        return self.model.get_input_embeddings().weight.norm(dim=1).mean()
    
    def extract_log_probs(self, expl_output, prompt_token_ids, prompt_token_embeddings, GRPO_size):
        generated_embeds = self.model.get_input_embeddings()(expl_output)
        prompt_and_expl_embeds = torch.cat([prompt_token_embeddings.repeat_interleave(GRPO_size, dim=0), generated_embeds], dim=1)
        # mask out generated-side padding (sequences that hit EOS early) and
        # carry the prompt's left-padding mask, since prompts vary in length now
        generated_mask = (expl_output != self.tokenizer.pad_token_id).long()
        attention_mask = torch.cat([prompt_token_ids["attention_mask"].repeat_interleave(GRPO_size, dim=0), generated_mask], dim=1)
        logits = self.model(inputs_embeds=prompt_and_expl_embeds, attention_mask=attention_mask).logits
        soft_logits = torch.nn.functional.log_softmax(logits, dim=-1)
        index = torch.unsqueeze(expl_output, dim=-1)
        prompt_len = prompt_token_ids["input_ids"].shape[1]
        log_probs = torch.gather(soft_logits[:, prompt_len-1:-1, :], dim=-1, index = index)
        log_probs = torch.squeeze(log_probs, dim=-1)
        mask = generated_mask.to(log_probs.dtype)
        mean_log_probs = (log_probs * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
        return mean_log_probs