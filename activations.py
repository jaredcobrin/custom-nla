from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "Qwen/Qwen2.5-1.5B"

prompts = [
    "Hello",
    "What is the meaning of life?"
]

# put in sentences
class Activations:
    def __init__(self, model_id: str):
        self.model = AutoModelForCausalLM.from_pretrained(model_id)
        self.model.requires_grad_(False)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model.eval()
        self.batch_size = 8
        self.current_activations = []
        self.current_token_ids = None

    def extract_activations(self, layer: int, prompts: list[str]):   
        self.current_activations.clear()
          

        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0] # [batch_size, seq_len, 1536]
            # take out attention mask vector
            attention_mask = self.current_token_ids["attention_mask"].to(output.device) # [batch_size, seq_len]
            # find amount of tokens in each prompt
            batch_indexes = torch.arange(output.shape[0], device=output.device) # [0,1,2,3,4,5,6,7] size[batch_size]
            final_tokens = attention_mask.sum(dim=1) - 1 # [batch_size] # final_token_ids
            # make vector of activations
            activations_batch_vector = output[batch_indexes, final_tokens, :]  # [batch_size, 1536]
            # take out last 
            self.current_activations.append(activations_batch_vector.detach())

        handle = self.model.model.layers[layer].register_forward_hook(hook)
        
        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i: i + self.batch_size]
            self.current_token_ids = self.tokenizer(batch_prompts, return_tensors='pt', padding=True).to(self.model.device)
            with torch.inference_mode():
                outputs = self.model(**self.current_token_ids)
        
        handle.remove()
        
        final_activations = torch.cat(self.current_activations, dim=0)

        return final_activations
    

        
        



# take out activations

# take out activations at certain layer

# 
