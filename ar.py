from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from peft import LoraConfig, get_peft_model
import activations

model_id = "Qwen/Qwen2.5-1.5B"

class AR:

    def __init__(self, model_id: str, r_value: int, target_module: list[str], lora_alpha: int):
        self.model_id = model_id
        self.r_value = r_value
        self.target_module = target_module
        self.lora_alpha = lora_alpha
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.config = LoraConfig(
            r=self.r_value,
            target_modules=self.target_module,
            lora_alpha=self.lora_alpha
        )
        self.model = get_peft_model(self.model, self.config)
        self.model.print_trainable_parameters()
        self.d_model = self.model.config.hidden_size
        self.model.model.lm_head = torch.nn.Identity()
        self.model.model.model.norm = torch.nn.Identity()
        self.value_head = torch.nn.Linear(self.d_model, self.d_model)



    def forward_pass(self, explanations: list[str], batch_size, GRPO_size):
        # LOAD DEVICE
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    

        #run model
        token_ids = self.tokenizer(explanations, return_tensors='pt', padding=True).to(device)
        # now need to fix taking the last token of logits, and do instead advanced slicing
        mask_ends = token_ids["attention_mask"].sum(dim=1) - 1
        indexes=torch.arange(len(explanations)).to(device)
        input_ids = token_ids["input_ids"]
        output_logits = self.model(input_ids=input_ids, attention_mask=token_ids["attention_mask"]).logits[indexes, mask_ends, :]
        ar_activations = self.value_head(output_logits)
        return ar_activations

