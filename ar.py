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
        self.model.lm_head = torch.nn.Identity()
        self.model.model.norm = torch.nn.Identity()
        self.value_head = torch.nn.Linear(self.d_model, self.d_model)



    def forward_pass(self, explanations: list[str]):
        #run model
        ar_activations = []
        token_ids = self.tokenizer(explanations, return_tensors='pt')
        input_ids = token_ids["input_ids"]
        output_logits = self.model(input_ids=input_ids).logits[:, -1, :]
        ar_activations = self.value_head(output_logits)
        return ar_activations

