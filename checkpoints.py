from activations import Activations
from av import AV
from ar import AR
import torch
import itertools
import os
import shutil


def save_checkpoints(step, av, ar, av_optimizer, ar_optimizer, N: int):

    # making new file for a step
    os.makedirs(f"checkpoints/step_{step:05d}", exist_ok=True)

    # Saving lora weights of av and ar + ar_value_head
    av.model.save_pretrained(f"checkpoints/step_{step:05d}/av_lora")
    ar.model.save_pretrained(f"checkpoints/step_{step:05d}/ar_lora")
    torch.save(ar.value_head.state_dict(), f"checkpoints/step_{step:05d}/ar_value_head.pt")

    # saving optimizers
    torch.save(av_optimizer.state_dict(), f"checkpoints/step_{step:05d}/av_optimizer.pt")
    torch.save(ar_optimizer.state_dict(), f"checkpoints/step_{step:05d}/ar_optimizer.pt")


    # Delete oldest checkpoint
    folders = sorted(os.listdir("checkpoints"))
    if len(folders) > N:
        shutil.rmtree(f"checkpoints/{folders[0]}")

def load_checkpoints(av, ar, av_optimizer, ar_optimizer):
    if not os.path.exists("checkpoints") or len(os.listdir("checkpoints")) == 0:
        return 0
    # sorting files
    folders = sorted(os.listdir("checkpoints"))

    # Using last one, loading in checkpoint for ar and av lora
    av_loaded = av.model.load_adapter(f"checkpoints/{folders[-1]}/av_lora", adapter_name="default")
    ar_base_loaded = ar.model.load_adapter(f"checkpoints/{folders[-1]}/ar_lora", adapter_name="default")
    
    # storing then loading in value head ar weights
    ar_value_head_stored = torch.load(f"checkpoints/{folders[-1]}/ar_value_head.pt")
    ar.value_head.load_state_dict(ar_value_head_stored)

    # storing then loading in optimizers values for av and ar
    av_optimizer_stored = torch.load(f"checkpoints/{folders[-1]}/av_optimizer.pt")
    av_optimizer.load_state_dict(av_optimizer_stored)

    ar_optimizer_stored = torch.load(f"checkpoints/{folders[-1]}/ar_optimizer.pt")
    ar_optimizer.load_state_dict(ar_optimizer_stored)

    return int(folders[-1].replace("step_",""))