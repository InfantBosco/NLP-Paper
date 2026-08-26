import os
import torch

def save_checkpoint(model, output_dir: str, filename: str = "checkpoint.pt") -> str:
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    torch.save(model.state_dict(), filepath)
    return filepath

def load_checkpoint(model, filepath: str):
    state_dict = torch.load(filepath, map_location="cpu")
    model.load_state_dict(state_dict)
    return model
