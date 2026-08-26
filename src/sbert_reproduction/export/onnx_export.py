import os
import torch

def export_sentence_encoder_to_onnx(model, output_path: str, max_seq_len: int = 128) -> str:
    """Exports SentenceEncoder PyTorch model to ONNX format with dynamic batch and sequence axes."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dummy_input_ids = torch.ones((1, max_seq_len), dtype=torch.long)
    dummy_attention_mask = torch.ones((1, max_seq_len), dtype=torch.long)

    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "sentence_embedding": {0: "batch_size"}
    }

    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        output_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["sentence_embedding"],
        dynamic_axes=dynamic_axes,
        opset_version=14
    )
    return output_path
