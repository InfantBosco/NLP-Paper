from typing import List, Dict, Any

class SmartBatchingCollate:
    """Collates and pads sentences dynamically within a mini-batch."""
    def __init__(self, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts_a = [item["sentence1"] for item in batch]
        texts_b = [item["sentence2"] for item in batch]

        features_a = self.tokenizer(texts_a, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        features_b = self.tokenizer(texts_b, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")

        result = {"features_a": features_a, "features_b": features_b}
        if "label" in batch[0]:
            import torch
            result["labels"] = torch.tensor([item["label"] for item in batch])
        return result
