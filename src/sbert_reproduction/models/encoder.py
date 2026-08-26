import torch.nn as nn

class TransformerEncoderWrapper(nn.Module):
    """Wraps HuggingFace AutoModel / BertModel transformer backbone."""
    def __init__(self, model_name: str = "bert-base-uncased"):
        super().__init__()
        self.model_name = model_name
        self.auto_model = None

    def load_pretrained(self):
        from transformers import AutoModel
        self.auto_model = AutoModel.from_pretrained(self.model_name)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        if self.auto_model is None:
            self.load_pretrained()
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        outputs = self.auto_model(**kwargs)
        return outputs.last_hidden_state
