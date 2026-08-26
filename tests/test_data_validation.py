import pytest
from sbert_reproduction.data.validation import validate_stsb_data, validate_nli_data

def test_validate_stsb_data_valid():
    sample = [{"split": "train", "score": "3.5", "sentence1": "Hello", "sentence2": "Hi"}]
    assert validate_stsb_data(sample) is True

def test_validate_stsb_data_invalid_score():
    sample = [{"split": "train", "score": "6.0", "sentence1": "Hello", "sentence2": "Hi"}]
    with pytest.raises(ValueError):
        validate_stsb_data(sample)

def test_validate_nli_data_valid():
    sample = [{"split": "train", "label": "entailment", "sentence1": "Cat", "sentence2": "Animal"}]
    assert validate_nli_data(sample) is True
