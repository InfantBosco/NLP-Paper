from sbert_reproduction.data.splits import filter_by_split

def test_filter_by_split():
    rows = [
        {"split": "train", "id": 1},
        {"split": "dev", "id": 2},
        {"split": "test", "id": 3}
    ]
    train_rows = filter_by_split(rows, "train")
    assert len(train_rows) == 1
    assert train_rows[0]["id"] == 1
