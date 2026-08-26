from typing import List

class CorpusContainer:
    def __init__(self, sentences: List[str]):
        self.sentences = sentences
        self.embeddings = None

    def __len__(self):
        return len(self.sentences)
