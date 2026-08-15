from collections import Counter
from typing import List, Dict, Tuple

class Tokenizer():

    def __init__(self):

        self.merges: Dict[Tuple[int, int], int] = {}
        self.vocab: Dict[int, bytes] = {i : bytes([i]) for i in range(256)}
        self.vocab_size = 512 


    def train(self, text: str, vocab_size: int = 512):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        tokens = list(text.encode("utf-8"))

        for i in range(num_merges):
            # Count pairs
            pairs = Counter()
            for j in range(len(tokens) - 1):
                pairs[(tokens[j], tokens[j + 1])] += 1

            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            new_id = 256 + i

            # Record the merge
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            # Replace the pair
            new_tokens = []
            j = 0
            while j < len(tokens):
                if j < len(tokens) - 1 and (tokens[j], tokens[j + 1]) == best_pair:
                    new_tokens.append(new_id)
                    j += 2
                else:
                    new_tokens.append(tokens[j])
                    j += 1
            tokens = new_tokens

        self.vocab_size = 256 + len(self.merges)