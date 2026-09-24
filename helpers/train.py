from collections import Counter
import time
from IPython.display import clear_output


def get_pairs(vocab):
    results = Counter()

    for tokens, freq in vocab.items():
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            results[pair] += freq

    return results


def merge_pairs(vocab, best_pair, new_id):
    new_vocab = Counter()

    for tokens, freq in vocab.items():
        result = []
        i = 0

        while i < len(tokens):
            if (
                i < len(tokens) - 1
                and (tokens[i], tokens[i + 1]) == best_pair
            ):
                result.append(new_id)
                i += 2
            else:
                result.append(tokens[i])
                i += 1

        new_vocab[tuple(result)] += freq

    return new_vocab


def train_BPE(pretokens, target_dataset_size):

    # 1. Count pretoken frequencies
    word_freq = Counter(pretokens)

    # 2. Convert words into byte sequences
    vocab = Counter({
        tuple(word.encode("utf-8")): freq
        for word, freq in word_freq.items()
    })

    # 3. Initialize byte IDs
    new_id = 256
    id_to_bytes = {
        i: bytes([i]) for i in range(256)
    }

    merge_history = {}

    # 4. Train BPE
    while new_id < target_dataset_size:
        clear_output(wait=True)
        print(f"Current vocab size: {new_id}", flush=True)
        # Count all adjacent pairs
        pairs = get_pairs(vocab)

        # Stop if no pairs remain
        if not pairs:
            break

        # Select most frequent pair
        best_pair = max(pairs, key=pairs.get)

        # Save merge
        merge_history[new_id] = best_pair

        # Create merged token bytes
        id_to_bytes[new_id] = (
            id_to_bytes[best_pair[0]]
            + id_to_bytes[best_pair[1]]
        )

        # Merge throughout vocabulary
        vocab = merge_pairs(vocab, best_pair, new_id)

        new_id += 1

    return vocab, merge_history, id_to_bytes


