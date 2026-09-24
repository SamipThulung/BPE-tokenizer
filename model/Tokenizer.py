
import pickle
import regex
from typing import Iterable, Iterator


# GPT2 pattern
GPT2_PATTERN = regex.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+(?:\p{M}+)?|"""
    r""" ?\p{N}+| ?[^\s\p{L}\p{N}]+[\r\n]*|"""
    r"""\s*[\r\n]+|\s+(?!\S)|\s+"""
)


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):

        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        # ID -> bytes
        self.id_to_bytes = dict(vocab)

        # bytes -> ID
        self.bytes_to_id = {
            token_bytes: token_id
            for token_id, token_bytes in vocab.items()
        }

        # Merge pair -> priority
        # Smaller rank means earlier merge.
        self.merge_ranks = {
            pair: rank
            for rank, pair in enumerate(merges)
        }

        # Special tokens, longest first
        self.special_tokens = sorted(
            set(self.special_tokens),
            key=len,
            reverse=True,
        )

        # Compile special-token pattern once
        if self.special_tokens:

            special_pattern = (
                "("
                + "|".join(
                    regex.escape(token)
                    for token in self.special_tokens
                )
                + ")"
            )

            self.special_pattern = regex.compile(
                special_pattern
            )

        else:
            self.special_pattern = None

    def _split_text(self, text: str):

        if self.special_pattern:

            pieces = self.special_pattern.split(text)

        else:

            pieces = [text]

        for piece in pieces:

            if not piece:
                continue

            if piece in self.special_tokens:
                yield piece

            else:
                yield from GPT2_PATTERN.findall(piece)

    def _encode_piece(self, piece: str) -> list[int]:

        # Convert pretoken into individual bytes.
        sequence = [
            bytes([b])
            for b in piece.encode("utf-8")
        ]

        # Repeatedly merge the highest-priority pair.
        while len(sequence) > 1:

            best_rank = float("inf")
            best_index = None

            for i in range(len(sequence) - 1):

                pair = (
                    sequence[i],
                    sequence[i + 1],
                )

                rank = self.merge_ranks.get(
                    pair,
                    float("inf"),
                )

                if rank < best_rank:

                    best_rank = rank
                    best_index = i

            # No mergeable pair remains.
            if best_index is None:
                break

            # Merge the selected pair.
            i = best_index

            merged = sequence[i] + sequence[i + 1]

            sequence[i:i + 2] = [merged]

        # Convert final bytes into IDs.
        token_ids = []

        for token_bytes in sequence:

            if token_bytes not in self.bytes_to_id:

                raise ValueError(
                    f"Token missing from vocabulary: "
                    f"{token_bytes!r}"
                )

            token_ids.append(
                self.bytes_to_id[token_bytes]
            )

        return token_ids

    def encode(self, text: str) -> list[int]:

        token_ids = []

        for piece in self._split_text(text):

            # Special tokens are single IDs.
            if piece in self.special_tokens:

                special_bytes = piece.encode("utf-8")

                if special_bytes not in self.bytes_to_id:

                    raise ValueError(
                        f"Special token missing from vocab: "
                        f"{piece!r}"
                    )

                token_ids.append(
                    self.bytes_to_id[special_bytes]
                )

            else:

                # Ordinary pretoken -> BPE -> IDs
                token_ids.extend(
                    self._encode_piece(piece)
                )

        return token_ids

    def encode_iterable(
        self,
        iterable: Iterable[str],
    ) -> Iterator[int]:

        for text in iterable:

            yield from self.encode(text)


    def decode(self, ids: list[int]) -> str:

        byte_chunks = []

        for token_id in ids:

            if token_id not in self.id_to_bytes:

                raise ValueError(
                    f"Unknown token ID: {token_id}"
                )

            byte_chunks.append(
                self.id_to_bytes[token_id]
            )

        return b"".join(byte_chunks).decode(
            "utf-8",
            errors="replace",
        )

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):

        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)

        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(
            vocab=vocab,
            merges=merges,
            special_tokens=special_tokens,
        )