"""Trie-constrained decoding for closed-set generative classification."""

from __future__ import annotations

from collections.abc import Callable, Iterable


class LegalLabelTrie:
    """A token trie whose terminal nodes only permit EOS."""

    def __init__(self, sequences: Iterable[Iterable[int]], eos_token_id: int) -> None:
        self.eos_token_id = int(eos_token_id)
        self.root: dict[int | None, dict] = {}
        count = 0
        for sequence in sequences:
            tokens = tuple(int(token) for token in sequence)
            if not tokens:
                raise ValueError("Legal labels must tokenize to at least one token")
            node = self.root
            for token in tokens:
                node = node.setdefault(token, {})
            node.setdefault(None, {})
            count += 1
        if count == 0:
            raise ValueError("At least one legal label is required")

    def allowed(self, prefix: Iterable[int]) -> list[int]:
        node = self.root
        consumed = []
        for value in prefix:
            token = int(value)
            consumed.append(token)
            if token not in node:
                raise RuntimeError(f"Generated prefix left the legal-label trie: {consumed}")
            node = node[token]
        result = [int(token) for token in node if token is not None]
        if None in node:
            result.append(self.eos_token_id)
        if not result:
            raise RuntimeError(f"Legal-label trie has no continuation for prefix: {consumed}")
        return sorted(set(result))

    def prefix_allowed_tokens_fn(self, prompt_length: int) -> Callable:
        if prompt_length < 0:
            raise ValueError("prompt_length must be non-negative")

        def allowed(_batch_id, input_ids):
            generated = input_ids[prompt_length:].tolist()
            # In batched generation, Transformers keeps invoking logits processors
            # for rows that have already emitted EOS while longer rows continue. It
            # may also append padding tokens to those finished rows. Validate that
            # EOS occurred at a legal terminal, then ignore the finished-row suffix.
            if self.eos_token_id in generated:
                terminal = generated[: generated.index(self.eos_token_id)]
                if self.eos_token_id not in self.allowed(terminal):
                    raise RuntimeError(f"EOS appeared before a complete legal label: {generated}")
                return [self.eos_token_id]
            return self.allowed(generated)

        return allowed


def label_token_sequences(tokenizer, labels: list[str]) -> tuple[list[list[int]], int]:
    """Tokenize exact labels with both natural post-colon spacing variants."""
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("Tokenizer must define eos_token_id")
    sequences: list[list[int]] = []
    owners: dict[tuple[int, ...], str] = {}
    for label in labels:
        for text in (label, " " + label):
            tokens = tokenizer.encode(text, add_special_tokens=False)
            key = tuple(int(token) for token in tokens)
            previous = owners.get(key)
            if previous is not None and previous != label:
                raise ValueError(
                    f"Labels {previous!r} and {label!r} have the same constrained tokenization"
                )
            if key not in owners:
                owners[key] = label
                sequences.append(list(key))
    return sequences, max(len(sequence) for sequence in sequences) + 1


def exact_generated_label(text: str, labels: list[str]) -> str:
    by_casefold = {label.strip().casefold(): label for label in labels}
    key = text.strip().casefold()
    if key not in by_casefold:
        raise RuntimeError(f"Constrained decoder produced a non-label string: {text!r}")
    return by_casefold[key]
