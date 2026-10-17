"""Re-ranking retrieved chunks for relevance."""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import CrossEncoder

DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# CrossEncoder logits; tune if refusal rate is too high or too low.
DEFAULT_MIN_RERANK_SCORE = -1.0


@lru_cache(maxsize=1)
def get_cross_encoder(model_name: str = DEFAULT_CROSS_ENCODER) -> CrossEncoder:
    return CrossEncoder(model_name)


def cross_encoder_scores(
    query: str,
    passages: list[str],
    model_name: str = DEFAULT_CROSS_ENCODER,
) -> list[float]:
    """Return a relevance score for each query–passage pair."""
    if not query.strip() or not passages:
        return []

    model = get_cross_encoder(model_name)
    pairs = [[query.strip(), passage] for passage in passages]
    raw_scores = model.predict(pairs)
    return [float(score) for score in raw_scores]
