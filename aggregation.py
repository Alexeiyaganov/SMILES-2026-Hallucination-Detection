"""
aggregation.py — Token aggregation strategy and feature extraction
               (student-implemented).

Converts per-token, per-layer hidden states from the extraction loop in
``solution.py`` into flat feature vectors for the probe classifier.

Two stages can be customised independently:

  1. ``aggregate`` — select layers and token positions, pool into a vector.
  2. ``extract_geometric_features`` — optional hand-crafted features
     (enabled by setting ``USE_GEOMETRIC = True`` in ``solution.py``).

Both stages are combined by ``aggregation_and_feature_extraction``, the
single entry point called from the notebook.
"""

from __future__ import annotations

import torch


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert per-token hidden states into a single feature vector.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
                        Layer index 0 is the token embedding; index -1 is the
                        final transformer layer.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D feature tensor of shape ``(hidden_dim,)`` or
        ``(k * hidden_dim,)`` if multiple layers are concatenated.

    Student task:
        Replace or extend the skeleton below with alternative layer selection,
        token pooling (mean, max, weighted), or multi-layer fusion strategies.
    """
    # ------------------------------------------------------------------
    # STUDENT: Replace or extend the aggregation below.
    # ------------------------------------------------------------------

    # Multi-layer fusion: select middle layers and mean-pool over real tokens.
    LAYER_INDICES = [8, 12, 16, 20]  # middle layers carry richest semantics

    selected = hidden_states[LAYER_INDICES]  # (n_selected, seq_len, hidden_dim)

    # Mean pooling over all real (non-padding) tokens per layer
    mask_expanded = attention_mask.unsqueeze(0).unsqueeze(-1)  # (1, seq_len, 1)
    selected_masked = selected * mask_expanded
    token_counts = mask_expanded.sum(dim=1).clamp(min=1)  # (1, 1)
    features = selected_masked.sum(dim=1) / token_counts  # (n_selected, hidden_dim)

    feature = features.flatten()  # (n_selected * hidden_dim,)

    return feature
    # ------------------------------------------------------------------


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract hand-crafted geometric / statistical features from hidden states.

    Called only when ``USE_GEOMETRIC = True`` in ``solution.py``.  The
    returned tensor is concatenated with the output of ``aggregate``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D float tensor of shape ``(n_geometric_features,)``.  The length
        must be the same for every sample.

    Student task:
        Replace the stub below.  Possible features: layer-wise activation
        norms, inter-layer cosine similarity (representation drift), or
        sequence length.
    """
    # ------------------------------------------------------------------
    # STUDENT: Replace or extend the geometric feature extraction below.
    # ------------------------------------------------------------------

    import torch.nn.functional as F

    real_mask = attention_mask.bool()
    n_layers = hidden_states.size(0)
    features_list = []

    for layer_idx in range(n_layers):
        layer = hidden_states[layer_idx]
        real_tokens = layer[real_mask]

        if real_tokens.size(0) == 0:
            features_list.append(torch.zeros(1))  # norm
            features_list.append(torch.zeros(1))  # std
            continue

        mean_vec = real_tokens.mean(dim=0)
        features_list.append(mean_vec.norm().unsqueeze(0))       # per-layer norm

        std_per_dim = real_tokens.std(dim=0)
        features_list.append(std_per_dim.mean().unsqueeze(0))    # per-layer mean std

    # Cosine similarity between first and last selected aggregatable layer
    LAYER_INDICES = [16, 20]
    first_real = hidden_states[LAYER_INDICES[0]][real_mask]
    last_real = hidden_states[LAYER_INDICES[-1]][real_mask]
    if first_real.size(0) > 0 and last_real.size(0) > 0:
        cos_sim = F.cosine_similarity(
            first_real.mean(dim=0).unsqueeze(0),
            last_real.mean(dim=0).unsqueeze(0),
        )
        features_list.append(cos_sim)
    else:
        features_list.append(torch.zeros(1))

    return torch.cat(features_list, dim=0)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append geometric features.

    Main entry point called from ``solution.ipynb`` for each sample.
    Concatenates the output of ``aggregate`` with that of
    ``extract_geometric_features`` when ``use_geometric=True``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``
                        for a single sample.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.
        use_geometric:  Whether to append geometric features.  Controlled by
                        the ``USE_GEOMETRIC`` flag in ``solution.ipynb``.

    Returns:
        A 1-D float tensor of shape ``(feature_dim,)`` where
        ``feature_dim = hidden_dim`` (or larger for multi-layer or geometric
        concatenations).
    """
    agg_features = aggregate(hidden_states, attention_mask)  # (feature_dim,)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
