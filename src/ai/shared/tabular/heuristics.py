"""Heuristic rules for inferring metric polarities and method roles."""

from __future__ import annotations

import re

from ai.shared.tabular.contracts import MetricDirection

_MAXIMIZE_PATTERNS = frozenset(
    {
        "acc",
        "accuracy",
        "top1",
        "top-1",
        "top5",
        "top-5",
        "f1",
        "f1-score",
        "f1_score",
        "precision",
        "recall",
        "bleu",
        "rouge",
        "map",
        "mrr",
        "ndcg",
        "auc",
        "roc",
        "pr-auc",
        "throughput",
        "reward",
        "psnr",
        "ssim",
        "snr",
        "iou",
        "miou",
        "dice",
        "pass@1",
        "pass@5",
        "exact_match",
        "em",
    }
)

_MINIMIZE_PATTERNS = frozenset(
    {
        "loss",
        "train_loss",
        "val_loss",
        "test_loss",
        "perplexity",
        "ppl",
        "latency",
        "runtime",
        "delay",
        "duration",
        "time",
        "error",
        "err",
        "error_rate",
        "wer",
        "cer",
        "mse",
        "mae",
        "rmse",
        "std",
        "var",
        "params",
        "parameters",
        "flops",
        "macs",
        "vram",
        "memory",
    }
)

_PROPOSED_METHOD_REGEX = re.compile(
    r"\b(ours|proposed|flux|our\s+model|our\s+framework|our\s+method)\b",
    re.IGNORECASE,
)


def infer_metric_direction(header: str) -> MetricDirection:
    """Infer whether higher or lower values are optimal based on standard ML literature conventions."""
    clean = re.sub(r"[^a-zA-Z0-9_\-@]", " ", header).lower().strip()
    tokens = {t.strip() for t in clean.split() if t.strip()}

    if tokens & _MAXIMIZE_PATTERNS:
        return MetricDirection.MAXIMIZE
    if tokens & _MINIMIZE_PATTERNS:
        return MetricDirection.MINIMIZE

    for tok in tokens:
        for max_pat in _MAXIMIZE_PATTERNS:
            if max_pat in tok:
                return MetricDirection.MAXIMIZE
        for min_pat in _MINIMIZE_PATTERNS:
            if min_pat in tok:
                return MetricDirection.MINIMIZE

    return MetricDirection.NEUTRAL


def is_proposed_method_name(name: str) -> bool:
    """Detect if method name represents the author's proposed model (e.g. 'Ours', 'Flux-Net')."""
    return bool(_PROPOSED_METHOD_REGEX.search(name))
