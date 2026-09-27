"""The competition metric: macro-averaged F_0.5 per Source 1 entity.

This is *not* flat precision/recall, and the difference is the whole game.
Getting it wrong here means every threshold and every model decision is tuned
against the wrong objective, so it is implemented first and tested against the
worked example in the problem statement.

How it works
------------
For each Source 1 entity independently, with predicted set P and true set T:

    precision  = |P n T| / |P|        (1.0 if P is empty, by convention)
    recall     = |P n T| / |T|        (1.0 if T is empty and P is empty, else 0)
    F_0.5      = 1.25 * P * R / (0.25 * P + R)

then the entity scores are averaged with equal weight over **all** Source 1
entities, singletons included.

Consequences that drive the design:

* **Singletons are first-class.** 5.6% of Source 1 entities have no true match.
  Each is worth a full 1.0 for predicting the empty list and a full 0.0 for
  predicting anything. Under a flat metric an entity like this is invisible;
  here it is worth as much as any other entity.
* **Precision is weighted 2x.** beta=0.5 means precision carries twice the
  weight of recall. A false merge on a singleton is the single most expensive
  error available, and it costs the entire 1.0 for that entity.
* **Per-entity averaging caps the damage from big entities.** One large
  franchise with 40 matches contributes 1/N of the score whether you get all 40
  or none. You cannot buy score by winning a few large entities.

The threshold should therefore be tuned to maximize this quantity directly,
not to maximize F1 and hope.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

BETA2 = 0.25  # beta^2 for beta = 0.5


def f_beta_from_counts(tp: int, n_pred: int, n_true: int, beta2: float = BETA2) -> float:
    """F_beta for one entity from raw counts.

    Handles the two empty-set conventions explicitly, because getting them
    wrong is easy and silently inflates or deflates the score:

    * nothing predicted, nothing true  -> 1.0 (correctly identified singleton)
    * nothing predicted, something true -> 0.0 (missed every match)
    * something predicted, nothing true -> 0.0 (false merge on a singleton)
    * both non-empty                   -> the standard formula
    """
    if n_pred == 0 and n_true == 0:
        return 1.0
    if n_pred == 0 or n_true == 0:
        return 0.0
    precision = tp / n_pred
    recall = tp / n_true
    denom = beta2 * precision + recall
    if denom <= 0:
        return 0.0
    return (1.0 + beta2) * precision * recall / denom


def f05_macro(
    predicted: Mapping[str, Set[str]],
    truth: Mapping[str, Set[str]],
    all_entities: Optional[Iterable[str]] = None,
) -> Dict[str, float]:
    """Macro-averaged F_0.5 over Source 1 entities.

    ``predicted`` / ``truth``  maps entity_id -> set of matched ids.
    ``all_entities``  the entities that must be scored. Defaults to the union of
                      truth keys and predicted keys, which is correct only if
                      every Source 1 entity appears in the ground truth. In this
                      challenge it does -- the training truth file has a row for
                      all 2,206,821 Source 1 records, including empty ones -- so
                      the default is safe, but passing it explicitly is clearer
                      at test time where the truth is unknown.
    """
    if all_entities is None:
        entities = set(truth) | set(predicted)
    else:
        entities = set(all_entities)

    scores = np.empty(len(entities), dtype=np.float64)
    tp_tot = pred_tot = true_tot = 0
    exact = 0
    for i, eid in enumerate(entities):
        t = truth.get(eid, set())
        p = predicted.get(eid, set())
        # Intersect on the predicted side so an id that does not exist cannot
        # inflate recall; the validator rejects such submissions anyway.
        tp = len(p & t)
        scores[i] = f_beta_from_counts(tp, len(p), len(t))
        tp_tot += tp
        pred_tot += len(p)
        true_tot += len(t)
        exact += int(p == t)

    flat_p = tp_tot / pred_tot if pred_tot else 0.0
    flat_r = tp_tot / true_tot if true_tot else 0.0
    return {
        "f05_macro": float(scores.mean()) if len(scores) else 0.0,
        "n_entities": len(entities),
        "n_perfect": int(exact),
        "pct_perfect": 100.0 * exact / max(1, len(entities)),
        "flat_precision": flat_p,
        "flat_recall": flat_r,
        "f05_flat": f_beta_from_counts(tp_tot, pred_tot, true_tot),
        "mean_pred_per_entity": pred_tot / max(1, len(entities)),
    }


def f05_macro_from_columns(
    s1_ids: Sequence[str],
    pred_lists: Sequence[Sequence[str]],
    true_lists: Sequence[Sequence[str]],
) -> Dict[str, float]:
    """Convenience wrapper for the columnar form the pipeline produces."""
    pred: Dict[str, Set[str]] = {}
    truth: Dict[str, Set[str]] = {}
    for s1, p, t in zip(s1_ids, pred_lists, true_lists):
        pred[s1] = set(p)
        truth[s1] = set(t)
    return f05_macro(pred, truth, all_entities=s1_ids)


def f05_from_scores(
    s1_ids: np.ndarray,
    cand_ids: np.ndarray,
    scores: np.ndarray,
    truth: Mapping[str, Set[str]],
    threshold: float,
) -> Dict[str, float]:
    """Score a scored candidate array directly, without building dicts.

    Used on the validation split during threshold tuning, where the candidate
    set is already in columnar form. Keeps the threshold sweep cheap enough to
    run over hundreds of candidates.
    """
    keep = scores >= threshold
    pred: Dict[str, List[str]] = {}
    if keep.any():
        ks1 = s1_ids[keep]
        kcid = cand_ids[keep]
        order = np.argsort(ks1, kind="stable")
        ks1, kcid = ks1[order], kcid[order]
        bounds = np.flatnonzero(np.r_[True, ks1[1:] != ks1[:-1], True])
        for a, b in zip(bounds[:-1], bounds[1:]):
            pred.setdefault(str(ks1[a]), []).extend(str(x) for x in kcid[a:b])
    return f05_macro(pred, truth, all_entities=list(truth.keys()))


def load_ground_truth(path: str) -> Tuple[Dict[str, Set[str]], List[str]]:
    """Read the challenge ground truth into ``{s1_id: set(matched_ids)}``.

    Preserves order and returns the full entity list, including the 123,247
    singletons whose match list is empty -- dropping those would silently
    change the denominator of the macro average and inflate the score.
    """
    truth: Dict[str, Set[str]] = {}
    order: List[str] = []
    with open(path, encoding="utf-8") as fh:
        header = fh.readline()
        if not header.lower().startswith("source1_entity_id"):
            raise ValueError(f"unexpected ground-truth header: {header!r}")
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            s1 = parts[0].strip()
            ids = parts[1].split(",") if len(parts) > 1 and parts[1].strip() else []
            truth[s1] = {x.strip() for x in ids if x.strip()}
            order.append(s1)
    return truth, order
