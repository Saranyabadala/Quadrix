"""Step 6 -- threshold tuning and evaluation.

The competition metric, and the only one, is **F_0.5**:

    F_0.5 = (1.25 * P * R) / (0.25 * P + R)

Precision and recall are still computed -- F_0.5 is derived from them and
cannot be computed without them -- but they are strictly internal. They are
never printed on their own and never returned as a reported result, because a
report that shows precision and recall side by side invites tuning against the
wrong number. F_0.5 is what gets reported; the components do not.

Every sklearn metric that used to be imported here (``roc_auc_score``,
``average_precision_score``, ``precision_recall_curve``) and the hand-rolled
``accuracy`` column are gone. They are not "extra" information about an F_0.5
model, they are a second and third set of decisions to make.

Why the threshold still has to be chosen deliberately
-----------------------------------------------------
A classifier returns a ranking; the threshold converts that ranking into
decisions, and every downstream consumer inherits the choice. The tradeoff is
genuinely asymmetric:

  * False positives are expensive and compounding. A wrong match is invisible
    in the output -- it merges two real businesses -- and it damages whatever
    consumes the result downstream. One entity merged into the wrong parent is
    worse than one entity left unmatched.
  * False negatives are visible and recoverable. A missing match shows up as an
    unmatched row, and it can be found later by reviewing low-scoring pairs.

That asymmetry is why the pipeline emits a *tiered* decision (matched /
manual_review / unmatched) rather than a binary one, and why `tune_threshold`
returns one tuned threshold that the caller then uses as the tiering's
`high_threshold`.

Evaluation honesty
------------------
Metrics are computed on whichever labels exist. When those are silver
(rule-derived) labels, the numbers are reported with ``label_kind`` set to
``silver`` and every report carries a warning. See labels.py for why.
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from .metrics import BETA2, f_beta_from_counts

#: The one reported metric, as a single named constant so the reporting layer,
#: the CSV columns and the plots cannot drift apart.
F05_LABEL = "F_0.5"


# ---------------------------------------------------------------------------
# F_0.5-only reporting surface
# ---------------------------------------------------------------------------

def f05_at(
    y_true: Sequence[int], y_score: Sequence[float], threshold: float
) -> float:
    """Flat-pairwise F_0.5 of ``y_score >= threshold`` against ``y_true``.

    Precision and recall are computed here and deliberately **not** returned:
    this function's return value is the metric that gets reported, so it must
    not be a vehicle for the components.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.size == 0:
        return 0.0
    pred = (y_score >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    n_pred = tp + int(((pred == 1) & (y_true == 0)).sum())
    n_true = tp + int(((pred == 0) & (y_true == 1)).sum())
    return f_beta_from_counts(tp, n_pred, n_true, BETA2)


def _threshold_grid(start: float, stop: float, step: float) -> List[float]:
    """[0.10, 0.15, ... 0.90] -- exact decimal steps, no float drift.

    Computed as ``start + i * step`` and rounded to 10 places rather than with
    ``np.arange``, so 0.10 + 16 * 0.05 is 0.90 and not 0.9000000000000001.
    """
    n = int(round((stop - start) / step)) + 1
    return [round(start + i * step, 10) for i in range(max(0, n))]


def tune_threshold(
    y_true: Sequence[int],
    y_score: Sequence[float],
    start: float = 0.10,
    stop: float = 0.90,
    step: float = 0.05,
    return_sweep: bool = False,
) -> Dict[str, float]:
    """Sweep 0.10 -> 0.90 in steps of 0.05 and return the F_0.5-maximizing one.

    This is the **only** place F_0.5 influences a parameter: it produces the
    `high_threshold` of the tiered decision (Change 7). It is *not* the reported
    evaluation -- that is `final_evaluation`, and it needs labels.

    Ties are broken toward the **higher** threshold. F_0.5 weights precision
    twice as heavily as recall, so when two operating points score identically
    the more conservative one is correct for a merge use case: a false merge is
    silent, a missed match is reviewable.

    Returns ``{"high_threshold", "f05", "n_thresholds"}`` and nothing else, so
    the tuned threshold cannot leak a competing metric into a report. Pass
    ``return_sweep=True`` to also receive a DataFrame with exactly two columns,
    ``threshold`` and ``f05``, for the diagnostic plot and CSV.
    """
    grid = _threshold_grid(start, stop, step)
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.size == 0 or not grid:
        return {"high_threshold": float(stop), "f05": 0.0, "n_thresholds": 0}

    scores = [f05_at(y_true, y_score, t) for t in grid]
    best = max(scores)
    # Highest threshold among the maximizers: search the reversed score list so
    # argmax lands on the last (largest-threshold) occurrence.
    best_threshold = grid[len(grid) - 1 - int(np.argmax(scores[::-1]))]
    out = {
        "high_threshold": round(float(best_threshold), 4),
        "f05": round(float(best), 4),
        "n_thresholds": int(len(grid)),
    }
    if return_sweep:
        return out, pd.DataFrame({"threshold": grid, F05_LABEL: scores})
    return out


def final_evaluation(
    y_true: Sequence[int],
    y_score: Sequence[float],
    threshold: float,
    label: str = "validation",
    verbose: bool = True,
) -> Dict[str, object]:
    """The reported evaluation: one number, F_0.5 at the chosen threshold.

    Called once per run, on the held-out split that threshold tuning never
    touched. Prints F_0.5 and nothing else -- no precision line, no recall line,
    no AUC, no accuracy.

    On unlabeled test data this function is **not** called: F_0.5 needs true
    labels, and inventing a proxy for them would report a number that does not
    mean anything.
    """
    f05 = f05_at(y_true, y_score, threshold)
    if verbose:
        print(f"  {F05_LABEL} ({label}) = {f05:.4f}")
    return {
        "split": label,
        "threshold": round(float(threshold), 4),
        F05_LABEL: round(float(f05), 4),
    }


def final_evaluation_macro(
    predicted: Mapping[str, Set[str]],
    truth: Mapping[str, Set[str]],
    all_s1_ids: Optional[Sequence[str]] = None,
    label: str = "validation",
    verbose: bool = True,
) -> Dict[str, object]:
    """The reported evaluation for the per-Source-1-entity metric.

    Same formula, applied per Source 1 entity and then averaged with equal
    weight over **all** of them, singletons included -- which is what the
    challenge actually scores. `metrics.f05_macro` returns precision, recall and
    a flat F_0.5 alongside; they are deliberately dropped here so the reported
    payload cannot carry a second metric.

    This, not `final_evaluation`, is what the entity-resolution pipeline calls:
    tuning happens on pair labels, but the score that means something is the
    per-entity one.
    """
    from .metrics import f05_macro

    result = f05_macro(predicted, truth, all_entities=all_s1_ids)
    f05 = float(result["f05_macro"])
    if verbose:
        print(f"  {F05_LABEL} ({label}) = {f05:.4f}")
    return {
        "split": label,
        F05_LABEL: round(f05, 4),
        "n_entities_scored": int(result["n_entities"]),
    }



# ---------------------------------------------------------------------------
# Legacy benchmark helpers
#
# These predate the F_0.5-only surface above and are still used by the richer
# synthetic benchmark path in pipeline.py. Their `precision`/`recall`/`f1`
# fields are internal selection inputs; nothing in the reporting layer prints
# them.
# ---------------------------------------------------------------------------

def sweep_thresholds(
    y_true: Sequence[int], y_score: Sequence[float], n_points: int = 200
) -> pd.DataFrame:
    """F_0.5 across a dense grid of probability thresholds.

    ``accuracy`` was removed: on a 1-99 imbalance it is dominated by the
    negatives and looks excellent while being useless. So were ``roc_auc`` and
    ``pr_auc``, which answered a question this pipeline does not ask.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    lo = float(np.nanmin(y_score)) if len(y_score) else 0.0
    hi = float(np.nanmax(y_score)) if len(y_score) else 1.0
    grid = np.unique(np.concatenate([
        np.linspace(max(0.0, lo), min(1.0, hi), n_points),
        np.array([0.5]),
    ]))

    rows = []
    for t in grid:
        pred = (y_score >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append({
            "threshold": float(t),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision,   # internal selection input
            "recall": recall,         # internal selection input
            "f1": f1,                 # internal selection input
            F05_LABEL: f_beta_from_counts(tp, tp + fp, tp + fn, BETA2),
            "n_predicted_positive": tp + fp,
        })
    return pd.DataFrame(rows)


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a proportion.

    Kept for the legacy precision-floor path: the estimated precisions it
    describes are often near 1.0 with a small denominator, where the textbook
    normal interval is degenerate (width 0) and misleadingly confident. Not a
    reported metric.
    """
    if trials <= 0:
        return (float("nan"), float("nan"))
    p = successes / trials
    denom = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2))
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def recommend_threshold(
    sweep: pd.DataFrame,
    target_precision: float = 0.99,
    min_recall_floor: float = 0.60,
) -> Dict[str, object]:
    """Operating points for the legacy benchmark path.

    ``max_f05`` is the reported one. ``max_f1`` and ``precision_target`` are
    kept because the richer benchmark's CLI exposes a precision floor; both
    remain internal selections, and the precision/recall fields inside them are
    never printed by the pipeline.
    """
    if sweep.empty:
        return {}

    f05_row = sweep.loc[sweep[F05_LABEL].idxmax()]
    f1_row = sweep.loc[sweep["f1"].idxmax()]

    ok = sweep[sweep["precision"] >= target_precision]
    if len(ok) > 0:
        # Require a few predicted positives, otherwise a threshold that predicts
        # nothing trivially achieves precision 1.0.
        ok = ok[ok["n_predicted_positive"] >= 5]
    prec_row = ok.loc[ok["f1"].idxmax()] if len(ok) > 0 else None

    def describe(row) -> Optional[Dict[str, object]]:
        if row is None:
            return None
        n = int(row["n_predicted_positive"])
        lo, hi = wilson_interval(int(row["tp"]), n)
        return {
            "threshold": round(float(row["threshold"]), 4),
            F05_LABEL: round(float(row[F05_LABEL]), 4),
            "precision": round(float(row["precision"]), 4),  # internal
            "recall": round(float(row["recall"]), 4),        # internal
            "f1": round(float(row["f1"]), 4),                # internal
            "n_predicted_positive": n,
            "precision_ci95": [round(lo, 4), round(hi, 4)],
        }

    return {
        "max_f05": describe(f05_row),
        "max_f1": describe(f1_row),
        "precision_target": describe(prec_row),
        "target_precision": target_precision,
        "note": (
            "max_f05 is the reported operating point: F_0.5 is the only "
            "reported metric, so the operating point is the one that maximizes "
            "it."
        ),
    }


def evaluate_at(
    y_true: Sequence[int], y_score: Sequence[float], threshold: float
) -> Dict[str, float]:
    """Per-threshold diagnostic dict for the legacy benchmark path.

    ``f05`` is the reported field. precision/recall/f1 stay because
    `recommend_threshold` selects on them internally.
    """
    y_true = np.asarray(y_true, dtype=int)
    pred = (np.asarray(y_score, dtype=float) >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "threshold": round(float(threshold), 4),
        "tp": tp, "fp": fp, "fn": fn,
        "n_predicted_positive": tp + fp,
        F05_LABEL: round(f_beta_from_counts(tp, tp + fp, tp + fn, BETA2), 4),
        "precision": round(precision, 4),  # internal
        "recall": round(recall, 4),        # internal
        "f1": round(f1, 4),                # internal
    }


def make_plots(
    sweep: pd.DataFrame,
    y_val: Sequence[int],
    y_score_val: Sequence[float],
    threshold: float,
    out_path: str,
    importances: Optional[pd.DataFrame] = None,
) -> str:
    """Write the F_0.5-vs-threshold and feature-importance figures.

    Only F_0.5 is plotted. A precision/recall panel would put two other
    metrics on the page beside the one actually being optimized, and invites
    picking an operating point by eye off the wrong curve.

    Uses the Agg backend explicitly: this runs headless in CI and on servers
    where no display is attached.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # --- F_0.5 vs threshold --------------------------------------------------
    ax = axes[0]
    ax.plot(sweep["threshold"], sweep[F05_LABEL], lw=2, color="steelblue")
    ax.axvline(threshold, color="crimson", ls=":", lw=2,
               label=f"chosen {threshold:.3f}")
    best = float(np.nanmax(sweep[F05_LABEL])) if len(sweep) else float("nan")
    ax.set_xlabel("decision threshold")
    ax.set_ylabel(F05_LABEL)
    ax.set_title(f"{F05_LABEL} vs threshold (validation)  |  best on grid {best:.4f}")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    # --- feature importance --------------------------------------------------
    ax = axes[1]
    if importances is not None and len(importances) > 0:
        top = importances.head(20).iloc[::-1]
        ax.barh(top["feature"], top["importance"], color="steelblue")
        ax.set_xlabel("importance (gain / permutation)")
        ax.set_title("Top 20 features")
        ax.grid(alpha=0.3, axis="x")
    else:
        ax.text(0.5, 0.5, "no importances", ha="center", va="center")

    fig.suptitle(
        f"Match-model threshold tuning  |  {F05_LABEL} is the only reported "
        "metric  |  NOTE: metrics on rule-derived (silver) labels are "
        "optimistic; see README",
        fontsize=10, color="darkred",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def sample_for_manual_review(
    scored: pd.DataFrame, threshold: float, n: int = 200, seed: int = 42
) -> pd.DataFrame:
    """Draw a random sample of *accepted* pairs for human spot-checking.

    Precision measured on the labeled set is only as trustworthy as those
    labels. Auditing a random sample of what the pipeline actually emits is the
    only way to get an unbiased read on real precision, and it is the sample
    that should be labelled and fed back to retune the threshold.
    """
    accepted = scored[scored["score"] >= threshold]
    if accepted.empty:
        return accepted
    rng = np.random.default_rng(seed)
    take = min(n, len(accepted))
    idx = rng.choice(accepted.index.values, size=take, replace=False)
    return accepted.loc[idx].sort_values("score", ascending=False)
