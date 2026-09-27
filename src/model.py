"""Step 5 -- training the match classifier.

Backend choice
--------------
LightGBM is preferred, but it is used through a thin wrapper so the pipeline
runs on machines where the compiled wheel cannot load. The macOS LightGBM wheel
links against LLVM's ``libomp.dylib``; without Homebrew's ``brew install libomp``
(or Linux/Windows equivalents) ``import lightgbm`` raises a dlopen error.

The fallback is ``sklearn.ensemble.HistGradientBoostingClassifier``, which is
LightGBM's own algorithm ported into scikit-learn -- the same histogram
binning, the same native NaN routing, the same leaf-wise growth. For this
problem it is not a second-class model, and it removes an undeclared system
dependency. The wrapper exposes one interface so nothing downstream changes:

    clf.fit(X, y) -> clf.predict_proba(X) -> clf.feature_importances_

The reported numbers say which backend actually ran, so nobody mistakes
fallback results for LightGBM results.

Splitting
---------
Stratified train/validation/test by label. The test set is touched exactly once,
at the very end, for the final numbers. Everything else -- early stopping,
threshold selection, feature-importance review -- happens on validation. This
separation is the whole reason the reported test numbers are trustworthy.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class SplitData:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def sizes(self) -> Dict[str, int]:
        return {
            "train": int(len(self.train)),
            "val": int(len(self.val)),
            "test": int(len(self.test)),
        }

    def reset(self) -> "SplitData":
        """Return a copy with a clean RangeIndex on each split.

        The splits come out of train_test_split with an arbitrary index, and the
        feature frame is later re-indexed and concatenated. Resetting keeps
        positional access (`.iloc`) and label-based access (`.loc`) in agreement
        everywhere downstream.
        """
        return SplitData(
            train=self.train.reset_index(drop=True),
            val=self.val.reset_index(drop=True),
            test=self.test.reset_index(drop=True),
        )


def make_splits(
    features: pd.DataFrame,
    label_col: str = "label",
    test_size: float = 0.20,
    val_size: float = 0.20,
    seed: int = 42,
) -> SplitData:
    """Stratified train/val/test split.

    Stratifying by label matters here more than usual: silver-label sets are
    heavily imbalanced (most pairs are non-matches), so an unstratified split
    can leave the validation set with too few positives for a threshold to mean
    anything.
    """
    labeled = features[features[label_col].notna()].copy()
    if labeled.empty:
        raise ValueError("no labeled rows to split on")

    y = labeled[label_col].astype(int)
    # First carve out the test set, then split what remains into train/val, each
    # time preserving class proportions.
    train_val, test = train_test_split(
        labeled, test_size=test_size, random_state=seed, stratify=y
    )
    rel_val = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val, test_size=rel_val, random_state=seed,
        stratify=train_val[label_col].astype(int),
    )
    return SplitData(train=train, val=val, test=test).reset()


@dataclass
class MatchClassifier:
    """Thin wrapper over LightGBM or sklearn's HistGradientBoosting."""

    prefer: str = "lightgbm"
    n_estimators: int = 400
    learning_rate: float = 0.06
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 20
    n_jobs: int = 1
    random_state: int = 42
    backend: str = field(default="", init=False)
    model: object = field(default=None, init=False)
    feature_cols: List[str] = field(default_factory=list, init=False)
    importances: Optional[pd.DataFrame] = field(default=None, init=False)

    def _try_lightgbm(self):
        try:
            import lightgbm as lgb  # noqa: F401

            return True, None
        except Exception as exc:  # dlopen failure, missing wheel, ...
            return False, str(exc).splitlines()[0] if str(exc) else "unavailable"

    def fit(self, train: pd.DataFrame, val: pd.DataFrame, feature_cols: List[str]) -> "MatchClassifier":
        self.feature_cols = list(feature_cols)

        have_lgb, lgb_err = self._try_lightgbm()
        use_lgb = have_lgb and self.prefer == "lightgbm"

        Xtr = train[self.feature_cols]
        ytr = train["label"].astype(int)
        Xva = val[self.feature_cols]
        yva = val["label"].astype(int)

        if use_lgb:
            import lightgbm as lgb

            self.backend = "lightgbm"
            self.model = lgb.LGBMClassifier(
                objective="binary",
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=self.max_leaf_nodes,
                min_child_samples=self.min_samples_leaf,
                subsample=0.9,
                subsample_freq=1,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=-1,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model.fit(
                    Xtr, ytr,
                    eval_set=[(Xva, yva)],
                    eval_metric="auc",
                    callbacks=[lgb.early_stopping(50, verbose=False)],
                )
            importances = self.model.booster_.feature_importance(importance_type="gain")
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier

            self.backend = "sklearn-histgb" + (
                "" if have_lgb else f" (lightgbm unavailable: {lgb_err})"
            )
            self.model = HistGradientBoostingClassifier(
                max_iter=self.n_estimators,
                learning_rate=self.learning_rate,
                max_leaf_nodes=self.max_leaf_nodes,
                min_samples_leaf=self.min_samples_leaf,
                l2_regularization=1.0,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=50,
                random_state=self.random_state,
            )
            self.model.fit(Xtr, ytr)
            # Permutation importance is the honest analogue of LightGBM gain for
            # a model that has no built-in gain measure. Computed on the
            # validation split, never on train, to avoid rewarding memorization.
            from sklearn.inspection import permutation_importance

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                perm = permutation_importance(
                    self.model, Xva, yva, n_repeats=5, random_state=self.random_state,
                    scoring="average_precision", n_jobs=self.n_jobs,
                )
            importances = perm.importances_mean

        self.importances = (
            pd.DataFrame({
                "feature": self.feature_cols,
                "importance": np.asarray(importances, dtype=float),
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        self.importances["importance_pct"] = (
            self.importances["importance"] / self.importances["importance"].sum() * 100.0
        )
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        X = frame[self.feature_cols]
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]
        # A bare LightGBM Booster (what `load` restores) exposes predict() only.
        return np.asarray(self.model.predict(X), dtype=float)

    def best_iteration(self) -> Optional[int]:
        for attr in ("best_iteration_", "best_iteration", "n_iter_"):
            value = getattr(self.model, attr, None)
            if value:
                return int(value)
        return 0

    def save(self, path: str) -> str:
        """Persist the fitted model and its feature order to a directory.

        Needed because the real-schema pipeline makes two streaming passes: one
        to gather training pairs, one to score with the trained model. Without a
        saved model the second pass would have to retrain, or the pipeline would
        have to hold every pair in memory between the passes.

        The feature order is saved alongside the model because a model applied
        to columns in a different order is silently wrong -- LightGBM and
        sklearn both index positionally, and a reordered frame produces
        confident nonsense rather than an error.
        """
        import json
        import os
        import pickle

        os.makedirs(path, exist_ok=True)
        params = {
            "prefer": self.prefer, "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate, "max_leaf_nodes": self.max_leaf_nodes,
            "min_samples_leaf": self.min_samples_leaf, "n_jobs": self.n_jobs,
            "random_state": self.random_state,
        }
        if self.backend == "lightgbm":
            self.model.booster_.save_model(os.path.join(path, "model.txt"))
        else:
            with open(os.path.join(path, "model.pkl"), "wb") as fh:
                pickle.dump(self.model, fh)
        with open(os.path.join(path, "meta.json"), "w") as fh:
            json.dump({"backend": self.backend, "feature_cols": self.feature_cols,
                       "params": params}, fh, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "MatchClassifier":
        """Restore a model saved by `save`. Used for unlabeled test prediction."""
        import json
        import os
        import pickle

        with open(os.path.join(path, "meta.json")) as fh:
            meta = json.load(fh)
        obj = cls(**meta.get("params", {}))
        obj.backend = meta["backend"]
        obj.feature_cols = list(meta["feature_cols"])
        if obj.backend == "lightgbm":
            import lightgbm as lgb

            obj.model = lgb.Booster(model_file=os.path.join(path, "model.txt"))
        else:
            with open(os.path.join(path, "model.pkl"), "rb") as fh:
                obj.model = pickle.load(fh)
        return obj
