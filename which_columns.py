"""Step 9: which columns give the fakes away?

Uses the fakes saved by the LAST run of generate_parts.py
(data/generated_parts_<method>.csv) and the same 200 held-out real cats.

Two measures per column:
  alone   : ROC-AUC of a classifier that sees ONLY this column.
            ~0.50 = this column's values look like the real ones on their own.
            Higher = the values themselves are off (wrong range, wrong counts...).
  drop    : permutation importance. Train on all columns, then scramble one column
            in the held-out fold and see how much the ROC-AUC falls.
            Big drop = the classifier leans on this column (alone OR with others).

A column with a big drop but alone ~0.5 means the values are fine on their own,
but they don't fit with the other columns (a broken relation).

Also prints real vs fake percentiles for the 3 worst columns.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from generate_parts import COLUMNS, load_faces


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", default=["tabpfn", "shuffled"])
    p.add_argument("--test", type=int, default=200)
    p.add_argument("--seed", type=int, default=0, help="must match the generate_parts.py run")
    p.add_argument("--repeats", type=int, default=5, help="scrambles per column per fold")
    return p.parse_args()


def alone_auc(X, y, j, seed):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    model = HistGradientBoostingClassifier(random_state=seed)
    return cross_val_score(model, X[:, [j]], y, cv=cv, scoring="roc_auc").mean()


def permutation_drops(X, y, seed, repeats):
    rng = np.random.default_rng(seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    drops = np.zeros(X.shape[1])
    full = []
    for train_idx, val_idx in cv.split(X, y):
        model = HistGradientBoostingClassifier(random_state=seed).fit(X[train_idx], y[train_idx])
        X_val, y_val = X[val_idx], y[val_idx]
        base = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        full.append(base)
        for j in range(X.shape[1]):
            for _ in range(repeats):
                scrambled = X_val.copy()
                scrambled[:, j] = rng.permutation(scrambled[:, j])
                drops[j] += base - roc_auc_score(y_val, model.predict_proba(scrambled)[:, 1])
    return np.mean(full), drops / (cv.get_n_splits() * repeats)


def describe(values):
    v = values[~np.isnan(values)]
    if len(v) == 0:
        return "all missing"
    p5, p50, p95 = np.percentile(v, [5, 50, 95])
    return f"5%={p5:7.2f}  median={p50:7.2f}  95%={p95:7.2f}  missing={np.isnan(values).mean():4.0%}"


def main():
    args = parse_args()
    faces = load_faces().sample(frac=1, random_state=args.seed).reset_index(drop=True)
    real = faces.iloc[-args.test:][COLUMNS].to_numpy(float)

    for method in args.methods:
        fake = pd.read_csv(f"data/generated_parts_{method}.csv")[COLUMNS].to_numpy(float)
        X = np.vstack([real, fake])
        y = np.r_[np.zeros(len(real)), np.ones(len(fake))]

        overall, drops = permutation_drops(X, y, args.seed, args.repeats)
        table = pd.DataFrame({
            "column": COLUMNS,
            "drop": drops.round(3),
            "alone": [round(alone_auc(X, y, j, args.seed), 3) for j in range(len(COLUMNS))],
        }).sort_values("drop", ascending=False)

        print(f"\n===== {method}: overall ROC-AUC {overall:.3f} =====")
        print("drop  = how much scrambling this column hurts the classifier (bigger = gives fakes away)")
        print("alone = ROC-AUC using only this column (0.5 = values look real on their own)\n")
        print(table.to_string(index=False))

        print(f"\nworst 3 columns, real vs {method}:")
        for col in table["column"].head(3):
            j = COLUMNS.index(col)
            print(f"  {col}")
            print(f"    real : {describe(real[:, j])}")
            print(f"    fake : {describe(fake[:, j])}")


if __name__ == "__main__":
    main()
