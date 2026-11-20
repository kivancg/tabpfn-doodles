"""Step 3: TabPFN draws its first cats, one column at a time.

How it works (the chain rule, same idea as the planets):
  column 0 (x0)  -> pick a real starting x
  column 1 (y0)  -> TabPFN predicts y0 from x0, we sample one value
  column 2 (pen0)-> always 1 (every cat starts with the pen lifted)
  column 3 (x1)  -> TabPFN predicts x1 from x0, y0, pen0, we sample
  ...
  column 191     -> predicted from the columns before it

For each column we fit TabPFN on the real cats:
  features = the first 2 points (where the head starts) + the last WINDOW columns
  target   = this column
then predict it for the cats we are generating and SAMPLE from the prediction
(not the average: the average of "ear here" and "ear there" is an ear in the middle).

x, y columns -> TabPFNRegressor, sampled from its predicted quantiles
pen columns  -> TabPFNClassifier, sampled from its predicted probabilities

Writes data/generated_tabpfn.csv and generated_grid.png
(top row: real cats, bottom row: TabPFN cats, first stroke in red).
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from tabpfn import TabPFNClassifier, TabPFNRegressor

from make_table import draw_points

QUANTILES = np.linspace(0.02, 0.98, 25)
ANCHOR_COLUMNS = 6  # x0, y0, pen0, x1, y1, pen1: where the main outline starts


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=int, default=500, help="how many real cats TabPFN sees")
    p.add_argument("--n", type=int, default=16, help="how many cats to generate")
    p.add_argument("--window", type=int, default=30, help="how many previous columns to look at")
    p.add_argument("--n-estimators", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def sample_regression(model, X, rng):
    """Sample one value per row from TabPFN's predicted distribution."""
    q = np.array(model.predict(X, output_type="quantiles", quantiles=list(QUANTILES)))
    q = np.sort(q, axis=0)  # shape (n_quantiles, n_rows)
    u = rng.uniform(QUANTILES[0], QUANTILES[-1], size=X.shape[0])
    return np.array([np.interp(u[i], QUANTILES, q[:, i]) for i in range(X.shape[0])])


def sample_class(model, X, rng):
    """Sample one class per row from TabPFN's predicted probabilities."""
    proba = model.predict_proba(X)
    return np.array([rng.choice(model.classes_, p=p / p.sum()) for p in proba])


def feature_columns(j, window):
    anchor = set(range(min(ANCHOR_COLUMNS, j)))
    recent = set(range(max(0, j - window), j))
    return sorted(anchor | recent)


def to_points(row):
    return [(row[i], row[i + 1], int(row[i + 2])) for i in range(0, len(row), 3)]


def save_grid(real, fake, path, n=8, size=128):
    scale = size / 256
    img = Image.new("RGB", (n * size, 2 * size), (255, 255, 255))
    pen = ImageDraw.Draw(img)
    for r, rows in enumerate((real, fake)):
        for i in range(min(n, len(rows))):
            draw_points(pen, to_points(rows[i]), i * size, r * size, scale)
            pen.rectangle([i * size, r * size, (i + 1) * size - 1, (r + 1) * size - 1], outline=(200, 200, 200))
    img.save(path)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv("data/cats_table.csv")
    train = df.sample(args.train, random_state=args.seed).to_numpy(dtype=float)
    columns = list(df.columns)
    fake = np.zeros((args.n, len(columns)))

    print(f"device {device}, {args.train} real cats, {len(columns)} columns, generating {args.n} cats")
    start = time.time()
    for j, name in enumerate(columns):
        target = train[:, j]
        if np.unique(target).size == 1:  # e.g. pen0 is always 1
            fake[:, j] = target[0]
            continue
        if j == 0:  # nothing to predict from yet: pick real starting points
            fake[:, j] = rng.choice(target, size=args.n)
            continue

        feats = feature_columns(j, args.window)
        X_train, X_fake = train[:, feats], fake[:, feats]
        if name.startswith("pen"):
            model = TabPFNClassifier(n_estimators=args.n_estimators, device=device, random_state=args.seed)
            model.fit(X_train, target.astype(int))
            fake[:, j] = sample_class(model, X_fake, rng)
        else:
            model = TabPFNRegressor(n_estimators=args.n_estimators, device=device, random_state=args.seed)
            model.fit(X_train, target)
            fake[:, j] = np.clip(sample_regression(model, X_fake, rng), 0, 255)

        if (j + 1) % 24 == 0:
            print(f"  column {j + 1}/{len(columns)}  ({time.time() - start:.0f}s)")

    pd.DataFrame(fake.round(1), columns=columns).to_csv("data/generated_tabpfn.csv", index=False)
    save_grid(train, fake, "generated_grid.png")
    print(f"done in {time.time() - start:.0f}s")
    print("saved data/generated_tabpfn.csv and generated_grid.png (top: real, bottom: TabPFN)")


if __name__ == "__main__":
    main()
