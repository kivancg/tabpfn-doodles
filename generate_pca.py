"""Step 4: generate cats in PCA space instead of 192 raw columns.

1. PCA learns ~20 whole-shape patterns ("components") from the real cats.
   Every cat = average cat + c1 * pattern1 + c2 * pattern2 + ... + c20 * pattern20.
   So each cat becomes 20 numbers (c1..c20) instead of 192.
2. Generate new sets of 20 numbers two ways:
     gaussian: each component sampled from a bell curve fitted to the real cats
     tabpfn:   chain rule over the 20 components (c1 picked from real values,
               c2 predicted from c1, c3 from c1..c2, ...), sampling each one
3. Turn the 20 numbers back into 64 points and draw them.

Pen columns are 0/1, which PCA can't keep exact. We scale them up before PCA
(so they matter as much as x/y) and round back to 0/1 after.

Writes pca_grid.png with 4 rows:
  real | real squeezed through PCA (best PCA can do) | gaussian | tabpfn
and data/generated_pca_gaussian.csv, data/generated_pca_tabpfn.csv
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from sklearn.decomposition import PCA
from tabpfn import TabPFNRegressor

from generate_tabpfn import sample_regression, to_points
from make_table import draw_points

PEN_SCALE = 100.0  # pen 0/1 -> 0/100 so PCA pays attention to it


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=int, default=500, help="how many real cats")
    p.add_argument("--components", type=int, default=20)
    p.add_argument("--n", type=int, default=16, help="how many cats to generate per method")
    p.add_argument("--n-estimators", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def encode(rows, pen_cols):
    out = rows.copy()
    out[:, pen_cols] *= PEN_SCALE
    return out


def decode(rows, pen_cols):
    out = rows.copy()
    out[:, pen_cols] = (out[:, pen_cols] / PEN_SCALE > 0.5).astype(float)
    out[:, pen_cols[0]] = 1.0  # every cat starts with the pen lifted
    xy = [i for i in range(out.shape[1]) if i not in set(pen_cols)]
    out[:, xy] = np.clip(out[:, xy], 0, 255)
    return out


def gaussian_codes(codes, n, rng):
    """Each component independently from a bell curve with the real mean and spread."""
    return rng.normal(codes.mean(axis=0), codes.std(axis=0), size=(n, codes.shape[1]))


def tabpfn_codes(codes, n, rng, n_estimators, seed, device):
    """Chain rule over the components."""
    fake = np.zeros((n, codes.shape[1]))
    fake[:, 0] = rng.choice(codes[:, 0], size=n)
    for j in range(1, codes.shape[1]):
        model = TabPFNRegressor(n_estimators=n_estimators, device=device, random_state=seed)
        model.fit(codes[:, :j], codes[:, j])
        fake[:, j] = sample_regression(model, fake[:, :j], rng)
    return fake


def save_grid(row_sets, labels, path, n=8, size=128):
    scale = size / 256
    label_w = 90
    img = Image.new("RGB", (label_w + n * size, len(row_sets) * size), (255, 255, 255))
    pen = ImageDraw.Draw(img)
    for r, (rows, label) in enumerate(zip(row_sets, labels)):
        pen.text((6, r * size + size // 2 - 6), label, fill=(0, 0, 0))
        for i in range(min(n, len(rows))):
            ox, oy = label_w + i * size, r * size
            draw_points(pen, to_points(rows[i]), ox, oy, scale)
            pen.rectangle([ox, oy, ox + size - 1, oy + size - 1], outline=(200, 200, 200))
    img.save(path)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv("data/cats_table.csv")
    columns = list(df.columns)
    pen_cols = [i for i, c in enumerate(columns) if c.startswith("pen")]
    real = df.sample(args.train, random_state=args.seed).to_numpy(dtype=float)

    pca = PCA(n_components=args.components, random_state=args.seed)
    codes = pca.fit_transform(encode(real, pen_cols))
    kept = pca.explained_variance_ratio_.sum()
    print(f"{args.train} real cats, {len(columns)} columns -> {args.components} components "
          f"(they keep {kept:.0%} of the variation)")

    squeezed = decode(pca.inverse_transform(codes), pen_cols)

    start = time.time()
    gauss = decode(pca.inverse_transform(gaussian_codes(codes, args.n, rng)), pen_cols)
    print(f"gaussian: {time.time() - start:.1f}s")

    start = time.time()
    tab = decode(pca.inverse_transform(
        tabpfn_codes(codes, args.n, rng, args.n_estimators, args.seed, device)), pen_cols)
    print(f"tabpfn:   {time.time() - start:.1f}s")

    pd.DataFrame(gauss.round(1), columns=columns).to_csv("data/generated_pca_gaussian.csv", index=False)
    pd.DataFrame(tab.round(1), columns=columns).to_csv("data/generated_pca_tabpfn.csv", index=False)
    save_grid([real, squeezed, gauss, tab], ["real", "real via PCA", "gaussian", "tabpfn"], "pca_grid.png")
    print("saved pca_grid.png (rows: real | real via PCA | gaussian | tabpfn)")


if __name__ == "__main__":
    main()
