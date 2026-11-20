"""Step 7: generate cats as PARTS (ears, whiskers, nose, eyes) and draw them.

Data: data/parts.csv from find_parts.py, face cats only. Measured in the head's frame
(head ~2 wide, chin at y=+1, up = negative y).

Missing parts are real information (some cats have no nose), so we add flags:
  has_nose, has_eyes, and whisker counts (0 = no whiskers).
A part's numbers only exist when its flag says the part is there.

Generators (200 cats each):
  shuffled : every column copied from a different random real cat
             -> right values for each part, but no relations between parts
  gaussian : every column from its own bell curve -> no relations either
  tabpfn   : chain rule, each column predicted from the columns before it
             -> can learn relations (e.g. big ears go with a wide head)

Score: classifier test, real held-out cats vs generated, ROC-AUC (0.5 = can't tell).
If tabpfn beats shuffled, it learned how the parts relate.

Drawing: every cat (real ones too) is redrawn from its numbers by the same code,
so the pictures compare like with like.

Writes parts_gen_grid.png, data/parts_scores.csv, data/generated_parts_*.csv
"""
import argparse
import math
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from tabpfn import TabPFNClassifier, TabPFNRegressor

from generate_tabpfn import sample_class, sample_regression

# generation order: head and ears first, then the parts that sit on the face
COLUMNS = ["head_aspect",
           "ear_l_x", "ear_l_y", "ear_l_angle", "ear_r_x", "ear_r_y", "ear_r_angle",
           "whisk_l", "whisk_r", "whisk_len", "whisk_tilt",
           "has_nose", "nose_x", "nose_y", "nose_size", "nose_shape", "nose_round",
           "has_eyes", "eye_gap", "eye_y", "eye_size"]
DISCRETE = {"whisk_l", "whisk_r", "has_nose", "nose_shape", "has_eyes"}
PRESENT_IF = {  # column -> rule saying whether this cat has that part
    "whisk_len": lambda r: r["whisk_l"] + r["whisk_r"] > 0,
    "whisk_tilt": lambda r: r["whisk_l"] + r["whisk_r"] > 0,
    "nose_x": lambda r: r["has_nose"] == 1, "nose_y": lambda r: r["has_nose"] == 1,
    "nose_size": lambda r: r["has_nose"] == 1, "nose_shape": lambda r: r["has_nose"] == 1,
    "nose_round": lambda r: (r["has_nose"] == 1) & (r["nose_shape"] == 1),
    "eye_gap": lambda r: r["has_eyes"] == 1, "eye_y": lambda r: r["has_eyes"] == 1,
    "eye_size": lambda r: r["has_eyes"] == 1,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=int, default=500)
    p.add_argument("--test", type=int, default=200)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--n-estimators", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_faces():
    df = pd.read_csv("data/parts.csv")
    df = df[df["is_face"]].copy()
    df["has_nose"] = df["nose_x"].notna().astype(int)
    df["has_eyes"] = df["eye_gap"].notna().astype(int)
    df["whisk_l"] = df["whisk_l"].clip(upper=6)
    df["whisk_r"] = df["whisk_r"].clip(upper=6)
    # nose_shape: 1 = a shape we can measure roundness of, 0 = a dot / tiny mark (no roundness).
    # (Before, dots got the median roundness: a made-up value shared by ~1/3 of real cats.)
    df["nose_shape"] = np.where(df["has_nose"] == 1, df["nose_round"].notna().astype(float), np.nan)
    return df[COLUMNS].reset_index(drop=True)


def present_mask(fake, col):
    rule = PRESENT_IF.get(col)
    return np.ones(len(fake), bool) if rule is None else rule(fake).to_numpy(bool)


def gen_independent(real, n, rng, gaussian):
    """shuffled (gaussian=False) or gaussian (gaussian=True): every column on its own."""
    fake = pd.DataFrame(index=range(n), columns=COLUMNS, dtype=float)
    for col in COLUMNS:
        values = real[col].dropna().to_numpy()
        if col in DISCRETE or not gaussian:
            fake[col] = rng.choice(values, size=n)
        else:
            fake[col] = rng.normal(values.mean(), values.std(), size=n)
        fake.loc[~present_mask(fake, col), col] = np.nan
    return fake


def gen_tabpfn(real, n, rng, n_estimators, seed, device):
    """Chain rule: each column predicted from all columns before it (NaN = part missing)."""
    fake = pd.DataFrame(index=range(n), columns=COLUMNS, dtype=float)
    fake[COLUMNS[0]] = rng.choice(real[COLUMNS[0]].to_numpy(), size=n)
    for j, col in enumerate(COLUMNS[1:], start=1):
        prev = COLUMNS[:j]
        known = real[col].notna().to_numpy()
        X_train, y_train = real.loc[known, prev].to_numpy(float), real.loc[known, col].to_numpy()
        todo = present_mask(fake, col)
        if todo.sum() == 0:
            continue
        X_fake = fake.loc[todo, prev].to_numpy(float)
        if col in DISCRETE:
            model = TabPFNClassifier(n_estimators=n_estimators, device=device, random_state=seed)
            model.fit(X_train, y_train.astype(int))
            fake.loc[todo, col] = sample_class(model, X_fake, rng)
        else:
            model = TabPFNRegressor(n_estimators=n_estimators, device=device, random_state=seed)
            model.fit(X_train, y_train)
            fake.loc[todo, col] = sample_regression(model, X_fake, rng)
    return fake


def classifier_auc(real, fake, seed):
    X = np.vstack([real.to_numpy(float), fake.to_numpy(float)])
    y = np.r_[np.zeros(len(real)), np.ones(len(fake))]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_val_score(HistGradientBoostingClassifier(random_state=seed), X, y, cv=cv, scoring="roc_auc")
    return scores.mean(), scores.std()


# ---------- drawing a cat from its part numbers ----------

def wobble(points, rng, amount=0.025):
    return [(x + rng.normal(0, amount), y + rng.normal(0, amount)) for x, y in points]


def ellipse_point(cx, cy, rx, ry, t):
    return cx + rx * math.cos(t), cy + ry * math.sin(t)


def draw_cat(pen, row, ox, oy, size, rng):
    unit = size * 0.2
    to_px = lambda pts: [(ox + size / 2 + x * unit, oy + size * 0.8 + (y - 1) * unit) for x, y in pts]
    line = lambda pts, color=(0, 0, 0), closed=False: pen.line(
        to_px(pts + ([pts[0]] if closed else [])), fill=color, width=2)

    aspect = float(np.clip(row["head_aspect"], 0.5, 1.8))
    cx, cy, rx, ry = 0.0, 1 - aspect, 1.0, aspect
    head = [ellipse_point(cx, cy, rx, ry, 2 * math.pi * k / 40) for k in range(40)]
    line(wobble(head, rng), closed=True)

    for side in ("l", "r"):
        tx, ty, ang = row[f"ear_{side}_x"], row[f"ear_{side}_y"], row[f"ear_{side}_angle"]
        if np.isnan(tx):
            continue
        t = math.atan2((ty - cy) / ry, (tx - cx) / rx)
        base = ellipse_point(cx, cy, rx, ry, t)
        height = math.dist(base, (tx, ty))
        half = max(height * math.tan(math.radians(np.clip(ang, 20, 120)) / 2), 0.1)
        d = half / max(rx, ry)
        line(wobble([ellipse_point(cx, cy, rx, ry, t - d), (tx, ty), ellipse_point(cx, cy, rx, ry, t + d)], rng),
             color=(220, 0, 0))

    if not np.isnan(row["whisk_len"]):
        wy = row["nose_y"] if not np.isnan(row["nose_y"]) else cy + 0.3 * ry
        for side, count in ((-1, row["whisk_l"]), (1, row["whisk_r"])):
            count = int(count)
            for k in range(count):
                spread = (k - (count - 1) / 2) * 12
                a = math.radians(np.clip(row["whisk_tilt"], 0, 45) * (1 if k % 2 else -1) * 0.5 + spread)
                start = (side * 0.35, wy + 0.05 * k)
                end = (start[0] + side * row["whisk_len"] * math.cos(a), start[1] + row["whisk_len"] * math.sin(a))
                line(wobble([start, end], rng, 0.01), color=(30, 90, 230))

    if row["has_eyes"] == 1:
        r = max(row["eye_size"] / 2, 0.04)
        for side in (-1, 1):
            ex, ey = side * row["eye_gap"] / 2, row["eye_y"]
            eye = [(ex + r * math.cos(2 * math.pi * k / 10), ey + r * math.sin(2 * math.pi * k / 10)) for k in range(10)]
            line(eye, color=(240, 140, 0), closed=True)

    if row["has_nose"] == 1:
        nx, ny, s = row["nose_x"], row["nose_y"], max(row["nose_size"] / 2, 0.05)
        if np.isnan(row["nose_round"]):  # dot nose
            s = 0.05
            nose = [(nx + s * math.cos(2 * math.pi * k / 6), ny + s * math.sin(2 * math.pi * k / 6)) for k in range(6)]
        elif row["nose_round"] < 0.75:
            nose = [(nx - s, ny - s * 0.6), (nx + s, ny - s * 0.6), (nx, ny + s * 0.6)]
        else:
            nose = [(nx + s * math.cos(2 * math.pi * k / 10), ny + s * math.sin(2 * math.pi * k / 10)) for k in range(10)]
        line(nose, color=(0, 160, 60), closed=True)


def save_grid(sets, labels, path, rng, n=8, size=128):
    label_w = 90
    img = Image.new("RGB", (label_w + n * size, len(sets) * size), (255, 255, 255))
    pen = ImageDraw.Draw(img)
    for r, (df, label) in enumerate(zip(sets, labels)):
        pen.text((6, r * size + size // 2 - 6), label, fill=(0, 0, 0))
        for i in range(min(n, len(df))):
            ox, oy = label_w + i * size, r * size
            draw_cat(pen, df.iloc[i], ox, oy, size, rng)
            pen.rectangle([ox, oy, ox + size - 1, oy + size - 1], outline=(200, 200, 200))
    img.save(path)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    faces = load_faces().sample(frac=1, random_state=args.seed).reset_index(drop=True)
    # test = always the last cats, so runs with different --train sizes share the same test set
    if args.train + args.test > len(faces):
        raise SystemExit(f"--train {args.train} + --test {args.test} is more than the {len(faces)} face cats")
    test = faces.iloc[-args.test:].reset_index(drop=True)
    train = faces.iloc[: args.train].reset_index(drop=True)
    print(f"{len(faces)} face cats; train {len(train)}, test {len(test)}, {len(COLUMNS)} columns")

    gen = {}
    for name, make in [("shuffled", lambda: gen_independent(train, args.n, rng, gaussian=False)),
                       ("gaussian", lambda: gen_independent(train, args.n, rng, gaussian=True)),
                       ("tabpfn", lambda: gen_tabpfn(train, args.n, rng, args.n_estimators, args.seed, device))]:
        start = time.time()
        gen[name] = make()
        gen[name].round(3).to_csv(f"data/generated_parts_{name}.csv", index=False)
        print(f"  {name:9s} {time.time() - start:5.1f}s")

    print("\nclassifier test: real (held out) vs generated, ROC-AUC (0.5 = can't tell)")
    sanity = train.sample(args.n, random_state=args.seed)
    rows = []
    for name, fake in [("real vs real", sanity), *gen.items()]:
        mean, std = classifier_auc(test, fake, args.seed)
        rows.append({"method": name, "roc_auc": round(mean, 3), "std": round(std, 3)})
        print(f"  {name:13s} {mean:.3f} ± {std:.3f}")
    pd.DataFrame(rows).to_csv("data/parts_scores.csv", index=False)

    save_grid([test, gen["shuffled"], gen["gaussian"], gen["tabpfn"]],
              ["real (redrawn)", "shuffled", "gaussian", "tabpfn"], "parts_gen_grid.png", rng)
    print("\nsaved parts_gen_grid.png and data/parts_scores.csv")


if __name__ == "__main__":
    main()
