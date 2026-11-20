"""Step 11: export cats for the human quiz.

Picks 40 real cats (held out: TabPFN never trained on them), 40 TabPFN cats and
40 shuffled cats, and writes them as part numbers to quiz_cats.json.
The quiz page draws all of them with the same drawing code.

Run generate_parts.py with the SAME seed first, so the generated files and the
held-out real cats come from the same split:
    python generate_parts.py --train 880 --seed 0
    python export_quiz.py --seed 0
"""
import argparse
import json

import numpy as np
import pandas as pd

from generate_parts import COLUMNS, load_faces


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--per-group", type=int, default=40)
    p.add_argument("--test", type=int, default=200)
    p.add_argument("--seed", type=int, default=0, help="must match the generate_parts.py run")
    return p.parse_args()


def to_records(df, group):
    out = []
    for _, row in df.iterrows():
        cat = {"group": group}
        for col in COLUMNS:
            v = row[col]
            cat[col] = None if pd.isna(v) else round(float(v), 3)
        out.append(cat)
    return out


def main():
    args = parse_args()
    faces = load_faces().sample(frac=1, random_state=args.seed).reset_index(drop=True)
    real = faces.iloc[-args.test:].head(args.per_group)
    tabpfn = pd.read_csv("data/generated_parts_tabpfn.csv").head(args.per_group)
    shuffled = pd.read_csv("data/generated_parts_shuffled.csv").head(args.per_group)

    cats = to_records(real, "real") + to_records(tabpfn, "tabpfn") + to_records(shuffled, "shuffled")
    rng = np.random.default_rng(args.seed)
    rng.shuffle(cats)
    for i, cat in enumerate(cats):
        cat["id"] = i

    with open("quiz_cats.json", "w") as f:
        json.dump({"columns": COLUMNS, "cats": cats}, f)
    print(f"wrote quiz_cats.json: {len(cats)} cats "
          f"({args.per_group} real, {args.per_group} tabpfn, {args.per_group} shuffled), seed {args.seed}")


if __name__ == "__main__":
    main()
