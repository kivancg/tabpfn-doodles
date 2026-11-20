# tabpfn-doodles

Can a tabular foundation model learn to draw? This project teaches [TabPFN](https://github.com/PriorLabs/TabPFN) to doodle cats from a few hundred [Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset) sketches, then tests whether a classifier can spot the fakes.

**Results page:** `docs/index.html` (published with GitHub Pages) · **Quiz:** `docs/quiz.html`

## The short version

- Raw pen points and PCA both failed. TabPFN needs every column to mean the same thing in every row, and point 30 was an ear in one cat and a whisker in another.
- Describing each cat as **parts and how they relate** worked: ears, whiskers, nose and eyes, measured relative to the head. That gives one row of 21 numbers per cat.
- On those numbers, a classifier can barely tell TabPFN's cats from real ones.

| Generator | ROC-AUC (0.5 = can't tell) | Range over 5 seeds |
|---|---|---|
| Real vs real (sanity check) | 0.497 | 0.473 – 0.540 |
| **TabPFN** | **0.564** | 0.516 – 0.586 |
| Shuffled (each part from a different real cat) | 0.775 | 0.738 – 0.800 |
| Gaussian (each column its own bell curve) | 0.782 | 0.765 – 0.818 |

880 training cats, 200 held-out real cats vs 200 generated, gradient-boosting classifier with 5-fold cross-validation.

TabPFN beat "shuffled" in every seed, so it learned how the parts fit together. It is close to real but not identical: slightly above "real vs real" in all five seeds.

Before that final run TabPFN scored 0.66. Permutation importance showed one column doing almost all the work: the nose's roundness. A third of real noses had exactly the same value, the median we had filled in for noses too small to measure. Replacing that made-up value with a yes/no "dot or shape" column took TabPFN from 0.647 to 0.570 on the same seed.

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python get_cats.py                 # 2,000 recognized cats -> data/cats_raw.ndjson
python make_table.py               # attempt 1: 64 points per cat -> data/cats_table.csv
python generate_tabpfn.py          # attempt 1: TabPFN point by point (fails)
python generate_pca.py             # attempt 2: PCA + TabPFN / Gaussian (fails)
python find_parts.py               # attempt 3: find ears, whiskers, nose, eyes -> data/parts.csv
python generate_parts.py --train 880 --seed 0   # generate and score (the main result)
python which_columns.py            # which columns give the fakes away
python export_quiz.py --seed 0     # cats for the quiz -> quiz_cats.json
```

Run `generate_parts.py` for seeds 0–4 to reproduce the table. Tested with tabpfn 10.2 and Python 3.14 on an RTX 3080. TabPFN generates 200 cats in about 20 seconds.

## Limits

- "Close to real" is measured on the 21 numbers, not on pictures.
- The part-finder uses simple rules and keeps 54% of cats: the faces where it found both ears.
- The drawings redraw those numbers, so real cats lose detail too. All groups use the same drawing code.
- No human study was run. The quiz is a playable demo.

## Data

Cat drawings from Google's [Quick, Draw! dataset](https://github.com/googlecreativelab/quickdraw-dataset), licensed CC BY 4.0. The data files are not in this repo; `get_cats.py` downloads them.
