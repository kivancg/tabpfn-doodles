"""Step 6: find the parts of each cat doodle (head, ears, whiskers, nose, eyes).

Everything is measured in the HEAD'S OWN FRAME, so a big cat and a small cat with the
same proportions get the same numbers (the "invariance" idea):
  - head = the longest stroke
  - center x = middle of the head, scale = half the head width (so the head is ~2 wide)
  - center y = one half-width above the chin (bottom of the head)
  - up is negative y (screen coordinates)

Rough rules:
  ears     : sharp peaks above the head, on the head line or as separate strokes
  whiskers : fairly straight strokes that reach past the sides of the head
  nose     : small stroke near the middle of the face
  eyes     : small strokes in the upper half of the face

Writes data/parts.csv (one row per cat, NaN = part not found) and parts_grid.png:
ears red, whiskers blue, nose green, eyes orange, head black, other strokes grey.
"""
import json
import math

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from make_table import resample_stroke, stroke_length

N_HEAD = 64        # points to resample the head line with
PEAK_ANGLE = 110   # degrees: a corner sharper than this counts as pointy


def corner_angle(pts, i, k=3, closed=False):
    n = len(pts)
    a = pts[(i - k) % n] if closed else pts[max(i - k, 0)]
    b = pts[(i + k) % n] if closed else pts[min(i + k, n - 1)]
    v1, v2 = a - pts[i], b - pts[i]
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0:
        return 180.0
    return math.degrees(math.acos(np.clip(np.dot(v1, v2) / norm, -1, 1)))


def head_frame(xs, ys):
    pts = np.array(resample_stroke(xs, ys, N_HEAD), dtype=float)
    x5, x95 = np.percentile(pts[:, 0], [5, 95])
    rx = max((x95 - x5) / 2, 1.0)
    cx = (x5 + x95) / 2
    cy = np.percentile(pts[:, 1], 95) - rx
    return cx, cy, rx


def to_frame(xs, ys, frame):
    cx, cy, rx = frame
    return np.column_stack([(np.array(xs) - cx) / rx, (np.array(ys) - cy) / rx])


def find_peaks(p, closed, top_y=-0.6, w=4, k=3):
    """Sharp peaks pointing up (local lowest y, corner sharper than PEAK_ANGLE): index list."""
    peaks = []
    y = p[:, 1]
    n = len(p)
    for i in range(n):
        window = [(i + d) % n if closed else min(max(i + d, 0), n - 1) for d in range(-w, w + 1)]
        if y[i] < top_y and y[i] == y[window].min() and corner_angle(p, i, k=k, closed=closed) < PEAK_ANGLE:
            if not peaks or i - peaks[-1] >= 2 * w:
                peaks.append(i)
    return peaks


def head_ears(head, closed):
    """Sharp peaks above the head on the head line."""
    return find_peaks(head, closed, top_y=-0.7)


def is_closed(p, length):
    return length > 0 and math.dist(p[0], p[-1]) < 0.35 * length


def stroke_info(xs, ys, frame, rx):
    p = to_frame(xs, ys, frame)
    length = stroke_length(xs, ys) / rx
    ends = math.dist(p[0], p[-1])
    diag = math.dist(p.min(axis=0), p.max(axis=0))
    return p, length, ends, diag


def roundness(p):
    """4*pi*area / perimeter^2: 1 for a circle, ~0.6 for a triangle."""
    if len(p) < 3:
        return np.nan
    x, y = p[:, 0], p[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))
    perim = np.sum(np.linalg.norm(p - np.roll(p, -1, axis=0), axis=1))
    return np.nan if perim == 0 else 4 * math.pi * area / perim ** 2


def find_parts(drawing):
    strokes = [s for s in drawing if len(s[0]) > 0]
    lengths = [stroke_length(*s) for s in strokes]
    h = int(np.argmax(lengths))
    frame = head_frame(*strokes[h])
    rx = frame[2]
    labels = {i: "other" for i in range(len(strokes))}
    labels[h] = "head"

    # head shape and ears drawn as part of the head line
    hx, hy = strokes[h]
    head = to_frame(*zip(*resample_stroke(hx, hy, N_HEAD)), frame)
    closed = math.dist(head[0], head[-1]) < 0.5
    ears = [(head[i, 0], head[i, 1], corner_angle(head, i, closed=closed)) for i in head_ears(head, closed)]
    ear_spans = [(i - 4, i + 4) for i in head_ears(head, closed)]
    y10 = np.percentile(head[:, 1], 10)

    smalls = []
    for i, (xs, ys) in enumerate(strokes):
        if i == h:
            continue
        p, length, ends, diag = stroke_info(xs, ys, frame, rx)
        # separate ear stroke(s): any stroke with sharp peaks above the head ("^" or "M")
        if length > 0.2 and len(xs) >= 3:
            even = to_frame(*zip(*resample_stroke(xs, ys, 32)), frame)
            peaks = [j for j in find_peaks(even, closed=False, top_y=-0.6, w=3, k=2)
                     if abs(even[j, 0]) < 1.6]
            if peaks:
                for j in peaks:
                    ears.append((even[j, 0], even[j, 1], corner_angle(even, j, k=2)))
                labels[i] = "ear"
                continue
        # whisker: straight, long enough, reaches past the side of the head
        straight = length > 0 and ends / length > 0.85
        if straight and length > 0.25 and np.abs(p[:, 0]).max() > 0.9 and -0.5 < p[:, 1].mean() < 1.0:
            labels[i] = "whisker"
            continue
        if diag < 0.7:
            smalls.append((i, p, diag, is_closed(p, length) or diag < 0.25))

    row = {"head_aspect": (np.percentile(head[:, 1], 95) - y10) / 2}

    # ears: leftmost and rightmost peak
    ears.sort(key=lambda e: e[0])
    left = ears[0] if ears and ears[0][0] < 0 else None
    right = ears[-1] if ears and ears[-1][0] > 0 else None
    for side, ear in (("l", left), ("r", right)):
        row[f"ear_{side}_x"] = ear[0] if ear else np.nan
        row[f"ear_{side}_y"] = ear[1] if ear else np.nan
        row[f"ear_{side}_angle"] = ear[2] if ear else np.nan

    # whiskers
    whiskers = [i for i, lab in labels.items() if lab == "whisker"]
    tilts, wlens = [], []
    left_n = right_n = 0
    for i in whiskers:
        p, length, _, _ = stroke_info(*strokes[i], frame, rx)
        if p[:, 0].mean() < 0:
            left_n += 1
        else:
            right_n += 1
        dx, dy = p[-1] - p[0]
        tilts.append(math.degrees(math.atan2(abs(dy), abs(dx))))
        wlens.append(length)
    row["whisk_l"], row["whisk_r"] = left_n, right_n
    row["whisk_len"] = np.mean(wlens) if wlens else np.nan
    row["whisk_tilt"] = np.mean(tilts) if tilts else np.nan

    # nose: small CLOSED shape (or tiny mark) near the middle of the face, not low like a mouth
    noses = [(i, p, d) for i, p, d, closed_or_tiny in smalls
             if closed_or_tiny and abs(p[:, 0].mean()) < 0.5 and -0.3 < p[:, 1].mean() < 0.45]
    nose = min(noses, key=lambda s: np.hypot(s[1][:, 0].mean(), s[1][:, 1].mean() - 0.2), default=None)
    if nose:
        labels[nose[0]] = "nose"
        row["nose_x"], row["nose_y"] = nose[1].mean(axis=0)
        row["nose_size"], row["nose_round"] = nose[2], roundness(nose[1])
    else:
        row.update(nose_x=np.nan, nose_y=np.nan, nose_size=np.nan, nose_round=np.nan)

    # eyes: up to two small strokes in the upper half of the face
    eyes = [(i, p, d) for i, p, d, _ in smalls
            if labels[i] == "other" and np.abs(p[:, 0]).max() < 0.95 and -1.0 < p[:, 1].mean() < 0.3]
    eyes = sorted(eyes, key=lambda s: abs(s[1][:, 1].mean() + 0.25))[:2]
    for i, _, _ in eyes:
        labels[i] = "eye"
    if len(eyes) == 2:
        (_, a, da), (_, b, db) = eyes
        row["eye_gap"] = abs(a[:, 0].mean() - b[:, 0].mean())
        row["eye_y"] = (a[:, 1].mean() + b[:, 1].mean()) / 2
        row["eye_size"] = (da + db) / 2
    else:
        row.update(eye_gap=np.nan, eye_y=np.nan, eye_size=np.nan)

    return row, labels, ear_spans


COLORS = {"head": (0, 0, 0), "ear": (220, 0, 0), "whisker": (30, 90, 230),
          "nose": (0, 160, 60), "eye": (240, 140, 0), "other": (190, 190, 190)}


def draw_cat(pen, drawing, labels, ear_spans, ox, oy, scale):
    strokes = [s for s in drawing if len(s[0]) > 0]
    for i, (xs, ys) in enumerate(strokes):
        pts = [(ox + x * scale, oy + y * scale) for x, y in zip(xs, ys)]
        if len(pts) > 1:
            pen.line(pts, fill=COLORS[labels[i]], width=2)
        elif pts:
            pen.ellipse([pts[0][0] - 1, pts[0][1] - 1, pts[0][0] + 1, pts[0][1] + 1], fill=COLORS[labels[i]])
        if labels[i] == "head":  # paint ear peaks on the head line red
            res = resample_stroke(xs, ys, N_HEAD)
            for a, b in ear_spans:
                seg = [(ox + res[j % N_HEAD][0] * scale, oy + res[j % N_HEAD][1] * scale) for j in range(a, b + 1)]
                pen.line(seg, fill=COLORS["ear"], width=3)


def main():
    with open("data/cats_raw.ndjson") as f:
        cats = [json.loads(line)["drawing"] for line in f]

    rows, drawn = [], []
    for drawing in cats:
        row, labels, spans = find_parts(drawing)
        rows.append(row)
        drawn.append((drawing, labels, spans))
    df = pd.DataFrame(rows)
    df.round(3).to_csv("data/parts.csv", index=False)

    n = len(df)
    both_ears = df["ear_l_x"].notna() & df["ear_r_x"].notna()
    any_whisker = (df["whisk_l"] + df["whisk_r"]) > 0
    print(f"{n} cats")
    print(f"  both ears found   {both_ears.mean():.0%}")
    print(f"  whiskers found    {any_whisker.mean():.0%}")
    print(f"  nose found        {df['nose_x'].notna().mean():.0%}")
    print(f"  two eyes found    {df['eye_gap'].notna().mean():.0%}")
    print(f"  ears + whiskers   {(both_ears & any_whisker).mean():.0%}")
    face = both_ears & df["head_aspect"].between(0.6, 1.6)
    df["is_face"] = face
    df.round(3).to_csv("data/parts.csv", index=False)
    print(f"  face cats (both ears, head not too wide): {face.sum()} ({face.mean():.0%})")

    cols, rows_n, size = 8, 3, 128
    img = Image.new("RGB", (cols * size, rows_n * size), (255, 255, 255))
    pen = ImageDraw.Draw(img)
    for k, (drawing, labels, spans) in enumerate(drawn[: cols * rows_n]):
        ox, oy = (k % cols) * size, (k // cols) * size
        draw_cat(pen, drawing, labels, spans, ox, oy, size / 256)
        pen.rectangle([ox, oy, ox + size - 1, oy + size - 1], outline=(220, 220, 220))
    img.save("parts_grid.png")
    print("saved data/parts.csv and parts_grid.png "
          "(ears red, whiskers blue, nose green, eyes orange, head black, other grey)")


if __name__ == "__main__":
    main()
