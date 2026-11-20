"""Step 2: turn each cat doodle into one table row of 40 points.

Each point is (x, y, pen):
  pen = 1  -> pen was lifted: jump to this point without drawing (start of a stroke)
  pen = 0  -> pen is down: draw a line from the previous point to this one

Points are shared out between strokes: every stroke gets at least 2 points
(so it can still be drawn), the rest go to strokes in proportion to their length.
Cats with too many strokes to fit (more than 20) are skipped.

Writes data/cats_table.csv (one row per cat, 120 columns) and
check_grid.png (top row: original cats, bottom row: the same cats rebuilt from the table,
with the first stroke in red).

Strokes are sorted longest first, so the first stroke is usually the head.
"""
import csv
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw

N_POINTS = 64
MIN_PER_STROKE = 2
DATA = Path("data")


def stroke_length(xs, ys):
    return sum(math.dist((xs[i], ys[i]), (xs[i + 1], ys[i + 1])) for i in range(len(xs) - 1))


def resample_stroke(xs, ys, k):
    """Return k points spaced evenly along one stroke."""
    cum = [0.0]
    for i in range(len(xs) - 1):
        cum.append(cum[-1] + math.dist((xs[i], ys[i]), (xs[i + 1], ys[i + 1])))
    total = cum[-1]
    if total == 0:  # a dot: repeat the same point
        return [(xs[0], ys[0])] * k
    out, seg = [], 0
    for j in range(k):
        target = total * j / (k - 1)
        while seg < len(cum) - 2 and cum[seg + 1] < target:
            seg += 1
        span = cum[seg + 1] - cum[seg]
        t = 0.0 if span == 0 else (target - cum[seg]) / span
        out.append((xs[seg] + t * (xs[seg + 1] - xs[seg]), ys[seg] + t * (ys[seg + 1] - ys[seg])))
    return out


def share_points(lengths, n_points):
    """How many points each stroke gets: MIN_PER_STROKE each, the rest by length."""
    n = len(lengths)
    extra = n_points - MIN_PER_STROKE * n
    total = sum(lengths) or 1.0
    exact = [extra * length / total for length in lengths]
    counts = [MIN_PER_STROKE + int(e) for e in exact]
    # hand out the leftover points to the strokes with the biggest remainders
    leftover = n_points - sum(counts)
    order = sorted(range(n), key=lambda i: exact[i] - int(exact[i]), reverse=True)
    for i in order[:leftover]:
        counts[i] += 1
    return counts


def cat_to_row(drawing):
    """One cat -> list of 40 (x, y, pen) points, or None if it has too many strokes."""
    strokes = [(xs, ys) for xs, ys in drawing if len(xs) > 0]
    # same order for every cat: longest stroke first (usually the head),
    # so column x0 means roughly the same thing in every row
    strokes.sort(key=lambda s: stroke_length(*s), reverse=True)
    if len(strokes) * MIN_PER_STROKE > N_POINTS:
        return None
    counts = share_points([stroke_length(xs, ys) for xs, ys in strokes], N_POINTS)
    points = []
    for (xs, ys), k in zip(strokes, counts):
        for j, (x, y) in enumerate(resample_stroke(xs, ys, k)):
            points.append((round(x, 1), round(y, 1), 1 if j == 0 else 0))
    return points


def draw_points(pen, points, ox, oy, scale):
    """Draw a list of (x, y, pen) points. The first stroke is drawn in red."""
    stroke = 0
    for (x0, y0, _), (x1, y1, up) in zip(points, points[1:]):
        if up:
            stroke += 1
            continue
        color = (220, 0, 0) if stroke == 0 else (0, 0, 0)
        pen.line([(ox + x0 * scale, oy + y0 * scale), (ox + x1 * scale, oy + y1 * scale)], fill=color, width=2)


def draw_original(pen, drawing, ox, oy, scale):
    for xs, ys in drawing:
        pts = [(ox + x * scale, oy + y * scale) for x, y in zip(xs, ys)]
        if len(pts) > 1:
            pen.line(pts, fill=(0, 0, 0), width=2)


def check_grid(originals, rows, path, n=8, size=128):
    """Top row: original cats. Bottom row: same cats rebuilt from the table."""
    scale = size / 256
    img = Image.new("RGB", (n * size, 2 * size), (255, 255, 255))
    pen = ImageDraw.Draw(img)
    for i in range(min(n, len(rows))):
        draw_original(pen, originals[i], i * size, 0, scale)
        draw_points(pen, rows[i], i * size, size, scale)
        pen.rectangle([i * size, 0, i * size + size - 1, size - 1], outline=(200, 200, 200))
        pen.rectangle([i * size, size, i * size + size - 1, 2 * size - 1], outline=(200, 200, 200))
    img.save(path)


def main():
    with open(DATA / "cats_raw.ndjson") as f:
        cats = [json.loads(line) for line in f]

    rows, originals, skipped = [], [], 0
    for cat in cats:
        row = cat_to_row(cat["drawing"])
        if row is None:
            skipped += 1
            continue
        rows.append(row)
        originals.append(cat["drawing"])

    header = [f"{name}{i}" for i in range(N_POINTS) for name in ("x", "y", "pen")]
    with open(DATA / "cats_table.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow([value for point in row for value in point])

    check_grid(originals, rows, "check_grid.png")
    print(f"{len(rows)} cats -> data/cats_table.csv ({len(header)} columns)")
    print(f"skipped {skipped} cats with more than {N_POINTS // MIN_PER_STROKE} strokes")
    print("saved check_grid.png (top: original, bottom: rebuilt from the table)")


if __name__ == "__main__":
    main()
