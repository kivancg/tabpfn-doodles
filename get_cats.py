"""Step 1: download cat doodles from Google's Quick, Draw! dataset and look at them.

Data: https://github.com/googlecreativelab/quickdraw-dataset (CC BY 4.0)
Each line of the file is one drawing:
  {"word": "cat", "recognized": true, "drawing": [[[x...], [y...]], ...], ...}
"drawing" is a list of strokes; each stroke is [x values, y values], 0-255.
"""
import json
import statistics
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw

URL = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/cat.ndjson"
OUT = Path("data")
N_KEEP = 2000  # how many recognized cats to keep


def stream_cats(n):
    """Read the file line by line and stop once we have n recognized cats."""
    kept, seen = [], 0
    with urllib.request.urlopen(URL) as resp:
        for line in resp:
            seen += 1
            drawing = json.loads(line)
            if drawing["recognized"]:
                kept.append(drawing)
                if len(kept) == n:
                    break
    return kept, seen


def draw_grid(cats, path, cols=4, size=128):
    """Draw the first cols*cols cats in a grid, black lines on white."""
    scale = size / 256
    grid = Image.new("L", (cols * size, cols * size), 255)
    pen = ImageDraw.Draw(grid)
    for i, cat in enumerate(cats[: cols * cols]):
        ox, oy = (i % cols) * size, (i // cols) * size
        for xs, ys in cat["drawing"]:
            points = [(ox + x * scale, oy + y * scale) for x, y in zip(xs, ys)]
            if len(points) > 1:
                pen.line(points, fill=0, width=2)
        pen.rectangle([ox, oy, ox + size - 1, oy + size - 1], outline=200)
    grid.save(path)


def main():
    OUT.mkdir(exist_ok=True)
    cats, seen = stream_cats(N_KEEP)

    with open(OUT / "cats_raw.ndjson", "w") as f:
        for cat in cats:
            f.write(json.dumps(cat) + "\n")

    strokes = [len(c["drawing"]) for c in cats]
    points = [sum(len(xs) for xs, _ in c["drawing"]) for c in cats]
    few = sum(s <= 4 for s in strokes) / len(strokes)

    print(f"read {seen} drawings, kept {len(cats)} recognized cats")
    print(f"strokes per cat: median {statistics.median(strokes)}, max {max(strokes)}")
    print(f"cats with 4 strokes or fewer: {few:.0%}")
    print(f"points per cat:  median {statistics.median(points)}, max {max(points)}")

    draw_grid(cats, "cat_grid.png")
    print("saved data/cats_raw.ndjson and cat_grid.png")


if __name__ == "__main__":
    main()
