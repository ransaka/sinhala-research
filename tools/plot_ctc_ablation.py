#!/usr/bin/env python3
"""Compare saved real-audio CTC overfit diagnostics."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


RUNS = [
    ("Original", Path("outputs/ctc-overfit-4x200")),
    ("No SpecAugment", Path("outputs/ctc-overfit-4x200-no-spec-augment")),
    ("No dropout/layerdrop", Path("outputs/ctc-overfit-4x200-no-dropout")),
    ("No regularization", Path("outputs/ctc-overfit-4x200-no-regularization")),
]


def main() -> None:
    destination = Path("outputs/ctc-overfit-comparison")
    destination.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for label, run in RUNS:
        config = json.loads((run / "run_config.json").read_text())
        losses = config["train_loss_by_epoch"]
        ax.plot(range(2, 2 * len(losses) + 1, 2), losses, label=label, linewidth=1.3)
    ax.set(xlabel="Optimizer updates", ylabel="CTC loss (mean per target code point)",
           title="Four real utterances, same data and optimizer settings")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.savefig(destination / "loss_comparison.png", dpi=180)
    plt.close(fig)

    tile_width, tile_height = 960, 400
    canvas = Image.new("RGB", (2 * tile_width, 2 * (tile_height + 36)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 22)
    for position, (label, run) in enumerate(RUNS):
        row, column = divmod(position, 2)
        x, y = column * tile_width, row * (tile_height + 36)
        draw.text((x + 12, y + 6), label, fill="black", font=font)
        image = Image.open(run / "diagnostics/train_01_posterior.png").convert("RGB")
        image = image.resize((tile_width, tile_height), Image.Resampling.LANCZOS)
        canvas.paste(image, (x, y + 36))
    canvas.save(destination / "posterior_comparison.png")

    progression = [
        ("Original, 200 updates", Path("outputs/ctc-overfit-4x200")),
        ("No regularization, 200 updates", Path("outputs/ctc-overfit-4x200-no-regularization")),
        ("No regularization, 500 total updates", Path("outputs/ctc-overfit-4-resumed-300")),
    ]
    progression_canvas = Image.new("RGB", (tile_width, len(progression) * (tile_height + 36)), "white")
    progression_draw = ImageDraw.Draw(progression_canvas)
    for position, (label, run) in enumerate(progression):
        y = position * (tile_height + 36)
        progression_draw.text((12, y + 6), label, fill="black", font=font)
        image = Image.open(run / "diagnostics/train_01_posterior.png").convert("RGB")
        progression_canvas.paste(image.resize((tile_width, tile_height), Image.Resampling.LANCZOS), (0, y + 36))
    progression_canvas.save(destination / "posterior_progression.png")

    first = json.loads((progression[1][1] / "run_config.json").read_text())["train_loss_by_epoch"]
    resumed = json.loads((progression[2][1] / "run_config.json").read_text())["train_loss_by_epoch"]
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.plot(range(2, 2 * len(first) + 1, 2), first, label="First 200 updates")
    ax.plot(range(202, 200 + 2 * len(resumed) + 1, 2), resumed, label="300 resumed updates")
    ax.axvline(200, linestyle="--", color="gray", linewidth=1, label="Optimizer reset")
    ax.set(xlabel="Total model-weight updates", ylabel="CTC loss (mean per target code point)",
           title="Four real utterances: no-regularization overfit progression")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.savefig(destination / "loss_progression.png", dpi=180)
    plt.close(fig)
    print(destination)


if __name__ == "__main__":
    main()
