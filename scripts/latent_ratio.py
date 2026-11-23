from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "outputs"
EXPERIMENTS = ["latent_n300", "latent_n900"]
OUT = ROOT / "latent"
LEVEL = 2

FAMILY = {"latent_n300": (plt.cm.Blues, "s", (0, (4, 2))),
          "latent_n900": (plt.cm.Oranges, "o", "-")}
STEPS = [0.45, 0.62, 0.79, 0.95]
MUTED, GRID, AXIS, SURFACE = "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def load(exp):
    n = int(re.search(r"n(\d+)$", exp).group(1))
    d = pd.read_csv(ROOT / exp / f"lambda_sweep_l{LEVEL}.csv")
    quantiles = sorted(float(c.removeprefix("lambda_q"))
                       for c in d.columns if c.startswith("lambda_q"))
    null = d[f"null_p50_n{n}"].to_numpy()
    return n, d["mode"].to_numpy(), {q: d[f"lambda_q{q:g}"].to_numpy() / null
                                     for q in quantiles}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)

    plt.rc("font", family="serif", size=20 * 8 / 13.5)
    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    table = {}
    for exp in EXPERIMENTS:
        n, modes, ratios = load(exp)
        table["mode"] = modes
        cmap, marker, ls = FAMILY[exp]
        for (q, r), t in zip(ratios.items(), STEPS):
            ax.plot(modes, r, color=cmap(t), marker=marker, ms=4.5, lw=1.8, ls=ls,
                    mec=SURFACE, mew=0.8, zorder=3, label=f"n={n}  q={q:g}")
            table[f"ratio_n{n}_q{q:g}"] = r

    ax.set_xlabel("mode (ascending λ)")
    ax.set_ylabel("λ / null median")
    ax.set_xticks(modes)
    ax.legend(frameon=False, ncol=2, loc="upper left")

    fig.tight_layout()
    png = OUT / f"lambda_null_ratio_l{LEVEL}.png"
    fig.savefig(png, facecolor=fig.get_facecolor(), dpi=300, bbox_inches="tight")
    pd.DataFrame(table).to_csv(OUT / f"lambda_null_ratio_l{LEVEL}.csv", index=False)
    print(f"wrote {png}")
