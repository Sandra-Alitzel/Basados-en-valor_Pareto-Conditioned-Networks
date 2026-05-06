import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

results_dir = Path("results")
seeds = [1000, 1001]
ref_point = np.array([0.0, -500.0])

# función hypervolume 2D inline (no necesita import)
def hypervolume_2d(front, ref):
    mask = (front[:, 0] > ref[0]) & (front[:, 1] > ref[1])
    pts = front[mask]
    if pts.size == 0:
        return 0.0
    order = np.argsort(-pts[:, 0])
    pts = pts[order]
    hv = 0.0
    prev_y = ref[1]
    for x, y in pts:
        if y <= prev_y:
            continue
        hv += (x - ref[0]) * (y - prev_y)
        prev_y = y
    return float(hv)

pcn_hvs_fix = []
for seed in seeds:
    folder = results_dir / f"benchmark_seed_{seed}"
    last_ckpt = sorted(folder.glob("checkpoint_iter*.pkl"))[-1]
    with open(last_ckpt, "rb") as f:
        data = pickle.load(f)
    pts = np.array(data["extra"]["pareto_front"]) * 100
    hv = hypervolume_2d(pts, ref_point)
    pcn_hvs_fix.append(hv)
    print(f"Seed {seed} — puntos: {len(pts)}, HV: {hv:.1f}")

pgmorl_hvs = np.array([67723.91, 66465.73])
print(f"\nPCN media:    {np.mean(pcn_hvs_fix):.1f}")
print(f"PGMORL media: {np.mean(pgmorl_hvs):.1f}")

# Boxplot
fig, ax = plt.subplots(figsize=(7, 5))

bp = ax.boxplot(
    [pcn_hvs_fix, pgmorl_hvs],
    labels=["PCN (ours)", "PGMORL"],
    patch_artist=True,
    medianprops=dict(color="orange", linewidth=2),
    widths=0.4,
)
bp["boxes"][0].set_facecolor("steelblue")
bp["boxes"][0].set_alpha(0.5)
bp["boxes"][1].set_facecolor("orange")
bp["boxes"][1].set_alpha(0.3)

# puntos individuales encima del boxplot
for i, hvs in enumerate([pcn_hvs_fix, pgmorl_hvs], 1):
    ax.scatter([i]*len(hvs), hvs, color="steelblue" if i==1 else "orange",
               zorder=5, s=60, alpha=0.8)
    ax.scatter(i, np.mean(hvs), marker="^", color="green", zorder=6, s=100)

ax.set_ylabel("Hypervolume")
ax.set_title("PCN vs PGMORL — HV (misma escala, mismo ref_point)")
ax.grid(True, alpha=0.3, axis="y")  # solo grid horizontal
ax.set_xlim(0.3, 2.7)

# anotar valores
for i, hvs in enumerate([pcn_hvs_fix, pgmorl_hvs], 1):
    for j, v in enumerate(hvs):
        ax.annotate(f"{v:.0f}", xy=(i, v), xytext=(8, 0),
                    textcoords="offset points", fontsize=9, va="center")

plt.tight_layout()
plt.savefig(results_dir / "boxplot_hv_corregido.png", dpi=150, facecolor="white")
plt.show()