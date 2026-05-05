# replot_final.py
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

results_dir = Path("results")
seeds = [1000, 1001]

# --- Cargar datos ---
pcn_fronts, pcn_logs = [], []

for seed in seeds:
    folder = results_dir / f"benchmark_seed_{seed}"
    last_ckpt = sorted(folder.glob("checkpoint_iter*.pkl"))[-1]
    with open(last_ckpt, "rb") as f:
        data = pickle.load(f)
    pts = np.array(data["extra"]["pareto_front"]) * 100
    pcn_fronts.append(pts)
    df = pd.read_csv(folder / "training_log.csv")
    df["best_obj0_scaled"] = df["best_obj0"] * 100
    df["best_obj1_scaled"] = df["best_obj1"] * 100
    pcn_logs.append(df)

hv_data   = np.load(results_dir / "benchmark_hypervolumes.npz", allow_pickle=True)
pcn_hvs   = hv_data["pcn"]    * 10000
pgmorl_hvs = hv_data["pgmorl"]

print("PCN HV rescalado:   ", pcn_hvs)
print("PGMORL HV original: ", pgmorl_hvs)

# --- Figura 3 paneles ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Panel 1 — Pareto fronts finales
ax = axes[0]
colors = ["steelblue", "royalblue"]
for i, (pts, seed) in enumerate(zip(pcn_fronts, seeds)):
    pts_sorted = pts[np.argsort(pts[:, 0])]
    ax.plot(pts_sorted[:, 0], pts_sorted[:, 1], "o-",
            color=colors[i], alpha=0.8, label=f"PCN seed {seed}")
ax.set_xlabel("Objective 0  (reward_run)")
ax.set_ylabel("Objective 1  (reward_ctrl)")
ax.set_title("Pareto fronts finales — PCN")
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 2 — Evolución del mejor punto durante entrenamiento
ax = axes[1]
for i, (df, seed) in enumerate(zip(pcn_logs, seeds)):
    ax.plot(df["iteration"], df["best_obj0_scaled"],
            color=colors[i], label=f"seed {seed} — obj0 (run)")
    ax.plot(df["iteration"], df["best_obj1_scaled"],
            color=colors[i], linestyle="--", alpha=0.6, label=f"seed {seed} — obj1 (ctrl)")
ax.set_xlabel("Iteración")
ax.set_ylabel("Mejor retorno (escala original)")
ax.set_title("Evolución del mejor punto")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 3 — Boxplot HV comparativo
ax = axes[2]
bp = ax.boxplot(
    [pcn_hvs, pgmorl_hvs],
    labels=["PCN (ours)", "PGMORL"],
    patch_artist=True,
    medianprops=dict(color="orange", linewidth=2),
)
bp["boxes"][0].set_facecolor("steelblue")
bp["boxes"][0].set_alpha(0.5)
bp["boxes"][1].set_facecolor("orange")
bp["boxes"][1].set_alpha(0.3)
for i, hvs in enumerate([pcn_hvs, pgmorl_hvs], 1):
    ax.scatter(i, np.mean(hvs), marker="^", color="green", zorder=5, s=80)
ax.set_ylabel("Hypervolume")
ax.set_title("PCN vs PGMORL — HV (escala unificada)")
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = results_dir / "replot_final.png"
plt.savefig(out, dpi=150)
plt.show()
print(f"Guardado en {out}")