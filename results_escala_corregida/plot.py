# explorar_resultados.py — corre esto primero
import numpy as np
from pathlib import Path
results_dir = Path("results")

# Ver qué hay en el npz
data = np.load(results_dir / "benchmark_hypervolumes.npz", allow_pickle=True)
print("Keys en npz:", list(data.keys()))
for k in data.keys():
    print(f"  {k}: {data[k]}")

# Ver qué hay en cada carpeta seed
for folder in sorted(Path("results").iterdir()):
    if folder.is_dir():
        print(f"\n{folder.name}:")
        for f in sorted(folder.iterdir()):
            print(f"  {f.name}")