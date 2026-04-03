import os
import csv
import time
import multiprocessing.util
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Suppress NFS temp-file cleanup errors
_orig_rmtree = multiprocessing.util._remove_temp_dir
def _quiet_remove_temp_dir(*args, **kwargs):
    try:
        _orig_rmtree(*args, **kwargs)
    except OSError:
        pass
multiprocessing.util._remove_temp_dir = _quiet_remove_temp_dir

from mia_em_loader import CellMapDataset3D, CropDatabase, discover_crops

CLASSES = ["ecs", "cell", "nuc", "mito", "er", "golgi"]

DATA_ROOT = "/nrs/cellmap/data"
NORMS_CSV = "examples/norms.csv"
CROPS_JSON = "crops.json"

NUM_BATCHES = 50
BATCH_SIZE = 4
WORKERS = [1, 10, 30, 62]

OUT_DIR = "benchmark_results"
os.makedirs(OUT_DIR, exist_ok=True)

# --- Load crop database ---
if os.path.exists(CROPS_JSON):
    db = CropDatabase.from_json(CROPS_JSON)
else:
    db = discover_crops(
        data_root=DATA_ROOT,
        norms_csv=NORMS_CSV,
        target_classes=CLASSES,
        target_resolution=8.0,
    )
    db.to_json(CROPS_JSON)

ds = CellMapDataset3D(
    crop_db=db,
    target_classes=CLASSES,
    target_resolution=8.0,
    input_size=(128, 128, 128),
    samples_per_epoch=NUM_BATCHES * BATCH_SIZE,
)

print(f"Benchmarking: {NUM_BATCHES} batches, batch_size={BATCH_SIZE}, input_size=128^3")
print(f"Workers: {WORKERS}\n")

def meta_collate(batch):
    """Collate tensors normally, keep meta dicts as a list."""
    from torch.utils.data.dataloader import default_collate
    tensors = default_collate([item[:4] for item in batch])
    metas = [item[4] for item in batch]
    return (*tensors, metas)


results = {}
for nw in WORKERS:
    loader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=nw, collate_fn=meta_collate)
    it = iter(loader)

    # warmup 2 batches
    for _ in range(2):
        next(it)

    t0 = time.perf_counter()
    count = 0
    for batch in it:
        count += 1
        if count >= NUM_BATCHES:
            break
    elapsed = time.perf_counter() - t0

    batches_per_sec = count / elapsed
    samples_per_sec = count * BATCH_SIZE / elapsed
    results[nw] = (elapsed, batches_per_sec, samples_per_sec)
    print(f"workers={nw:>2}  |  {elapsed:6.1f}s  |  {batches_per_sec:5.1f} batch/s  |  {samples_per_sec:6.1f} sample/s")

    # Clean up workers before next iteration (avoids NFS temp-file errors)
    del it, loader

# --- Save CSV ---
csv_path = os.path.join(OUT_DIR, "benchmark_workers.csv")
base_time = results[WORKERS[0]][0]
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["workers", "time_s", "batch_per_s", "sample_per_s", "speedup"])
    for nw in WORKERS:
        elapsed, bps, sps = results[nw]
        writer.writerow([nw, f"{elapsed:.2f}", f"{bps:.2f}", f"{sps:.2f}", f"{base_time / elapsed:.2f}"])
print(f"\nCSV saved to {csv_path}")

# --- Plot ---
workers_list = WORKERS
samples_per_sec = [results[nw][2] for nw in workers_list]
speedups = [base_time / results[nw][0] for nw in workers_list]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.plot(workers_list, samples_per_sec, "o-", color="tab:blue", linewidth=2)
ax1.set_xlabel("Number of workers")
ax1.set_ylabel("Samples / sec")
ax1.set_title("Throughput")
ax1.set_xticks(workers_list)
ax1.grid(True, alpha=0.3)

ax2.plot(workers_list, speedups, "o-", color="tab:orange", linewidth=2)
ax2.axline((0, 0), slope=1, color="gray", linestyle="--", alpha=0.5, label="ideal")
ax2.set_xlabel("Number of workers")
ax2.set_ylabel(f"Speedup vs {WORKERS[0]} worker")
ax2.set_title("Scaling")
ax2.set_xticks(workers_list)
ax2.legend()
ax2.grid(True, alpha=0.3)

fig.suptitle(f"DataLoader benchmark (batch_size={BATCH_SIZE}, input=128³, {NUM_BATCHES} batches)", fontsize=11)
plt.tight_layout()

png_path = os.path.join(OUT_DIR, "benchmark_workers.png")
fig.savefig(png_path, dpi=150)
print(f"Plot saved to {png_path}")
