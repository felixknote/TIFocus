import os
import queue
import threading
import time
import numpy as np
import tifffile
from skimage.filters import laplace, sobel_h, sobel_v
import pandas as pd
from tqdm.auto import tqdm
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt

matplotlib.use("Agg")

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_DIRECTORIES = [
    r"E:\Data\2025_12_19 CRISPRi Reference Plate Imaging\P1",
    r"E:\Data\2025_12_19 CRISPRi Reference Plate Imaging\P2",
    r"E:\Data\2025_12_19 CRISPRi Reference Plate Imaging\P3",
    r"E:\Data\2025_12_19 CRISPRi Reference Plate Imaging\P4",
    r"E:\Data\2025_12_19 CRISPRi Reference Plate Imaging\P5",
    r"E:\Data\2025_12_19 CRISPRi Reference Plate Imaging\P6",
]

OUTPUT_SUBFOLDER = "TIFOCUS"
IO_THREADS       = 1   # HDD: 1 sequential reader beats N concurrent seekers
QUEUE_DEPTH      = 8   # stacks buffered between I/O and compute (~200 MB)
COMPUTE_WORKERS  = 16  # CPU threads for metrics + save


# ── I/O ────────────────────────────────────────────────────────────────────────

def _get_axis_order(tf):
    """Return the axes string from tifffile metadata, or None if unavailable."""
    try:
        return tf.series[0].axes  # e.g. 'ZYX', 'YXZ', 'CZYX', ...
    except Exception:
        return None


def load_tif_zstack(file_path):
    try:
        with tifffile.TiffFile(file_path) as tf:
            axes      = _get_axis_order(tf)
            img_stack = tf.asarray()

        if img_stack.size == 0:
            print(f"⚠️ Empty TIFF, skipping: {file_path}")
            return None, None
        if img_stack.dtype != np.uint16:
            raise ValueError(f"Expected uint16 TIFF, got {img_stack.dtype}: {file_path}")

        # Ensure Z is axis 0 ───────────────────────────────────────────────────
        if axes and 'Z' in axes:
            z_idx = axes.index('Z')
            if z_idx != 0:
                img_stack = np.moveaxis(img_stack, z_idx, 0)
        elif img_stack.ndim == 3 and img_stack.shape[-1] <= 5:
            # Fallback heuristic when TIFF has no axis metadata
            print(f"   ↕ axis-swap heuristic (no Z metadata, shape {img_stack.shape}): {file_path}")
            img_stack = np.moveaxis(img_stack, -1, 0)

        # float32 sufficient for 16-bit data; halves bandwidth vs float64
        float_stack = img_stack.astype(np.float32) / 65535.0
        return float_stack, img_stack
    except Exception as e:
        print(f"⚠️ Could not load {file_path}: {e}")
        return None, None


def save_best_slice(best_slice, idx, combined_scores, laplacian_vars, stem, output_directory):
    """
    Save a single pre-selected uint16 slice as a lossless 16-bit TIFF.
    best_slice and idx are resolved by the caller before releasing the queue slot.
    """
    if best_slice.dtype != np.uint16:
        raise TypeError(
            f"save_best_slice received {best_slice.dtype} — expected uint16. "
            "Pass original_stack[idx] from load_tif_zstack."
        )

    out_path = os.path.join(output_directory, f"{stem}_sharpest_z{idx:03d}.tif")
    # compression=None → uncompressed, lossless; photometric explicit to avoid version-dependent inference
    try:
        tifffile.imwrite(out_path, best_slice, compression=None, photometric='minisblack')
    except OSError:
        # Retry once — handles transient Windows file-lock (e.g. Explorer thumbnail COM surrogate)
        time.sleep(0.5)
        tifffile.imwrite(out_path, best_slice, compression=None, photometric='minisblack')

    return {
        "File":                   stem,
        "Max_Focus_Slice":        idx,
        "Max_Combined_Score":     combined_scores[idx],
        "Max_Laplacian_Variance": laplacian_vars[idx],
    }


# ── Focus metrics ──────────────────────────────────────────────────────────────

def laplacian_variance(image):
    """Primary focus metric (70%): classical Laplacian variance."""
    return np.var(laplace(image))


def sum_modified_laplacian(image):
    """Secondary focus metric (20%): Sum Modified Laplacian."""
    ly = np.abs(image[:-2, :] - 2 * image[1:-1, :] + image[2:, :])
    lx = np.abs(image[:, :-2] - 2 * image[:, 1:-1] + image[:, 2:])
    return np.sum(ly) + np.sum(lx)


def gradient_variance(image):
    """Tertiary focus metric (10%): gradient magnitude variance."""
    return np.var(np.sqrt(sobel_h(image) ** 2 + sobel_v(image) ** 2))


def compute_combined_focus_metric(float_stack, weights=(0.70, 0.20, 0.10)):
    """
    Compute weighted, normalised focus score for every z-slice.
    Returns (combined_scores, laplacian_vars).
    """
    laplacian_vars, sml_vals, gradvar_vals = [], [], []

    for z in range(float_stack.shape[0]):
        s = float_stack[z]
        laplacian_vars.append(laplacian_variance(s))
        sml_vals.append(sum_modified_laplacian(s))
        gradvar_vals.append(gradient_variance(s))

    def normalize(vec):
        vec = np.array(vec, dtype=np.float64)
        rng = vec.max() - vec.min()
        if rng == 0:
            print("⚠️  All z-scores identical — single-slice or uniformly flat stack.")
            return np.zeros_like(vec)
        return (vec - vec.min()) / rng

    combined = (
        weights[0] * normalize(laplacian_vars) +
        weights[1] * normalize(sml_vals) +
        weights[2] * normalize(gradvar_vals)
    )
    return combined, laplacian_vars


# ── Pipeline ───────────────────────────────────────────────────────────────────

def analyze_tif_stacks(data_directory, output_directory,
                        io_threads=IO_THREADS,
                        compute_workers=COMPUTE_WORKERS,
                        queue_depth=QUEUE_DEPTH):
    """
    Producer-consumer pipeline optimised for spinning HDD:
      - io_threads sequential readers fill a bounded queue (minimises seek thrashing)
      - compute_workers CPU threads drain the queue (metrics + save)
    """
    # sorted() encourages sequential on-disk access for the HDD
    tif_files = sorted(
        f for f in os.listdir(data_directory)
        if f.lower().endswith((".tif", ".tiff"))
    )
    if not tif_files:
        print("   No TIFF files found — skipping.")
        return pd.DataFrame()

    n = len(tif_files)
    print(f"   {n} files — {io_threads} I/O thread(s), {compute_workers} compute threads, "
          f"queue depth {queue_depth}.")

    # Divide file list evenly across I/O threads
    chunks    = [tif_files[i::io_threads] for i in range(io_threads)]
    work_q    = queue.Queue(maxsize=queue_depth)
    results   = []
    failed    = []
    lock      = threading.Lock()
    pbar      = tqdm(total=n, desc="   Processing", unit="file")

    # ── Producer: reads files sequentially, pushes (filename, stacks) ─────────
    def io_producer(file_chunk):
        for filename in file_chunk:
            float_stack, original_stack = load_tif_zstack(
                os.path.join(data_directory, filename)
            )
            work_q.put((filename, float_stack, original_stack))  # blocks when queue full

    # ── Consumer: compute metrics, save best slice ─────────────────────────────
    def compute_worker():
        while True:
            item = work_q.get()
            if item is None:
                break
            filename, float_stack, original_stack = item
            try:
                if float_stack is None or original_stack is None:
                    with lock:
                        failed.append(filename)
                    continue
                combined_scores, laplacian_vars = compute_combined_focus_metric(float_stack)
                del float_stack
                idx        = int(np.argmax(combined_scores))
                best_slice = original_stack[idx].copy()
                del original_stack
                # Include extension (. → _) to avoid collisions between .tif and .tiff inputs
                stem   = filename.replace(".", "_")
                result = save_best_slice(best_slice, idx, combined_scores, laplacian_vars,
                                         stem, output_directory)
                with lock:
                    results.append(result)
            except Exception as exc:
                print(f"\n⚠️  Error on {filename}: {exc}")
                with lock:
                    failed.append(filename)
            finally:
                pbar.update(1)

    # Start compute workers first so the queue drains immediately
    workers = [threading.Thread(target=compute_worker, daemon=True)
               for _ in range(compute_workers)]
    for w in workers:
        w.start()

    # Start I/O producers
    producers = [threading.Thread(target=io_producer, args=(chunk,), daemon=True)
                 for chunk in chunks]
    for p in producers:
        p.start()

    # Wait for all producers to finish, then signal workers to stop
    for p in producers:
        p.join()
    for _ in range(compute_workers):
        work_q.put(None)
    for w in workers:
        w.join()

    pbar.close()

    if failed:
        print(f"   ⚠️  {len(failed)} file(s) failed or skipped.")

    return pd.DataFrame(results).sort_values("File").reset_index(drop=True) if results else pd.DataFrame()


# ── Reporting ──────────────────────────────────────────────────────────────────

def save_summary(df, output_directory, folder_label):
    """Save CSV and a 4-panel summary plot for one folder's results."""
    csv_path = os.path.join(output_directory, "focus_summary.csv")
    df.to_csv(csv_path, index=False)

    slices  = df["Max_Focus_Slice"]
    scores  = df["Max_Combined_Score"]
    lapvar  = df["Max_Laplacian_Variance"]
    unique_slices = sorted(slices.unique())

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(folder_label, fontsize=15, fontweight="bold", y=0.98)

    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.32)
    ax_score  = fig.add_subplot(gs[0, 0])
    ax_lap    = fig.add_subplot(gs[0, 1])
    ax_hist   = fig.add_subplot(gs[0, 2])
    ax_box    = fig.add_subplot(gs[1, 0])
    ax_cdf    = fig.add_subplot(gs[1, 1])
    ax_joint  = fig.add_subplot(gs[1, 2])

    # ── Combined score vs slice index ─────────────────────────────────────────
    sc = ax_score.scatter(slices, scores, c=scores, cmap="viridis",
                          alpha=0.55, edgecolors="none", s=18)
    fig.colorbar(sc, ax=ax_score, label="Combined score")
    ax_score.set_xlabel("Best focus slice"); ax_score.set_ylabel("Combined focus score")
    ax_score.set_title("Score vs slice", fontweight="bold")
    ax_score.grid(True, linestyle="--", alpha=0.4)

    # ── Laplacian variance vs slice index ─────────────────────────────────────
    sc2 = ax_lap.scatter(slices, lapvar, c=lapvar, cmap="plasma",
                         alpha=0.55, edgecolors="none", s=18)
    fig.colorbar(sc2, ax=ax_lap, label="Laplacian variance")
    ax_lap.set_xlabel("Best focus slice"); ax_lap.set_ylabel("Laplacian variance")
    ax_lap.set_title("Laplacian variance vs slice (70% weight)", fontweight="bold")
    ax_lap.grid(True, linestyle="--", alpha=0.4)

    # ── Slice index histogram ─────────────────────────────────────────────────
    bins = [s - 0.5 for s in unique_slices] + [unique_slices[-1] + 0.5]
    ax_hist.hist(slices, bins=bins, color="#4C72B0", edgecolor="white", linewidth=0.6)
    ax_hist.set_xlabel("Best focus slice"); ax_hist.set_ylabel("Count")
    ax_hist.set_title("Slice selection distribution", fontweight="bold")
    ax_hist.set_xticks(unique_slices)
    ax_hist.grid(True, axis="y", linestyle="--", alpha=0.4)

    # ── Laplacian variance boxplot per slice ──────────────────────────────────
    data_by_slice = [df.loc[slices == s, "Max_Laplacian_Variance"].values
                     for s in unique_slices]
    bp = ax_box.boxplot(data_by_slice, labels=unique_slices, patch_artist=True,
                        medianprops=dict(color="black", linewidth=1.5))
    colors = cm.get_cmap("viridis")(np.linspace(0.2, 0.8, len(unique_slices)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set(facecolor=color, alpha=0.7)
    ax_box.set_xlabel("Best focus slice"); ax_box.set_ylabel("Laplacian variance")
    ax_box.set_title("Laplacian variance per slice", fontweight="bold")
    ax_box.grid(True, axis="y", linestyle="--", alpha=0.4)

    # ── CDF of combined score ─────────────────────────────────────────────────
    sorted_scores = np.sort(scores)
    cdf = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    ax_cdf.plot(sorted_scores, cdf, color="#DD8452", linewidth=1.8)
    ax_cdf.axvline(scores.median(), color="grey", linestyle="--", linewidth=1,
                   label=f"Median {scores.median():.3f}")
    ax_cdf.set_xlabel("Combined focus score"); ax_cdf.set_ylabel("CDF")
    ax_cdf.set_title("Score CDF", fontweight="bold")
    ax_cdf.legend(fontsize=8)
    ax_cdf.grid(True, linestyle="--", alpha=0.4)

    # ── Score vs Laplacian variance joint scatter ─────────────────────────────
    sc3 = ax_joint.scatter(lapvar, scores, c=slices, cmap="tab10",
                           alpha=0.5, edgecolors="none", s=18)
    fig.colorbar(sc3, ax=ax_joint, label="Slice index")
    ax_joint.set_xlabel("Laplacian variance"); ax_joint.set_ylabel("Combined score")
    ax_joint.set_title("Score vs Laplacian (coloured by slice)", fontweight="bold")
    ax_joint.grid(True, linestyle="--", alpha=0.4)

    plot_path = os.path.join(output_directory, "focus_summary_plot.png")
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"   ✅ CSV  → {csv_path}")
    print(f"   ✅ Plot → {plot_path}")
    print(f"   Stats: mean slice {slices.mean():.1f} | "
          f"range {slices.min()}–{slices.max()} | "
          f"LaplacianVar {lapvar.min():.2e}–{lapvar.max():.2e}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    all_results = []

    for i, data_dir in enumerate(DATA_DIRECTORIES, 1):
        folder_label = os.path.basename(data_dir)
        print(f"\n[{i}/{len(DATA_DIRECTORIES)}] {folder_label}")
        print(f"  Input  : {data_dir}")

        if not os.path.isdir(data_dir):
            print(f"  ⚠️  Directory not found — skipping.")
            continue

        output_dir = os.path.join(data_dir, OUTPUT_SUBFOLDER)
        os.makedirs(output_dir, exist_ok=True)
        print(f"  Output : {output_dir}")

        df = analyze_tif_stacks(data_dir, output_dir)

        if df.empty:
            print("  No results for this folder.")
            continue

        df.insert(0, "Folder", folder_label)
        all_results.append(df)

        save_summary(df, output_dir, folder_label)

    # ── Combined summary across all folders ───────────────────────────────────
    if len(all_results) > 1:
        combined_dir = os.path.dirname(DATA_DIRECTORIES[0])
        combined_csv = os.path.join(combined_dir, "focus_summary_ALL.csv")
        combined_df  = pd.concat(all_results, ignore_index=True)
        combined_df.to_csv(combined_csv, index=False)
        print(f"\n✅ Combined CSV ({len(combined_df)} files) → {combined_csv}")

    print("\n✅ All folders complete.")


if __name__ == "__main__":
    main()
