# TIFocus

Automated best-focus slice extraction from 16-bit microscopy z-stack TIFFs. Designed for high-throughput plate imaging where each field of view is acquired as a short z-stack and a single in-focus plane is needed for downstream analysis.

## How it works

Each z-stack is scored per slice using a weighted combination of three focus metrics:

| Metric | Weight | Description |
|--------|--------|-------------|
| Laplacian variance | 70% | Primary sharpness estimator — highest overall performance at normal imaging conditions ([Pertuz et al. 2013](https://doi.org/10.1016/j.patcog.2012.11.011)) |
| Sum Modified Laplacian | 20% | Suppresses wrap-around border artifacts via array slicing |
| Gradient magnitude variance | 10% | Captures edge energy complementary to Laplacian |

Scores are min-max normalised per stack and combined. The slice with the highest combined score is saved as a lossless uncompressed 16-bit TIFF. The original pixel values are preserved exactly — no rescaling, no normalisation, no compression.

## Features

- **Lossless 16-bit output** — `tifffile.imwrite` with `compression=None`; dtype enforced at load and save
- **Metadata-aware axis detection** — reads `series[0].axes` from the TIFF header to identify the Z axis; falls back to a shape heuristic only when metadata is absent
- **Producer-consumer pipeline** — 1 sequential I/O thread minimises HDD seek thrashing; configurable compute thread pool for CPU-bound metric computation
- **float32 metric computation** — 16-bit source data normalised to float32 (not float64) for metric computation, halving memory bandwidth
- **Per-folder CSV + 6-panel summary plot** — slice distribution, score CDF, per-slice boxplots, joint scatter
- **Combined CSV** across all folders

## Output

For each input TIFF `<name>.tif` the sharpest slice is written to:

```
<input_folder>/TIFOCUS/<name>_tif_sharpest_z<NNN>.tif
```

Per-folder summary files:
```
<input_folder>/TIFOCUS/focus_summary.csv
<input_folder>/TIFOCUS/focus_summary_plot.png
```

Combined summary across all folders:
```
<parent_of_first_folder>/focus_summary_ALL.csv
```

## Requirements

```
numpy
tifffile
scikit-image
pandas
tqdm
matplotlib
```

Install into your environment:

```bash
pip install numpy tifffile scikit-image pandas tqdm matplotlib
```

## Configuration

Edit the top of `TIFocus.py`:

```python
DATA_DIRECTORIES = [
    r"E:\Data\experiment\P1",
    r"E:\Data\experiment\P2",
    # ...
]

OUTPUT_SUBFOLDER = "TIFOCUS"   # created inside each input folder
IO_THREADS       = 1           # keep at 1 for HDD; raise to 2–4 for NVMe
QUEUE_DEPTH      = 8           # stacks buffered between I/O and compute
COMPUTE_WORKERS  = 16          # CPU threads for metrics + save
```

**Input requirements:** uint16 grayscale TIFF z-stacks. Non-uint16 files are rejected with a clear error message rather than silently converted.

## Usage

```bash
python TIFocus.py
```

## Notes

- On spinning HDDs, a single sequential I/O thread outperforms concurrent readers because random seeks dominate latency
- On NVMe SSDs, raise `IO_THREADS` to 2–4 and `COMPUTE_WORKERS` to match your core count
- A one-time retry on `OSError` during save handles transient Windows file locks (e.g. Explorer thumbnail generation)
