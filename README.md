# TIFocus

A tool to analyze TIFF z-stacks in order to extract the sharpest frame and perform quality control for high-throughput microscopy imaging. TIFocus uses a weighted combination of three focus metrics to robustly identify the best focal plane across large image datasets.

Experiments have shown that Laplacian-based operators have the best overall performance under normal imaging conditions.
Reference: https://doi.org/10.1016/j.patcog.2012.11.011

---

## Features

- Multi-metric focus scoring with configurable weights
- Parallel processing via multi-threaded execution
- Memory-safe concurrent loading via semaphore-controlled throughput
- 16-bit TIFF output preserving original bit depth
- Multi-folder sequential batch processing
- Per-folder and combined CSV summaries
- Summary scatter plots for QC review

---

## Focus Metrics

TIFocus combines three complementary sharpness estimators:

| Metric | Weight | Description |
|---|---|---|
| Laplacian Variance | 70% | Variance of the Laplacian-filtered image. Primary metric. |
| Sum Modified Laplacian | 20% | Sum of absolute second-order differences along both axes. |
| Gradient Variance | 10% | Variance of the Sobel gradient magnitude. |

Each metric is independently normalised to [0, 1] before weighting, so differences in absolute scale across metrics do not bias the result.

---

## Requirements

```
numpy
scikit-image
pandas
matplotlib
tqdm
```

Install with:

```bash
pip install numpy scikit-image pandas matplotlib tqdm
```

---

## Usage

### 1. Configure folders and parameters

At the top of the notebook, edit the `DATA_DIRECTORIES` list and performance parameters:

```python
DATA_DIRECTORIES = [
    r"E:\Data\Experiment_01\P1",
    r"E:\Data\Experiment_01\P2",
]

OUTPUT_SUBFOLDER     = "TIFOCUS Z-Extract"
MAX_WORKERS          = 18   # parallel threads
MAX_CONCURRENT_LOADS = 18   # max z-stacks in RAM simultaneously
```

### 2. Run the notebook

Execute all cells. Each folder is processed sequentially; parallelism runs within each folder.

### 3. Outputs

For each input folder, a subfolder named `TIFOCUS Z-Extract` is created containing:

| File | Description |
|---|---|
| `{filename}_sharpest_z{N}.tif` | Best-focus slice saved as 16-bit TIFF |
| `focus_summary.csv` | Per-file results: best slice index, combined score, Laplacian variance |
| `focus_summary_plot.png` | Scatter plots of score distributions for QC |

If multiple folders are provided, a combined `focus_summary_ALL.csv` is also written one level above the plate folders.

---

## Parameter Tuning

### `MAX_WORKERS`
Controls CPU parallelism. A good starting point is `logical_core_count - 2`.

### `MAX_CONCURRENT_LOADS`
Caps the number of z-stacks held in RAM simultaneously. To size this:

```
MAX_CONCURRENT_LOADS = floor(available_RAM_GB / stack_size_GB)
```

For small stacks (≤ 50 MB), set `MAX_WORKERS` and `MAX_CONCURRENT_LOADS` equal — there is no benefit to throttling loads.

### Example: Xeon W5-2565 (10 cores / 20 threads), 42 MB stacks

```python
MAX_WORKERS          = 18
MAX_CONCURRENT_LOADS = 18
```

---

## Output Format

All extracted slices are saved as **unsigned 16-bit TIFF**. If the source image is not already `uint16`, it is converted via a safe float-normalised rescale using `skimage.img_as_uint` — no clipping occurs.

---

## Notes

- TIFocus uses `ThreadPoolExecutor` rather than `ProcessPoolExecutor`. NumPy and scikit-image release the GIL during array operations, giving real CPU parallelism without the process-spawn overhead and pickle fragility that causes crashes in Jupyter on Windows.
- Low-contrast images are silently skipped (suppressed `UserWarning`). Failed files are collected and printed at the end of each folder run.
