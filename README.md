# GPU-Accelerated QX Test for Population Genetics

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![R](https://img.shields.io/badge/R-4.0%2B-blue)](https://www.r-project.org/)
[![CUDA 13](https://img.shields.io/badge/CUDA-13.x-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)



---
# **Version 1** Fast Pairwise Qₓ Statistic (CPU + GPU)

**Fast implementation of the pairwise Qₓ polygenic selection statistic** introduced by Berg & Coop (2014) for detecting directional polygenic adaptation between population pairs. Achieves **13-31x speedup** over CPU baseline using NVIDIA GPU acceleration with CuPy.

### Method
**FastQx** implements the **pairwise Qₓ statistic** (Berg & Coop, 2014, PLoS Genetics) for detecting signatures of directional polygenic adaptation between two populations, using the exact Hudson (1992) F_ST estimator recommended by Bhatia et al. (2013).


$$
\boxed{Q_x = \frac{\left( \sum_i \beta_i (p_{1i} - p_{2i}) \right)^2}{\sum_i \beta_i^2 \cdot 2 p_{1i} p_{2i} \cdot \hat{F}_{\text{ST},i}}}
$$

where:
- Numerator: squared difference in polygenic scores
- Denominator: expected variance under neutral drift
- $\hat{F}_{\text{ST},i}$ = **Hudson (1992) estimator as per Bhatia et al. (2013, Genome Research, Eq. S7)**:
  $$
  \hat{F}_{\text{ST},i} = \frac{(p_{1i} - p_{2i})^2}{p_{1i}(1-p_{2i}) + p_{2i}(1-p_{1i})}
  $$
  (exact, unbiased for large samples)

**v1.0 (current) **
- Optional neutral SNP panel → global neutral F_ST replaces per-locus F_ST
- Eliminates ascertainment bias from GWAS SNPs
- closer to the full multivariate Qₓ even for closely related populations
- batched version Still 13–31× faster than CPU baseline (CUDA) depending on the dataset

**v0.9 behavior preserved** (when no neutral SNPs provided):
- Falls back to per-locus Hudson F_ST from GWAS SNPs (fast screening mode)

Empirical p-values via **effect-size permutation null** (`sample(beta)`) — robust to GWAS stratification bias (Sohail et al., 2019, eLife).

## 📊 Performance Summary
 ### original 300 tests dataset with  each  having around 500 snps
| Method | Time (300 tests, 10k perms) | Speedup (vs cpu) |
|--------|------------------------------|---------|
| **CPU Baseline** | 105 seconds | 1x | N/A |
| **version 1 (Sequential GPU)** | ~11 seconds | ~10x | 
| **version 2 (Batch GPU)** | **~8 sec** | **~13x** |

### out of memory error tests between sequential and batched gpu versions

| Method | Time (303 tests ( same 300 + 3 problematic tests), 10k perms) | Speedup |  Batch size for a 16 gb 5060 ti | 
|--------|------------------------------|---------|
| **version 1 (Sequential GPU)** | **210 seconds** | **1x** | ---------|
| **version 2 (Batch GPU)** | **~77 seconds** | ~2.8 | 7|

### Stress test dataset: 100 tests with 2000-5000 SNPs each, 10,000 permutations


| Method | Time (100 tests, 10k perms) | Speedup |  Batch size for a 16 gb 5060 ti | 
|--------|------------------------------|---------|---------|
| **version 1 (Sequential GPU)** | ~603.85 seconds | 1x | with out of memory errors|
| **version 2 (Batch GPU)** | **~22 sec** | **~35x** | 7| no errors|
---


## ✨ Key Features

### Two Optimized GPU Implementations

#### **version 1: Sequential GPU (Adaptive)** ⭐ *Recommended for small datasets and debugging cases*
- Processes tests **one-by-one** on GPU
- Each test gets **perfectly-sized** permutation matrix (`n_snps × n_perm`)
- **Adaptive OOM handling**: Automatically excludes tests too large for GPU
- **Best for**: small datasets, debugging phase.

#### **version 2: Batch GPU** 🚀 *Fastest for uniform datasets*
- Processes tests in **batches** (configurable batch size)
- Each test gets its own matrix (no memory waste!)
- **True parallel compute**: Queues all operations, single sync per batch
- **faster** option
- **Best for**: Large uniform datasets, sufficient GPU memory (16GB+)

### Key Innovations

✅ **Zero Memory Waste**: Per-test matrix generation (not max-size matrices)  
✅ **Pure GPU Computation**: Permutation matrices generated directly on GPU  
✅ **Adaptive OOM Recovery**: Automatic exclusion of problematic tests  
✅ **Numerical Accuracy**: < 1e-5 difference from CPU baseline  

---

## 🚀 Quick Start


### Installation

```r
# Install R packages
install.packages("reticulate")

# Install CuPy (GPU computing library)
reticulate::py_install("cupy-cuda12x")  # For CUDA 12.x
# OR
reticulate::py_install("cupy-cuda11x")  # For CUDA 11.x
```

### Basic Usage

```r
# Load library and import CuPy
library(reticulate)
cp <- import("cupy")

# Run Sequential GPU (V10) - Memory-efficient
result <- compute_Qx_sequential_gpu_pure_adaptive(
  maf1 = population1_allele_freqs,
  maf2 = population2_allele_freqs,
  beta = gwas_effect_sizes,
  n_perm = 10000,
  seed = 123
)

# Run Batch GPU (V11) - Maximum speed
results <- compute_Qx_batch_gpu_adaptive(
  maf1_list = list_of_maf1_vectors,
  maf2_list = list_of_maf2_vectors,
  beta_list = list_of_beta_vectors,
  n_perm = 10000,
  seed = 123,
  batch_size = 8  # Tune based on GPU memory
)
```

---

## 📈 Batch Size Optimization

**Critical parameter**: `batch_size` must fit in GPU memory to avoid OOM thrashing.

### Memory Estimation Formula

```
GPU Memory per batch ≈ batch_size × avg_n_snps × n_perm × 4 bytes × 7 (overhead)
```

### Recommended Batch Sizes (16GB GPU, 10k permutations)

| Avg SNPs per Test | Recommended Batch Size | Memory Usage |
|-------------------|------------------------|--------------|
| 400-600 | 60 | ~1.5 GB |
| 1000-2000 | 20-30 | ~8-12 GB |
| 2000-5000 | **7-8** | ~12-14 GB |
| 5000-10000 | 3-5 | ~10-15 GB |

⚠️ **Warning**: Using too large a batch size causes severe performance degradation due to OOM retry loops!

### Performance Cliff Example

*Stress test (300 tests, 2000-5000 SNPs, 16GB GPU):*

| Batch Size | Time | Result |
|------------|------|--------|
| 7-8 | **34 sec** | ✅ Optimal |
| 9+ | 60+ min | ❌ OOM thrashing |

---

## 🎯 When to Use Each Version

| Scenario | Recommended Version | Why |
|----------|-------------------|-----|
| **Mixed test sizes** (100-5000 SNPs) | **version 1** | Handles variety well, adaptive OOM |
| **Uniform test sizes** + large dataset | **version 2** | Maximum speed via batching |
| **Speed critical** + 16GB+ GPU | **version 2** | Fastest option (tune batch size!) |
| **Testing/debugging** | **version 1** | Simpler, easier to debug |

---

## 🧹 GPU Memory Management

CuPy caches GPU memory for performance. Use `gpu_cleanup()` between runs:

```r
gpu_cleanup <- function() {
  cp$cuda$Stream$null$synchronize()
  mempool <- cp$get_default_memory_pool()
  mempool$free_all_free()
  pinned_mempool <- cp$get_default_pinned_memory_pool()
  pinned_mempool$free_all_blocks()
  gc(); gc(); gc()
}

# Use between function calls
results1 <- compute_Qx_batch_gpu_adaptive(...)
gpu_cleanup()
results2 <- compute_Qx_batch_gpu_adaptive(...)
```

---


## 🔍 Verification

All GPU implementations produce **numerically identical** results to the CPU baseline:

- **Qx (observed statistic)**: < 1e-5 difference (floating-point precision)
- **Fst**: Exact match
- **p-values**:   extremely close (uses different random seeds for permutations, expected)

---

## 📁 Repository Structure

```
.
├── gpu_qx_demo.ipynb          # Main demonstration notebook
├── README.md                  # This file
```

---

## ⚠️ Important Notes

### NA Handling
Following the original implementation:
1. Pre-filter entire SNPs (rows) with NA using `na.omit()`
2. Ensures observed and permuted calculations use identical SNP sets
3. Safety: `na.rm = TRUE` in sums as failsafe

### Random Seeds
- CPU and GPU use **different random number generators**
- Qx (observed) values **must match** across methods
- p-values **will slightly differ** (different permutation samples)
- This is **expected behavior**, not a bug!

### Demo Data
The notebook uses **simulated data** to demonstrate the GPU optimization technique. Real genomic data is subject to data use agreements and is not included in this repository.

---
### Limitations of the two-population approximation

The implementations in **FastQx** use a **pairwise (two-population) simplification** of the full multivariate Qₓ statistic (Berg & Coop, 2014). While mathematically valid and widely used for exploratory scans, this approximation has known limitations:

| Limitation | Explanation | Practical consequence |
|------------|-------------|-----------------------|
| **Ignores hierarchical population structure** | The full Qₓ uses a complete covariance matrix F estimated from neutral SNPs across all populations. The pairwise version implicitly assumes a star phylogeny (no shared drift between the two populations after their split). | Overestimates drift variance (and thus underestimates Qₓ) when the two populations are closely related (e.g., French vs. British, Tuscans vs. Spaniards). Signals can be diluted or lost. |
| **Less powerful than full multivariate Qₓ for >2 populations** | With many populations, the full Qₓ (or PCA-based versions) extracts more signal by jointly modeling all covariances. | Pairwise scans require multiple-testing correction across ~1,200 pairs (Bonferroni ≈ 4×10⁻⁵), reducing sensitivity compared to a single multivariate test. |
| **Permutation null assumes exchangeable effect sizes** | Shuffling betas destroys true signal but assumes the distribution of ascertained effect sizes is representative — a standard and accepted assumption (Sohail et al., 2019). | Remains robust to the main GWAS stratification biases that plagued early Qₓ studies. |

**When the approximation works well**  
- Distant population pairs (e.g., Yoruba vs. French, Han vs. Papuan) → shared drift is negligible → pairwise ≈ full Qₓ  
- Large allele frequency differences (strong clines: height, pigmentation, lactose persistence)  
- Exploratory genome-wide or trait-wide screens where speed is critical

**Bottom line**  
FastQx is designed as a **high-throughput screening tool**. Significant pairwise hits should ideally be confirmed with the full multivariate Qₓ (e.g., using the original Berg & Coop code, Colbran’s Julia implementation, or PCA-based tests) before claiming strong evidence of polygenic adaptation.


### License

MIT License 

### Citation

If you use this tool in published research, please cite:

> Berg JJ, Coop G (2014). A population genetic signal of polygenic adaptation. PLoS Genetics 10(8):e1004412. https://doi.org/10.1371/journal.pgen.1004412  
> Sohail M, et al. (2019). Polygenic adaptation on height is overestimated due to uncorrected stratification. eLife 8:e39702. https://doi.org/10.7554/eLife.39702

If the software contributed substantially, please also consider citing:

> Pasquale Caterino (2025). FastQx: High-performance GPU implementation of the pairwise Qₓ statistic. [https://github.com/pasq-cat/FastQx](https://github.com/pasq-cat/FastQx)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the repository maintainer.

---

## 🙏 Acknowledgments

- Original QX test implementation: #toadd
- GPU optimization: This implementation

---

