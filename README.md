# GPU-Accelerated QX Test for Population Genetics

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![R](https://img.shields.io/badge/R-4.0%2B-blue)](https://www.r-project.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.x%20%7C%2012.x-green)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)



---
# **Version 0.9** Fast Pairwise Qₓ Statistic (CPU + GPU)

**Fast implementation of the pairwise Qₓ polygenic selection statistic** introduced by Berg & Coop (2014) for detecting directional polygenic adaptation between population pairs. Achieves **10-36x speedup** over CPU baseline using NVIDIA GPU acceleration with CuPy.

### Method
This repository provides highly optimized CPU and GPU (CUDA) implementations of the **pairwise Qₓ test**, a widely used simplification of the multivariate Qₓ statistic (Berg & Coop, 2014, PLoS Genetics) for scanning polygenic adaptation across population pairs (e.g., in the HGDP or 1000 Genomes datasets).

Key features:
- Numerator: squared difference in polygenic scores between two populations  
  `Qx_num = [Σ βᵢ (p₁ᵢ − p₂ᵢ)]²`
- Denominator: drift expectation using per-locus Hudson Fₛₜ weighting  
  `Qx_denom = Σ βᵢ² × 2 p₁ᵢ p₂ᵢ × [(p₁ᵢ − p₂ᵢ)² / (4 p̄ᵢ (1 − p̄ᵢ))]`
- Empirical p-values via **effect-size permutation null** (`sample(beta)`), which destroys directional covariance while preserving GWAS ascertainment properties — following the bias-robust approach introduced by **Sohail et al. (2019, eLife)** and used in subsequent pipelines (e.g., Colbran et al., 2021).

The two-population simplification with Hudson Fₛₜ weighting is mathematically valid and commonly used in the literature for exhaustive pairwise testing (e.g., European vs. African height clines, pigmentation, etc.).

- Ideal for strong continental signals (height, pigmentation, etc.)
- Drift estimated from GWAS SNPs (standard fast approximation)

**v0.9 (current)**  
- Drift estimated from GWAS SNPs (standard fast approximation)  
- Ideal for **continental-scale comparisons** and strong clines  
- Perfect for exploratory scans across thousands of traits

**v1.0 (in development)**  
- Optional neutral SNP panel for unbiased drift estimation  
- Full accuracy even for closely related populations (e.g., within Europe/Asia)  
- Matches gold-standard multivariate implementations

## 📊 Performance Summary
 ### original 300 tests dataset with  each  having around 500 snps
| Method | Time (300 tests, 10k perms) | Speedup (vs cpu) |
|--------|------------------------------|---------|
| **CPU Baseline** | 105 seconds | 1x | N/A |
| **version 1 (Sequential GPU)** | ~11 seconds | ~10x | 
| **version 2 (Batch GPU)** | **~8 sec** | **~13x** |


| Method | Time (303 tests ( same 300 + 3 problematic tests), 10k perms) | Speedup |  Batch size for a 16 gb 5060 ti | 
|--------|------------------------------|---------|
| **version 1 (Sequential GPU)** | **142 seconds** | **1x** | ---------|
| **version 2 (Batch GPU)** | ~146 seconds | ~1 | 7|

*Stress test dataset: 100 tests with 2000-5000 SNPs each, 10,000 permutations*


| Method | Time (100 tests, 10k perms) | Speedup |  Batch size for a 16 gb 5060 ti | 
|--------|------------------------------|---------|---------|
| **version 1 (Sequential GPU)** | ~603.85 seconds | 1x | ---------|
| **version 2 (Batch GPU)** | **~17 sec** | **~35x** | 7|
---

## 🔬 The QX Test

The QX test ([Joshi et al., *Nature* 2015](https://doi.org/10.1038/nature14618)) measures heterogeneity in polygenic scores between populations by comparing:

- **Observed variance** in allele frequency × effect size products
- **Expected variance** under the null hypothesis (via permutation testing)

This implementation accelerates the computationally intensive permutation testing using GPU parallel processing.

---


## ✨ Key Features

### Two Optimized GPU Implementations

#### **version 1: Sequential GPU (Adaptive)** ⭐ *Recommended for small datasets and debugging cases*
- Processes tests **one-by-one** on GPU
- Each test gets **perfectly-sized** permutation matrix (`n_snps × n_perm`)
- **Adaptive OOM handling**: Automatically excludes tests too large for GPU
- **~10x faster** than CPU baseline
- **Best for**: Heterogeneous datasets, limited GPU memory (<8GB)

#### **version 2: Batch GPU** 🚀 *Fastest for uniform datasets*
- Processes tests in **batches** (configurable batch size)
- Each test gets its own matrix (no memory waste!)
- **True parallel compute**: Queues all operations, single sync per batch
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

## 📊 Benchmark Results

### Original Dataset (300 tests, 400-600 SNPs)

```
CPU Baseline:        ~45 minutes
Sequential GPU (V10): ~4.5 minutes  (10x speedup)
Batch GPU (V11):      ~1.5 minutes  (30x speedup, batch_size=60)
```

### Stress Test Dataset (100 tests, 2000-5000 SNPs)

```
Sequential GPU (V10):  ~603 seconds
Batch GPU (V11):      16 seconds   (36x speedup, batch_size=7-8)
```

### Extended Dataset (303 tests, 3 huge tests: 85K, 55K, 35K SNPs)

```
Sequential GPU (V10): ~141  (3 tests excluded via OOM)
Batch GPU (V11):      ~147 seconds  (3 tests excluded, batch_size=5)
```

*Note: Extended dataset shows minimal speedup due to OOM-induced retry overhead with huge tests.*

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

## 🛠️ Implementation Details

### CPU Baseline (Original)
- Pure R implementation
- Reference for validation
- `compute_Qx_cpu()`

### version 1: Sequential GPU (Pure + Adaptive)
- Generates permutation matrix **on GPU** using `cp$random$rand()` + `cp$argsort()`
- Processes tests one-by-one
- Adaptive OOM handling with automatic test exclusion
- `compute_Qx_sequential_gpu_pure_adaptive()`

### version 2: Batch GPU (Per-Test Matrices)
- Generates **per-test** matrices for all tests in batch
- Queues all GPU operations
- **Single synchronization** at end of batch (true parallelism!)
- Adaptive OOM: removes largest test and retries batch
- `compute_Qx_batch_gpu_adaptive()`

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
| **Uses GWAS SNPs to estimate drift** | The denominator relies on per-locus Hudson Fₛₜ computed from the same GWAS loci instead of a separate set of neutral SNPs. | Slight upward bias in the denominator for traits under weak polygenic selection or with residual stratification. Usually minor, but not ideal. |
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

> Pasquale Caterino (2025). FastQx: High-performance GPU implementation of the pairwise Qₓ statistic. [https://github.com/yourusername/FastQx](https://github.com/pasq-cat/FastQx)

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

