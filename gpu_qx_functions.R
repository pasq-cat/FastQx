# gpu_qx_functions.R
# Standalone R script for GPU-accelerated Qx test functions

# Load required packages and import CuPy
library(reticulate)
cp <- import("cupy")

# Hudson Fst estimator (CPU version)
Fst <- function(maf1, maf2, n1 = NULL, n2 = NULL) {
  if (!is.null(n1) && !is.null(n2)) {
    num <- (maf1 - maf2)^2 - maf1 * (1 - maf1) / n1 - maf2 * (1 - maf2) / n2
    denom <- maf1 * (1 - maf2) + maf2 * (1 - maf1) + 1e-12
    fst_vals <- ifelse(denom > 0, num / denom, 0)
  } else {
    fst_num <- (maf1 - maf2)^2
    fst_denom <- maf1 * (1 - maf2) + maf2 * (1 - maf1) + 1e-12
    fst_vals <- ifelse(fst_denom > 0, fst_num / fst_denom, 0)
  }
  return(fst_vals)
}

# CPU implementation of Qx statistic
compute_Qx_cpu <- function(maf1, maf2, beta, neutral_maf1 = NULL, neutral_maf2 = NULL, n1 = NULL, n2 = NULL, n_perm = 10000, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  data_complete <- na.omit(data.frame(maf1 = maf1, maf2 = maf2, beta = beta))
  maf1 <- data_complete$maf1
  maf2 <- data_complete$maf2
  beta <- data_complete$beta
  diff_obs <- maf1 * beta - maf2 * beta
  N_Qx_obs <- sum(diff_obs)^2
  if (!is.null(neutral_maf1) && !is.null(neutral_maf2)) {
    keep_neut <- complete.cases(neutral_maf1, neutral_maf2)
    n1_neutral <- neutral_maf1[keep_neut]
    n2_neutral <- neutral_maf2[keep_neut]
    Fst_neutral <- mean(Fst(n1_neutral, n2_neutral, n1 = n1, n2 = n2), na.rm = TRUE)
    Va_term <- sum(beta^2 * 2 * maf1 * maf2)
    D_Qx <- Va_term * Fst_neutral
  } else {
    fst_values <- Fst(maf1, maf2, n1 = n1, n2 = n2)
    D_Qx <- sum((beta^2) * (2 * maf1 * maf2) * fst_values)
  }
  Qx_obs <- N_Qx_obs / D_Qx
  Fst_obs <- ifelse(!is.null(neutral_maf1) && !is.null(neutral_maf2), Fst_neutral, mean(fst_values, na.rm = TRUE))
  N_Qx_perm <- numeric(n_perm)
  for (i in 1:n_perm) {
    beta_perm <- sample(beta)
    diff_perm <- maf1 * beta_perm - maf2 * beta_perm
    N_Qx_perm[i] <- sum(diff_perm)^2
  }
  Qx_perm <- N_Qx_perm / D_Qx
  empirical_p <- mean(abs(Qx_perm) >= abs(Qx_obs))
  return(list(Qx = Qx_obs, Fst = Fst_obs, p_value = empirical_p))
}

# GPU memory cleanup utility
gpu_cleanup <- function() {
  cat("🔥 GPU MEMORY CLEANUP\n\n")
  tryCatch({
    cat("Step 1: Synchronizing GPU...\n")
    cp$cuda$Stream$null$synchronize()
    cat("Step 2: Freeing memory pools...\n")
    mempool <- cp$get_default_memory_pool()
    mempool$free_all_free()
    pinned_mempool <- cp$get_default_pinned_memory_pool()
    pinned_mempool$free_all_blocks()
    cat("Step 3: CPU garbage collection...\n")
    gc();
    cat("\n✅ GPU memory cleaned successfully!\n")
    cat("ℹ️  Memory should be fully released (check nvidia-smi)\n")
  }, error = function(e) {
    cat("\n❌ Cleanup failed:\n")
    cat(sprintf("   %s\n", conditionMessage(e)))
    cat("\n💡 Try restarting R session if problems persist\n")
  })
}


# Version 1 ADAPTIVE: Sequential GPU with PURE GPU + ADAPTIVE OOM HANDLING
compute_Qx_sequential_gpu_pure_adaptive <- function(maf1, maf2, beta,neutral_maf1 = NULL, neutral_maf2 = NULL,n1 = NULL, n2 = NULL,  n_perm = 10000, seed = NULL,
                                                     test_name = NULL, test_id = NULL) {
  result <- tryCatch({
    if (!is.null(seed)) set.seed(seed)
    data_complete <- na.omit(data.frame(maf1 = maf1, maf2 = maf2, beta = beta))
    maf1 <- data_complete$maf1
    maf2 <- data_complete$maf2
    beta <- data_complete$beta
    n <- length(beta)
    if (n == 0) {
      return(list(Qx = NA, Fst = NA, p_value = NA, test_name = test_name, test_id = test_id, n_snps = 0, error = "No valid SNPs after NA removal", excluded = FALSE))
    }
    maf1_gpu <- cp$array(maf1, dtype = cp$float64)
    maf2_gpu <- cp$array(maf2, dtype = cp$float64)
    beta_gpu <- cp$array(beta, dtype = cp$float64)
    n1_gpu <- if (!is.null(n1)) cp$array(rep(n1, length(maf1)), dtype = cp$float64) else NULL
    n2_gpu <- if (!is.null(n2)) cp$array(rep(n2, length(maf2)), dtype = cp$float64) else NULL
    diff_obs <- maf1_gpu * beta_gpu - maf2_gpu * beta_gpu
    N_Qx_obs <- cp$sum(diff_obs)**2
    if (!is.null(neutral_maf1) && !is.null(neutral_maf2)) {
      neut_keep <- complete.cases(neutral_maf1, neutral_maf2)
      neu1 <- cp$array(neutral_maf1[neut_keep], dtype = cp$float64)
      neu2 <- cp$array(neutral_maf2[neut_keep], dtype = cp$float64)
      if (!is.null(n1_gpu) && !is.null(n2_gpu)) {
        fst_num <- (maf1_gpu - maf2_gpu)**2 - maf1_gpu * (1 - maf1_gpu) / n1_gpu - maf2_gpu * (1 - maf2_gpu) / n2_gpu
      } else {
        fst_num <- (maf1_gpu - maf2_gpu)**2
      }
      fst_denom <- neu1 * (1 - neu2) + neu2 * (1 - neu1) + 1e-12
      fst_neutral_global <- cp$mean(fst_num / fst_denom)
      Va_term <- cp$sum(beta_gpu**2 * 2 * maf1_gpu * maf2_gpu)
      D_Qx <- Va_term * fst_neutral_global
      Fst_obs <- as.numeric(fst_neutral_global)
    } else {
      if (!is.null(n1_gpu) && !is.null(n2_gpu)) {
        fst_num <- (maf1_gpu - maf2_gpu)**2 - maf1_gpu * (1 - maf1_gpu) / n1_gpu - maf2_gpu * (1 - maf2_gpu) / n2_gpu
      } else {
        fst_num <- (maf1_gpu - maf2_gpu)**2
      }
      fst_denom <- maf1_gpu * (1 - maf2_gpu) + maf2_gpu * (1 - maf1_gpu) + 1e-12
      fst_vals <- cp$where(fst_denom > 0, fst_num / fst_denom, 0)
      D_Qx <- cp$sum((beta_gpu**2) * (2 * maf1_gpu * maf2_gpu) * fst_vals)
      Fst_obs <- cp$mean(fst_vals)
    }
    Qx_obs <- N_Qx_obs / D_Qx
    if (!is.null(seed)) { cp$random$seed(as.integer(seed)) }
    n_gpu <- as.integer(n)
    n_perm_gpu <- as.integer(n_perm)
    random_matrix <- cp$random$rand(n_gpu, n_perm_gpu, dtype = cp$float32)
    perm_indices_all <- cp$argsort(random_matrix, axis = 0L)
    rm(random_matrix)
    cp$cuda$Stream$null$synchronize()
    cp$get_default_memory_pool()$free_all_blocks()
    beta_perm_all <- beta_gpu[perm_indices_all]
    rm(perm_indices_all)
    cp$cuda$Stream$null$synchronize()
    cp$get_default_memory_pool()$free_all_blocks()
    maf1_expanded <- cp$expand_dims(maf1_gpu, axis = 1L)
    maf2_expanded <- cp$expand_dims(maf2_gpu, axis = 1L)
    maf1_beta_perm <- maf1_expanded * beta_perm_all
    maf2_beta_perm <- maf2_expanded * beta_perm_all
    rm(beta_perm_all)
    cp$cuda$Stream$null$synchronize()
    cp$get_default_memory_pool()$free_all_blocks()
    diff_perm_all <- maf1_beta_perm - maf2_beta_perm
    rm(maf1_beta_perm, maf2_beta_perm)
    cp$cuda$Stream$null$synchronize()
    cp$get_default_memory_pool()$free_all_blocks()
    diff_sum_perm <- cp$sum(diff_perm_all, axis = 0L)
    rm(diff_perm_all)
    cp$cuda$Stream$null$synchronize()
    cp$get_default_memory_pool()$free_all_blocks()
    N_Qx_perm <- diff_sum_perm**2
    Qx_perm_all <- N_Qx_perm / D_Qx
    Qx_perm_abs <- cp$abs(Qx_perm_all)
    Qx_obs_abs <- cp$abs(Qx_obs)
    exceeds <- Qx_perm_abs >= Qx_obs_abs
    p_value_gpu <- cp$mean(exceeds$astype(cp$float64))
    Qx_final <- as.numeric(Qx_obs$get())
    Fst_final <- as.numeric(Fst_obs$get())
    p_value_final <- as.numeric(p_value_gpu$get())
    rm(maf1_gpu, maf2_gpu, beta_gpu, Qx_perm_all,fst_num,fst_denom,fst_vals,D_Qx,N_Qx_perm,Qx_obs_abs,diff_sum_perm,exceeds)
    if (exists("fst_neutral_global")) {
      rm(fst_neutral_global,fst_denom_n,fst_num_n,Va_term)
    }
    cp$cuda$Stream$null$synchronize()
    cp$get_default_memory_pool()$free_all_blocks()
    list(Qx = Qx_final, Fst = Fst_final, p_value = p_value_final, test_name = test_name, test_id = test_id, n_snps = n, excluded = FALSE)
  }, error = function(e) {
    error_class <- class(e)
    error_msg <- conditionMessage(e)
    is_python_memory <- any(grepl("MemoryError", error_class, ignore.case = TRUE))
    is_cuda_oom <- any(grepl("CUDARuntimeError|cudaErrorMemoryAllocation", error_class, ignore.case = TRUE))
    is_r_error <- any(error_class == "error")
    is_oom <- is_python_memory || is_cuda_oom || is_r_error
    if (is_oom) {
      tryCatch({
        cp$cuda$Stream$null$synchronize()
        mempool <- cp$get_default_memory_pool()
        pinned_mempool <- cp$get_default_pinned_memory_pool()
        mempool$free_all_blocks()
        pinned_mempool$free_all_blocks()
      }, error = function(cleanup_error) {})
      gc(); gc();
      return(list(Qx = NA, Fst = NA, p_value = NA, test_name = test_name, test_id = test_id, n_snps = length(beta), excluded = TRUE, error = "OUT_OF_MEMORY"))
    } else {
      stop(e)
    }
  })
  return(result)
}

# Version 2: Batch GPU with PER-TEST matrices (no waste!)
compute_Qx_batch_gpu_adaptive <- function(maf1, maf2, beta,neutral_maf1 = NULL, neutral_maf2 = NULL,  test_names = NULL, test_ids = NULL,n1 = NULL, n2 = NULL,
                                                   n_perm = 10000, seed = NULL, batch_size = 60, max_removal_attempts = 50) {
  n_tests_original <- length(maf1)
  n_perm_int <- as.integer(n_perm)
  all_results <- vector("list", n_tests_original)
  excluded_tests <- list()
  if (is.null(test_names)) test_names <- paste0("Test_", 1:n_tests_original)
  if (is.null(test_ids)) test_ids <- paste0("ID_", 1:n_tests_original)
  for (i in 1:n_tests_original) {
    data_complete <- na.omit(data.frame(maf1 = maf1[[i]], maf2 = maf2[[i]], beta = beta[[i]]))
    maf1[[i]] <- data_complete$maf1
    maf2[[i]] <- data_complete$maf2
    beta[[i]] <- data_complete$beta
  }
  use_neutral <- !is.null(neutral_maf1) && !is.null(neutral_maf2)
  if (use_neutral) {
    if (length(neutral_maf1) != n_tests_original || length(neutral_maf2) != n_tests_original) {
      stop("neutral_maf1 and neutral_maf2 must have the same length as maf1/maf2/beta")
    }
    for (i in 1:n_tests_original) {
      neut_complete <- na.omit(data.frame(neutral_maf1 = neutral_maf1[[i]], neutral_maf2 = neutral_maf2[[i]]))
      neutral_maf1[[i]] <- neut_complete$neutral_maf1
      neutral_maf2[[i]] <- neut_complete$neutral_maf2
    }
  }
  n_snps <- sapply(beta, length)
  batch_starts <- seq(1, n_tests_original, by = batch_size)
  for (batch_idx in seq_along(batch_starts)) {
    batch_start <- batch_starts[batch_idx]
    batch_end <- min(batch_start + batch_size - 1, n_tests_original)
    batch_indices <- batch_start:batch_end
    batch_removal_count <- 0
    while (length(batch_indices) > 0 && batch_removal_count < max_removal_attempts) {
      batch_result <- tryCatch({
        maf1_batch <- maf1[batch_indices]
        maf2_batch <- maf2[batch_indices]
        beta_batch <- beta[batch_indices]
        n_snps_batch <- n_snps[batch_indices]
        n1_batch <- if (!is.null(n1)) n1[batch_indices] else NULL
        n2_batch <- if (!is.null(n2)) n2[batch_indices] else NULL
        neutral1_batch <- if (use_neutral) neutral_maf1[batch_indices] else NULL
        neutral2_batch <- if (use_neutral) neutral_maf2[batch_indices] else NULL
        batch_length <- length(batch_indices)
        perm_matrices_gpu <- vector("list", batch_length)
        for (i in 1:batch_length) {
          n_i <- as.integer(n_snps_batch[i])
          random_matrix <- cp$random$rand(n_i, n_perm_int, dtype = cp$float32)
          perm_matrices_gpu[[i]] <- cp$argsort(random_matrix, axis = 0L)
          rm(random_matrix)
        }
        maf1_gpu <- lapply(maf1_batch, function(x) cp$array(x, dtype = cp$float64))
        maf2_gpu <- lapply(maf2_batch, function(x) cp$array(x, dtype = cp$float64))
        beta_gpu <- lapply(beta_batch, function(x) cp$array(x, dtype = cp$float64))
        n1_batch_gpu <- if (!is.null(n1_batch)) lapply(n1_batch, function(x) cp$array(x, dtype = cp$int32))
        n2_batch_gpu <- if (!is.null(n2_batch)) lapply(n2_batch, function(x) cp$array(x, dtype = cp$int32))
        if (use_neutral) {
          neutral1_gpu <- lapply(neutral1_batch, function(x) cp$array(x, dtype = cp$float64))
          neutral2_gpu <- lapply(neutral2_batch, function(x) cp$array(x, dtype = cp$float64))
        }
        Qx_obs_list   <- vector("list", batch_length)
        Fst_obs_list  <- vector("list", batch_length)
        p_value_list  <- vector("list", batch_length)
        neutral_Fst_batch <- numeric(batch_length)
        for (i in 1:batch_length) {
          diff_obs <- maf1_gpu[[i]] * beta_gpu[[i]] - maf2_gpu[[i]] * beta_gpu[[i]]
          N_Qx_obs <- cp$sum(diff_obs)**2
          if (use_neutral) {
            if (!is.null(n1_batch_gpu) && !is.null(n2_batch_gpu)) {
              fst_num_n <- ((neutral1_gpu[[i]] - neutral2_gpu[[i]])**2 - neutral1_gpu[[i]] * (1 - neutral1_gpu[[i]]) / n1_batch_gpu[[i]] - neutral2_gpu[[i]] * (1 - neutral2_gpu[[i]]) / n2_batch_gpu[[i]])
            } else {
              fst_num_n <- (neutral1_gpu[[i]] - neutral2_gpu[[i]])**2
            }
            fst_denom_n <- neutral1_gpu[[i]] * (1 - neutral2_gpu[[i]]) + neutral2_gpu[[i]] * (1 - neutral1_gpu[[i]]) + 1e-12
            neutral_Fst_batch[i] <-  as.numeric(cp$mean(fst_num_n / fst_denom_n)$item())
            Va_term <- cp$sum(beta_gpu[[i]]**2 * 2 * maf1_gpu[[i]] * maf2_gpu[[i]])
            D_Qx <- Va_term * neutral_Fst_batch[i]
            Fst_obs_list[[i]] <- neutral_Fst_batch[i]
          } else {
            if (!is.null(n1_batch_gpu) && !is.null(n2_batch_gpu)) {
              fst_num <- (maf1_gpu[[i]] - maf2_gpu[[i]])**2 - maf1_gpu[[i]] * (1 - maf1_gpu[[i]]) / n1_batch_gpu[[i]] - maf2_gpu[[i]] * (1 - maf2_gpu[[i]]) / n2_batch_gpu[[i]]
            } else {
              fst_num <- (maf1_gpu[[i]] - maf2_gpu[[i]])**2
            }
            fst_denom <- maf1_gpu[[i]] * (1 - maf2_gpu[[i]]) + maf2_gpu[[i]] * (1 - maf1_gpu[[i]]) + 1e-12
            fst_vals <- cp$where(fst_denom > 0, fst_num / fst_denom, 0)
            D_Qx <- cp$sum(beta_gpu[[i]]**2 * 2 * maf1_gpu[[i]] * maf2_gpu[[i]] * fst_vals)
            Fst_obs_list[[i]] <- cp$mean(fst_vals)
          }
          Qx_obs_list[[i]] <- N_Qx_obs / D_Qx
          beta_perm <- beta_gpu[[i]][perm_matrices_gpu[[i]]]
          maf1_exp <- maf1_gpu[[i]][, NULL]
          maf2_exp <- maf2_gpu[[i]][, NULL]
          diff_perm <- maf1_exp * beta_perm - maf2_exp * beta_perm
          diff_sum_perm <- cp$sum(diff_perm, axis = 0L)
          N_Qx_perm <- diff_sum_perm**2
          Qx_perm <- N_Qx_perm / D_Qx
          p_value_list[[i]] <- cp$mean((cp$abs(Qx_perm) >= cp$abs(Qx_obs_list[[i]]))$astype(cp$float64))
        }
        batch_results <- lapply(1:batch_length, function(i) {
          list(
            Qx      = as.numeric(Qx_obs_list[[i]]$get()),
            Fst     = Fst_obs_list[[i]],
            p_value = as.numeric(p_value_list[[i]]$get()),
            test_name = test_names[batch_indices[i]],
            test_id   = test_ids[batch_indices[i]],
            n_snps    = n_snps_batch[i],
            neutral_used = use_neutral
          )
        })
        rm(perm_matrices_gpu, maf1_gpu, maf2_gpu, beta_gpu,beta_perm,diff_perm,diff_sum_perm,N_Qx_perm,Qx_perm)
        rm(Qx_obs_list, Fst_obs_list, p_value_list)
        list(success = TRUE, results = batch_results)
      }, error = function(e) {
        list(success = FALSE, error = e$message)
      }, finally = {
          if (use_neutral) {
            rm(neutral1_gpu, neutral2_gpu, neutral_Fst_batch)
          }
          cp$cuda$Stream$null$synchronize()
          mempool <- cp$get_default_memory_pool()
          mempool$free_all_free()
          pinned_mempool <- cp$get_default_pinned_memory_pool()
          pinned_mempool$free_all_blocks()
          gc();
        })
      if (batch_result$success) {
        for (i in seq_along(batch_indices)) {
          all_results[[batch_indices[i]]] <- batch_result$results[[i]]
        }
        break
      } else {
        batch_removal_count <- batch_removal_count + 1
        if (length(batch_indices) == 1) {
          excluded_tests[[length(excluded_tests) + 1]] <- list(
            index = batch_indices[1],
            test_name = test_names[batch_indices[1]],
            test_id = test_ids[batch_indices[1]],
            n_snps = n_snps[batch_indices[1]],
            error = batch_result$error
          )
          break
        }
        batch_n_snps <- n_snps[batch_indices]
        largest_idx_in_batch <- which.max(batch_n_snps)
        largest_global_idx <- batch_indices[largest_idx_in_batch]
        excluded_tests[[length(excluded_tests) + 1]] <- list(
          index = largest_global_idx,
          test_name = test_names[largest_global_idx],
          test_id = test_ids[largest_global_idx],
          n_snps = n_snps[largest_global_idx],
          error = batch_result$error
        )
        batch_indices <- batch_indices[-largest_idx_in_batch]
      }
    }
  }
  n_successful <- sum(!sapply(all_results, is.null))
  n_excluded   <- length(excluded_tests)
  return(list(
    results       = all_results,
    excluded      = excluded_tests,
    n_successful  = n_successful,
    n_excluded    = n_excluded,
    neutral_used  = use_neutral
  ))
}

# End of gpu_qx_functions.R
