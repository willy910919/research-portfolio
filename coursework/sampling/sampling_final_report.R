set.seed(20260627)

options(stringsAsFactors = FALSE)

out_tables <- file.path("output", "tables")
out_figures <- file.path("output", "figures")
dir.create(out_tables, recursive = TRUE, showWarnings = FALSE)
dir.create(out_figures, recursive = TRUE, showWarnings = FALSE)

iter <- 3000

write_result <- function(x, file_name) {
  write.csv(x, file.path(out_tables, file_name), row.names = FALSE, fileEncoding = "UTF-8")
}

mse <- function(est, truth) {
  mean((est - truth)^2)
}

relative_bias <- function(est, truth) {
  (mean(est) - truth) / truth
}

allocate_integer <- function(weights, total_n, min_each = 2L) {
  raw <- total_n * weights / sum(weights)
  n_h <- floor(raw)
  n_h[n_h < min_each] <- min_each
  while (sum(n_h) < total_n) {
    frac <- raw - floor(raw)
    idx <- order(frac, decreasing = TRUE)
    for (j in idx) {
      if (sum(n_h) >= total_n) break
      n_h[j] <- n_h[j] + 1L
    }
  }
  while (sum(n_h) > total_n) {
    idx <- order(n_h - min_each, decreasing = TRUE)
    for (j in idx) {
      if (sum(n_h) <= total_n) break
      if (n_h[j] > min_each) n_h[j] <- n_h[j] - 1L
    }
  }
  n_h
}

plot_lines <- function(df, x_col, y_col, group_col, main, xlab, ylab, file_name,
                       legend_pos = "topright") {
  groups <- unique(df[[group_col]])
  cols <- seq_along(groups)
  png(file.path(out_figures, file_name), width = 1200, height = 800, res = 140)
  first <- TRUE
  ylim <- range(df[[y_col]], na.rm = TRUE)
  for (i in seq_along(groups)) {
    sub <- df[df[[group_col]] == groups[i], ]
    sub <- sub[order(sub[[x_col]]), ]
    if (first) {
      plot(sub[[x_col]], sub[[y_col]], type = "b", pch = 19, col = cols[i],
           ylim = ylim, main = main, xlab = xlab, ylab = ylab)
      first <- FALSE
    } else {
      lines(sub[[x_col]], sub[[y_col]], type = "b", pch = 19, col = cols[i])
    }
  }
  legend(legend_pos, legend = groups, col = cols, pch = 19, lty = 1, bty = "n")
  grid()
  dev.off()
}

# -----------------------------------------------------------------------------
# 1. Simple random sampling
# -----------------------------------------------------------------------------

simulate_srs <- function() {
  N <- 5000
  n_set <- c(30, 60, 120, 240, 480)
  y <- rpois(N, lambda = 10)
  mu <- mean(y)
  S2 <- var(y)

  perf_rows <- list()
  coverage_rows <- list()
  counter <- 1L
  cov_counter <- 1L

  for (n in n_set) {
    ybar_wor <- numeric(iter)
    ybar_wr_n <- numeric(iter)
    ybar_wr_nu <- numeric(iter)
    cover_z <- logical(iter)
    cover_t <- logical(iter)

    for (b in seq_len(iter)) {
      s_wor <- sample.int(N, n, replace = FALSE)
      ys_wor <- y[s_wor]
      ybar_wor[b] <- mean(ys_wor)
      se_wor <- sqrt((1 - n / N) * var(ys_wor) / n)
      cover_z[b] <- (ybar_wor[b] - qnorm(0.975) * se_wor <= mu) &&
        (mu <= ybar_wor[b] + qnorm(0.975) * se_wor)
      cover_t[b] <- (ybar_wor[b] - qt(0.975, df = n - 1) * se_wor <= mu) &&
        (mu <= ybar_wor[b] + qt(0.975, df = n - 1) * se_wor)

      s_wr <- sample.int(N, n, replace = TRUE)
      ybar_wr_n[b] <- mean(y[s_wr])
      unique_units <- unique(s_wr)
      ybar_wr_nu[b] <- mean(y[unique_units])
    }

    estimators <- list(
      "SRSWOR_ybar" = ybar_wor,
      "SRSWR_ybar_n" = ybar_wr_n,
      "SRSWR_ybar_nu" = ybar_wr_nu
    )

    for (est_name in names(estimators)) {
      est <- estimators[[est_name]]
      perf_rows[[counter]] <- data.frame(
        section = "Simple random sampling",
        estimator = est_name,
        n = n,
        empirical_mean = mean(est),
        true_mean = mu,
        bias = mean(est) - mu,
        relative_bias = relative_bias(est, mu),
        empirical_variance = var(est),
        empirical_mse = mse(est, mu)
      )
      counter <- counter + 1L
    }

    coverage_rows[[cov_counter]] <- data.frame(
      section = "Simple random sampling",
      n = n,
      true_mean = mu,
      finite_population_variance = S2,
      coverage_z = mean(cover_z),
      coverage_t = mean(cover_t)
    )
    cov_counter <- cov_counter + 1L
  }

  perf <- do.call(rbind, perf_rows)
  coverage <- do.call(rbind, coverage_rows)
  write_result(perf, "srs_performance.csv")
  write_result(coverage, "srs_coverage.csv")

  plot_lines(perf, "n", "empirical_mse", "estimator",
             "SRS estimator MSE by sample size", "Sample size n",
             "Empirical MSE", "fig_srs_mse.png")

  bias_plot <- perf
  bias_plot$abs_bias <- abs(bias_plot$bias)
  plot_lines(bias_plot, "n", "abs_bias", "estimator",
             "SRS absolute bias by sample size", "Sample size n",
             "Absolute bias", "fig_srs_bias.png")

  coverage_long <- rbind(
    data.frame(n = coverage$n, method = "z interval", coverage = coverage$coverage_z),
    data.frame(n = coverage$n, method = "t interval", coverage = coverage$coverage_t)
  )
  plot_lines(coverage_long, "n", "coverage", "method",
             "SRSWOR 95% confidence interval coverage", "Sample size n",
             "Coverage probability", "fig_srs_coverage.png")

  list(performance = perf, coverage = coverage)
}

# -----------------------------------------------------------------------------
# 2. Ratio estimation under SRSWOR
# -----------------------------------------------------------------------------

generate_bivariate_poisson <- function(N, target_rho, base_mu = 20) {
  mu1 <- base_mu
  mu2 <- base_mu
  mu3 <- target_rho * base_mu / (1 - target_rho)
  x1 <- rpois(N, mu1)
  x2 <- rpois(N, mu2)
  x3 <- rpois(N, mu3)
  x <- x1 + x3
  y <- x2 + x3
  data.frame(x = x, y = y)
}

simulate_ratio <- function() {
  N <- 6000
  n_set <- c(50, 100, 200, 400)
  target_rhos <- seq(0.1, 0.9, by = 0.1)
  rows <- list()
  counter <- 1L

  for (rho in target_rhos) {
    pop <- generate_bivariate_poisson(N, rho)
    x <- pop$x
    y <- pop$y
    mu_x <- mean(x)
    mu_y <- mean(y)
    actual_cor <- cor(x, y)
    R_true <- mu_y / mu_x
    e_pop <- y - R_true * x

    for (n in n_set) {
      ratio_est <- numeric(iter)
      ybar_est <- numeric(iter)
      cover_vhat <- logical(iter)
      cover_vpop <- logical(iter)

      v_pop_linearized <- (1 - n / N) * var(e_pop) / n

      for (b in seq_len(iter)) {
        s <- sample.int(N, n, replace = FALSE)
        xs <- x[s]
        ys <- y[s]
        ybar_est[b] <- mean(ys)
        R_hat <- mean(ys) / mean(xs)
        ratio_est[b] <- R_hat * mu_x
        e_s <- ys - R_hat * xs
        v_hat <- (1 - n / N) * var(e_s) / n

        se_hat <- sqrt(max(v_hat, 0))
        se_pop <- sqrt(max(v_pop_linearized, 0))
        cover_vhat[b] <- (ratio_est[b] - qnorm(0.975) * se_hat <= mu_y) &&
          (mu_y <= ratio_est[b] + qnorm(0.975) * se_hat)
        cover_vpop[b] <- (ratio_est[b] - qnorm(0.975) * se_pop <= mu_y) &&
          (mu_y <= ratio_est[b] + qnorm(0.975) * se_pop)
      }

      mse_ratio <- mse(ratio_est, mu_y)
      mse_ybar <- mse(ybar_est, mu_y)
      rows[[counter]] <- data.frame(
        section = "Ratio estimation under SRSWOR",
        target_cor = rho,
        actual_cor = actual_cor,
        n = n,
        true_mean_y = mu_y,
        mean_ratio_estimator = mean(ratio_est),
        bias_ratio = mean(ratio_est) - mu_y,
        empirical_mse_ratio = mse_ratio,
        empirical_mse_ybar = mse_ybar,
        relative_efficiency_ybar_over_ratio = mse_ybar / mse_ratio,
        coverage_estimated_variance = mean(cover_vhat),
        coverage_population_linearized_variance = mean(cover_vpop)
      )
      counter <- counter + 1L
    }
  }

  ratio_summary <- do.call(rbind, rows)
  write_result(ratio_summary, "ratio_performance.csv")

  ratio_n200 <- ratio_summary[ratio_summary$n == 200, ]
  plot_lines(ratio_n200, "target_cor", "empirical_mse_ratio", "section",
             "Ratio estimator MSE by correlation (n = 200)", "Target correlation",
             "Empirical MSE", "fig_ratio_mse_rho_n200.png")

  plot_lines(ratio_summary, "target_cor", "relative_efficiency_ybar_over_ratio", "n",
             "Relative efficiency of ratio estimator", "Target correlation",
             "MSE(ybar) / MSE(ratio)", "fig_ratio_relative_efficiency.png",
             legend_pos = "topleft")

  ratio_rho07 <- ratio_summary[abs(ratio_summary$target_cor - 0.7) < 1e-8, ]
  coverage_long <- rbind(
    data.frame(n = ratio_rho07$n, method = "estimated variance",
               coverage = ratio_rho07$coverage_estimated_variance),
    data.frame(n = ratio_rho07$n, method = "population linearized variance",
               coverage = ratio_rho07$coverage_population_linearized_variance)
  )
  plot_lines(coverage_long, "n", "coverage", "method",
             "Ratio estimator CI coverage at target correlation 0.7",
             "Sample size n", "Coverage probability", "fig_ratio_coverage_rho07.png")

  ratio_summary
}

# -----------------------------------------------------------------------------
# 3. Stratified random sampling
# -----------------------------------------------------------------------------

simulate_stratified <- function() {
  N_h <- c(1500, 1500, 1000, 1000)
  H <- length(N_h)
  N <- sum(N_h)
  n_set <- c(120, 240, 480)
  mean_h <- c(20, 40, 70, 110)
  sd_h <- c(5, 12, 25, 50)
  stratum <- rep(seq_len(H), times = N_h)
  y <- unlist(lapply(seq_len(H), function(h) rnorm(N_h[h], mean = mean_h[h], sd = sd_h[h])))
  mu <- mean(y)
  W_h <- N_h / N
  S_h <- tapply(y, stratum, sd)
  rows <- list()
  counter <- 1L

  strat_estimate <- function(n_h) {
    means <- numeric(H)
    for (h in seq_len(H)) {
      idx_h <- which(stratum == h)
      s_h <- sample(idx_h, n_h[h], replace = FALSE)
      means[h] <- mean(y[s_h])
    }
    sum(W_h * means)
  }

  for (n in n_set) {
    n_prop <- allocate_integer(N_h, n)
    n_neyman <- allocate_integer(N_h * S_h, n)

    est_prop <- numeric(iter)
    est_neyman <- numeric(iter)
    est_srs <- numeric(iter)

    for (b in seq_len(iter)) {
      est_prop[b] <- strat_estimate(n_prop)
      est_neyman[b] <- strat_estimate(n_neyman)
      s_srs <- sample.int(N, n, replace = FALSE)
      est_srs[b] <- mean(y[s_srs])
    }

    estimates <- list(
      "Proportional allocation" = est_prop,
      "Neyman allocation" = est_neyman,
      "SRSWOR baseline" = est_srs
    )
    mse_srs <- mse(est_srs, mu)

    for (method in names(estimates)) {
      est <- estimates[[method]]
      rows[[counter]] <- data.frame(
        section = "Stratified random sampling",
        method = method,
        n = n,
        true_mean = mu,
        empirical_mean = mean(est),
        bias = mean(est) - mu,
        empirical_variance = var(est),
        empirical_mse = mse(est, mu),
        relative_efficiency_vs_srs = mse_srs / mse(est, mu),
        allocation = if (method == "Proportional allocation") {
          paste(n_prop, collapse = "-")
        } else if (method == "Neyman allocation") {
          paste(n_neyman, collapse = "-")
        } else {
          "not applicable"
        }
      )
      counter <- counter + 1L
    }
  }

  strat_summary <- do.call(rbind, rows)
  write_result(strat_summary, "stratified_performance.csv")
  plot_lines(strat_summary, "n", "relative_efficiency_vs_srs", "method",
             "Stratified sampling relative efficiency", "Total sample size n",
             "MSE(SRSWOR) / MSE(method)", "fig_stratified_relative_efficiency.png",
             legend_pos = "bottomright")
  strat_summary
}

# -----------------------------------------------------------------------------
# 4. Cluster sampling design
# -----------------------------------------------------------------------------

simulate_cluster <- function() {
  M <- 400
  m <- 20
  N <- M * m
  within_cluster_n <- 5
  total_n_set <- c(100, 200, 400)
  icc_set <- c(0.1, 0.3, 0.6)
  total_var <- 100
  cluster_id <- rep(seq_len(M), each = m)
  rows <- list()
  counter <- 1L

  for (icc in icc_set) {
    sigma_cluster <- sqrt(total_var * icc)
    sigma_error <- sqrt(total_var * (1 - icc))
    cluster_effect <- rnorm(M, mean = 0, sd = sigma_cluster)
    y <- 50 + cluster_effect[cluster_id] + rnorm(N, mean = 0, sd = sigma_error)
    mu <- mean(y)
    actual_icc <- sigma_cluster^2 / (sigma_cluster^2 + sigma_error^2)

    for (total_n in total_n_set) {
      clusters_to_sample <- total_n / within_cluster_n
      est_cluster <- numeric(iter)
      est_srs <- numeric(iter)

      for (b in seq_len(iter)) {
        chosen_clusters <- sample.int(M, clusters_to_sample, replace = FALSE)
        chosen_units <- unlist(lapply(chosen_clusters, function(cl) {
          idx <- which(cluster_id == cl)
          sample(idx, within_cluster_n, replace = FALSE)
        }))
        est_cluster[b] <- mean(y[chosen_units])

        s_srs <- sample.int(N, total_n, replace = FALSE)
        est_srs[b] <- mean(y[s_srs])
      }

      mse_cluster <- mse(est_cluster, mu)
      mse_srs <- mse(est_srs, mu)

      rows[[counter]] <- data.frame(
        section = "Cluster sampling design",
        method = "Cluster sampling",
        target_icc = icc,
        actual_icc = actual_icc,
        total_secondary_units = total_n,
        clusters_sampled = clusters_to_sample,
        within_cluster_sample_size = within_cluster_n,
        true_mean = mu,
        empirical_mean = mean(est_cluster),
        bias = mean(est_cluster) - mu,
        empirical_variance = var(est_cluster),
        empirical_mse = mse_cluster,
        relative_efficiency_vs_srs = mse_srs / mse_cluster
      )
      counter <- counter + 1L

      rows[[counter]] <- data.frame(
        section = "Cluster sampling design",
        method = "SRSWOR baseline",
        target_icc = icc,
        actual_icc = actual_icc,
        total_secondary_units = total_n,
        clusters_sampled = NA,
        within_cluster_sample_size = NA,
        true_mean = mu,
        empirical_mean = mean(est_srs),
        bias = mean(est_srs) - mu,
        empirical_variance = var(est_srs),
        empirical_mse = mse_srs,
        relative_efficiency_vs_srs = 1
      )
      counter <- counter + 1L
    }
  }

  cluster_summary <- do.call(rbind, rows)
  write_result(cluster_summary, "cluster_performance.csv")

  cluster_plot <- cluster_summary[cluster_summary$total_secondary_units == 200, ]
  plot_lines(cluster_plot, "target_icc", "empirical_mse", "method",
             "Cluster sampling and SRSWOR MSE at total n = 200",
             "Target intra-cluster correlation", "Empirical MSE",
             "fig_cluster_mse_icc_n200.png", legend_pos = "topleft")

  cluster_re <- cluster_summary[cluster_summary$method == "Cluster sampling", ]
  plot_lines(cluster_re, "target_icc", "relative_efficiency_vs_srs", "total_secondary_units",
             "Cluster sampling relative efficiency", "Target intra-cluster correlation",
             "MSE(SRSWOR) / MSE(cluster)", "fig_cluster_relative_efficiency.png",
             legend_pos = "topright")

  cluster_summary
}

cat("Running simple random sampling simulations...\n")
srs_results <- simulate_srs()
cat("Running ratio estimation simulations...\n")
ratio_results <- simulate_ratio()
cat("Running stratified sampling simulations...\n")
stratified_results <- simulate_stratified()
cat("Running cluster sampling simulations...\n")
cluster_results <- simulate_cluster()

settings <- data.frame(
  item = c("random_seed", "iterations", "SRS_N", "ratio_N", "stratified_N",
           "cluster_M", "cluster_m", "cluster_within_sample_size"),
  value = c("20260627", iter, 5000, 6000, 5000, 400, 20, 5)
)
write_result(settings, "simulation_settings.csv")

sink(file.path(out_tables, "session_info.txt"))
print(sessionInfo())
sink()

cat("Done. Tables are in output/tables and figures are in output/figures.\n")
