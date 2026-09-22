# Original pure computation excerpt; see PROVENANCE.json.

single_factor_reliability_operator <- function(
    full_forecast,
    ablated_forecast,
    actual,
    active,
    decay,
    prior_equivalent_quarters,
    signal_cap_quantile) {
  n <- length(full_forecast)
  raw_signal <- log(full_forecast / ablated_forecast)
  realized_increment <- log(actual / ablated_forecast)
  reliability_weight <- rep(1, n)
  history_n <- integer(n)
  cap_value <- rep(Inf, n)
  capped_signal <- raw_signal
  adjusted_signal <- raw_signal
  log_forecast_adjustment <- numeric(n)

  for (target_i in seq_len(n)) {
    if (!active[target_i]) next
    past <- which(seq_len(n) < target_i & active)
    history_n[target_i] <- length(past)

    if (length(past) >= 4L) {
      recency_weight <- decay^(target_i - 1L - past)
      signal_scale <- stats::median(raw_signal[past]^2)
      prior_strength <-
        prior_equivalent_quarters * max(signal_scale, 1e-8)
      numerator <- prior_strength + sum(
        recency_weight * raw_signal[past] * realized_increment[past]
      )
      denominator <- prior_strength + sum(
        recency_weight * raw_signal[past]^2
      )
      reliability_weight[target_i] <- min(1, max(0, numerator / denominator))
    }

    # This reproduces the E23-specific implementation exactly: no cap is used
    # before eight prior active observations exist.
    if (length(past) >= 8L && signal_cap_quantile < 1) {
      cap_value[target_i] <- as.numeric(stats::quantile(
        abs(raw_signal[past]),
        probs = signal_cap_quantile,
        names = FALSE,
        type = 8
      ))
      capped_signal[target_i] <-
        sign(raw_signal[target_i]) *
        min(abs(raw_signal[target_i]), cap_value[target_i])
    }

    adjusted_signal[target_i] <-
      reliability_weight[target_i] * capped_signal[target_i]
    log_forecast_adjustment[target_i] <-
      adjusted_signal[target_i] - raw_signal[target_i]
  }

  data.frame(
    Raw_marginal_log_signal = raw_signal,
    Reliability_weight = reliability_weight,
    Reliability_history_N = history_n,
    Historical_signal_cap = cap_value,
    Capped_marginal_log_signal = capped_signal,
    Adjusted_marginal_log_signal = adjusted_signal,
    Log_forecast_adjustment = log_forecast_adjustment,
    Signal_was_capped =
      abs(raw_signal) > abs(capped_signal) + 1e-12,
    stringsAsFactors = FALSE
  )
}
