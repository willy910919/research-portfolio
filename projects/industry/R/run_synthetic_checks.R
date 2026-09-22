# New synthetic tests for unchanged original pure functions. Run from projects/industry.
source('R/single_factor_reliability_operator.R', encoding = 'UTF-8')
source('R/growth_tables.R', encoding = 'UTF-8')

n <- 20L
base <- rep(100, n)
signal <- seq(.01, .20, length.out = n)
full <- base * exp(signal)
actual <- base * exp(.4 * signal)
args <- list(full_forecast = full, ablated_forecast = base, actual = actual,
             active = rep(TRUE, n), decay = .95,
             prior_equivalent_quarters = 4, signal_cap_quantile = .9)
a <- do.call(single_factor_reliability_operator, args)
stopifnot(all(a$Reliability_weight >= 0 & a$Reliability_weight <= 1))
stopifnot(identical(a$Reliability_history_N, 0:19))
stopifnot(all(is.infinite(a$Historical_signal_cap[1:8])))
# A future actual must not modify weights/forecasts at an earlier origin.
args$actual[20] <- 1e8
b <- do.call(single_factor_reliability_operator, args)
stopifnot(identical(a, b))
args$actual[19] <- 1e8
b <- do.call(single_factor_reliability_operator, args)
stopifnot(identical(a[1:19, ], b[1:19, ]))

months <- format(seq(as.Date('2025-01-01'), as.Date('2027-12-01'), by = 'month'), '%Y-%m')
input <- do.call(rbind, lapply(seq_along(.growth_industries), function(j) {
  data.frame(industry = .growth_industries[j], month = months,
    value = (100+j*10)*1.01^(0:35), status = c(rep('實際值',18),rep('預測值',18)),
    model = 'synthetic_geometric_path', as_of = '2026-06',
    source_cell = paste0('synthetic_',seq_len(36)), source_file = 'generated_in_memory',
    unit = 'synthetic_index', vintage = 'artificial_fixture', stringsAsFactors = FALSE)
}))
result <- calculate_growth_tables(input)
stopifnot(identical(vapply(result,nrow,integer(1)),c(monthly=120L,quarterly=40L,annual=10L)))
stopifnot(max(abs(result$monthly[['月增率(%)']] - 1)) < 1e-10)
stopifnot(max(abs(result$monthly[['年增率(%)']] - (1.01^12-1)*100)) < 1e-10)
stopifnot(max(abs(result$quarterly[['季增率(%)']] - (1.01^3-1)*100)) < 1e-10)
stopifnot(max(abs(result$annual[['年增率(%)']] - (1.01^12-1)*100)) < 1e-10)
stopifnot(all(result$annual[['資料性質']][result$annual[['年度']] == 2026] == '實際值＋預測值'))
cat('PASS: quarterly reliability temporal checks and monthly/quarterly/annual growth arithmetic.\n')
cat('Synthetic inputs only; no original levels, forecasts, source cells, or workbooks.\n')
