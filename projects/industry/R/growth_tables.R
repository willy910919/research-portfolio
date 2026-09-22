# Original pure computation excerpt; see PROVENANCE.json.

.growth_industries <- c("製造業", "金屬機電工業", "資訊電子工業", "化學工業", "民生工業")

.growth_status <- function(x) {
  if (all(x == "實際值")) "實際值" else if (all(x == "預測值")) "預測值" else "實際值＋預測值"
}
.growth_row <- function(...) as.data.frame(list(...), check.names = FALSE, stringsAsFactors = FALSE)
.growth_stack <- function(x) { ans <- do.call(rbind, x); rownames(ans) <- NULL; ans }
.growth_rate <- function(current, previous) (current / previous - 1) * 100

calculate_growth_tables <- function(input) {
  monthly <- quarterly <- annual <- list()
  for (industry in .growth_industries) {
    d <- input[input$industry == industry, , drop = FALSE]
    # 月增率與年增率均直接比較產值，不是比較已四捨五入的成長率。
    for (i in 13:36) {
      monthly[[length(monthly) + 1L]] <- .growth_row(
        產業 = industry, 月份 = d$month[i], 產值 = d$value[i],
        `月增率(%)` = .growth_rate(d$value[i], d$value[i - 1L]),
        `年增率(%)` = .growth_rate(d$value[i], d$value[i - 12L]),
        資料性質 = d$status[i], 上月月份 = d$month[i - 1L], 上月產值 = d$value[i - 1L],
        上月資料性質 = d$status[i - 1L], 去年同月 = d$month[i - 12L], 去年同月產值 = d$value[i - 12L],
        去年同月資料性質 = d$status[i - 12L], 模型 = d$model[i], 資料截止 = d$as_of[i],
        是否季調 = "否", 數值單位 = d$unit[i],
        計算方式 = "月增率=(本月/上月−1)×100；年增率=(本月/去年同月−1)×100",
        來源儲存格 = d$source_cell[i], 來源檔名 = d$source_file[i], 資料版本說明 = d$vintage[i]
      )
    }
    # 先建立2025–2027所有曆季，讓2026Q1可正確使用2025Q4與2025Q1作分母。
    q_values <- q_status <- q_actual <- q_predicted <- vector("list", 12L)
    q_labels <- paste0(rep(2025:2027, each = 4), "Q", rep(1:4, 3))
    for (q in 1:12) {
      ids <- ((q - 1L) * 3L + 1L):(q * 3L)
      q_values[[q]] <- sum(d$value[ids])
      q_status[[q]] <- .growth_status(d$status[ids])
      q_actual[[q]] <- sum(d$status[ids] == "實際值")
      q_predicted[[q]] <- sum(d$status[ids] == "預測值")
    }
    for (q in 5:12) {
      quarterly[[length(quarterly) + 1L]] <- .growth_row(
        產業 = industry, 季度 = q_labels[q], 季度產值合計 = q_values[[q]],
        `季增率(%)` = .growth_rate(q_values[[q]], q_values[[q - 1L]]),
        `年增率(%)` = .growth_rate(q_values[[q]], q_values[[q - 4L]]),
        資料性質 = q_status[[q]], 前季 = q_labels[q - 1L], 前季產值合計 = q_values[[q - 1L]],
        前季資料性質 = q_status[[q - 1L]], 去年同季 = q_labels[q - 4L],
        去年同季產值合計 = q_values[[q - 4L]], 去年同季資料性質 = q_status[[q - 4L]],
        當季實際月數 = q_actual[[q]], 當季預測月數 = q_predicted[[q]],
        前季實際月數 = q_actual[[q - 1L]], 前季預測月數 = q_predicted[[q - 1L]],
        去年同季實際月數 = q_actual[[q - 4L]], 去年同季預測月數 = q_predicted[[q - 4L]],
        模型 = d$model[1], 資料截止 = d$as_of[1], 是否季調 = "否", 數值單位 = d$unit[1],
        計算方式 = "先合計曆季3個月產值；季增率=(本季合計/前季合計−1)×100；年增率=(本季合計/去年同季合計−1)×100",
        來源檔名 = d$source_file[1], 資料版本說明 = d$vintage[1]
      )
    }
    # 全年以12個月產值合計比較，不平均月增率或季增率。
    y_values <- y_status <- y_actual <- y_predicted <- vector("list", 3L)
    for (y in 1:3) {
      ids <- ((y - 1L) * 12L + 1L):(y * 12L)
      y_values[[y]] <- sum(d$value[ids])
      y_status[[y]] <- .growth_status(d$status[ids])
      y_actual[[y]] <- sum(d$status[ids] == "實際值")
      y_predicted[[y]] <- sum(d$status[ids] == "預測值")
    }
    for (y in 2:3) {
      annual[[length(annual) + 1L]] <- .growth_row(
        產業 = industry, 年度 = 2024L + y, 年度產值合計 = y_values[[y]],
        `年增率(%)` = .growth_rate(y_values[[y]], y_values[[y - 1L]]),
        資料性質 = y_status[[y]], 上年度 = 2023L + y, 上年度產值合計 = y_values[[y - 1L]],
        上年度資料性質 = y_status[[y - 1L]], 當年實際月數 = y_actual[[y]], 當年預測月數 = y_predicted[[y]],
        上年實際月數 = y_actual[[y - 1L]], 上年預測月數 = y_predicted[[y - 1L]],
        模型 = d$model[1], 資料截止 = d$as_of[1], 是否季調 = "否", 數值單位 = d$unit[1],
        計算方式 = "先合計全年12個月產值；年增率=(本年合計/上年合計−1)×100",
        來源檔名 = d$source_file[1], 資料版本說明 = d$vintage[1]
      )
    }
  }
  result <- list(monthly = .growth_stack(monthly), quarterly = .growth_stack(quarterly), annual = .growth_stack(annual))
  stopifnot(identical(vapply(result, nrow, integer(1)), c(monthly = 120L, quarterly = 40L, annual = 10L)))
  for (table in result) for (column in table) if (anyNA(column) || (is.numeric(column) && any(!is.finite(column)))) stop("計算結果含缺值或非有限數。", call. = FALSE)
  result
}
