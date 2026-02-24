# 03_export_figures.R
# goal:
# - export figures for the report / README
# - returns time series
# - GARCH conditional volatility
# - structural breaks plot (Bai-Perron)

library(readr)
library(lubridate)
library(ggplot2)
library(strucchange)

# 0) Create output folder
dir.create("reports/figures", recursive = TRUE, showWarnings = FALSE)

# 1) Load data
if(!file.exists("data_processed/fx_returns.csv")) {
  stop("fx_returns.csv not found. Run 01_cleaning.R first.")
}

if(!file.exists("data_processed/garch_volatility.csv")) {
  stop("garch_volatility.csv not found. Run 02_breaks_garch.R first.")
}

fx <- read_csv("data_processed/fx_returns.csv", show_col_types = FALSE)
fx$date <- as.Date(fx$date)
fx <- fx[order(fx$date), ]

garch <- read_csv("data_processed/garch_volatility.csv", show_col_types = FALSE)
garch$date <- as.Date(garch$date)
garch <- garch[order(garch$date), ]

# 2) Plot 1: returns time series
p1 <- ggplot(fx, aes(x = date, y = log_return)) +
  geom_line() +
  labs(
    title = "EUR/RON Log Returns",
    x = "Date",
    y = "Log return"
  )

ggsave(
  filename = "reports/figures/returns_timeseries.png",
  plot = p1,
  width = 9,
  height = 4
)

cat("saved: reports/figures/returns_timeseries.png\n")

# 3) Plot 2: GARCH volatility (sigma)
p2 <- ggplot(garch, aes(x = date, y = garch_sigma)) +
  geom_line() +
  labs(
    title = "GARCH(1,1) Conditional Volatility (Sigma)",
    x = "Date",
    y = "Sigma"
  )

ggsave(
  filename = "reports/figures/garch_volatility.png",
  plot = p2,
  width = 9,
  height = 4
)

cat("saved: reports/figures/garch_volatility.png\n")

# 4) Plot 3: structural breaks (Bai-Perron)
rets <- fx$log_return[!is.na(fx$log_return)]

bp <- breakpoints(rets ~ 1)

png("reports/figures/breaks_plot.png", width = 1100, height = 450)
plot(bp, main = "Structural Breaks (Bai-Perron) on EUR/RON Returns")
dev.off()

cat("saved: reports/figures/breaks_plot.png\n")
