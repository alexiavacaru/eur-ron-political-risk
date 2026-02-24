# 02_breaks_garch.R
# goal:
# - check stationarity (ADF)
# - find structural breaks (Bai-Perron)
# - fit a simple GARCH(1,1) model
# - save conditional volatility (sigma)

library(readr)
library(lubridate)
library(tseries)
library(strucchange)
library(rugarch)

# ----------------------------
# 1) Load returns data
# ----------------------------
file_path <- "data_processed/fx_returns.csv"

if(!file.exists(file_path)) {
  stop("fx_returns.csv not found. Run 01_cleaning.R first.")
}

fx <- read_csv(file_path, show_col_types = FALSE)
fx$date <- as.Date(fx$date)

# keep only valid returns
rets_df <- fx[!is.na(fx$log_return), c("date", "log_return")]

if(nrow(rets_df) < 30) {
  stop("Not enough observations for breaks/GARCH. Check your data.")
}

rets <- rets_df$log_return

cat("Loaded returns:", nrow(rets_df), "observations\n")
cat("Date range:", as.character(min(rets_df$date)), "to", as.character(max(rets_df$date)), "\n\n")

# ----------------------------
# 2) ADF test (stationarity)
# ----------------------------
cat("ADF stationarity test:\n")
adf_result <- adf.test(rets)
print(adf_result)

# ----------------------------
# 3) Structural breaks (Bai-Perron)
# ----------------------------
cat("\nStructural breaks (Bai-Perron) on returns mean:\n")
bp <- breakpoints(rets ~ 1)
print(summary(bp))

# ----------------------------
# 4) GARCH(1,1) model
# ----------------------------
cat("\nFitting GARCH(1,1)...\n")

spec <- ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
  mean.model     = list(armaOrder = c(0, 0), include.mean = TRUE),
  distribution.model = "norm"
)

fit <- ugarchfit(spec, data = rets, solver = "hybrid")

cat("\nGARCH model fitted successfully.\n")
show(fit)

# extract conditional volatility (sigma)
sigma_t <- as.numeric(sigma(fit))

# ----------------------------
# 5) Save output file
# ----------------------------
garch_out <- data.frame(
  date = rets_df$date,
  garch_sigma = sigma_t
)

write_csv(garch_out, "data_processed/garch_volatility.csv")
cat("\nsaved: data_processed/garch_volatility.csv\n")
