# 01_cleaning.R
# goal: read EUR/RON FX data and create returns + rolling volatility measures

library(readr)
library(dplyr)
library(lubridate)
library(zoo)

# --- read FX data
fx <- read_csv("data_processed/fx_rates_clean.csv", show_col_types = FALSE)
fx$date <- as.Date(fx$date)
fx <- fx[order(fx$date), ]

# --- compute log returns
fx$log_return <- c(NA, diff(log(fx$fx_rate)))

# --- rolling volatility
# NOTE: data is monthly, but we keep the required names:
# vol_7d  ~ 3 months
# vol_14d ~ 6 months
# vol_30d ~ 12 months
fx$vol_7d  <- rollapply(fx$log_return, 3,  sd, fill = NA, align = "right")
fx$vol_14d <- rollapply(fx$log_return, 6,  sd, fill = NA, align = "right")
fx$vol_30d <- rollapply(fx$log_return, 12, sd, fill = NA, align = "right")

# --- save
write_csv(fx, "data_processed/fx_returns.csv")
cat("saved: data_processed/fx_returns.csv\n")
