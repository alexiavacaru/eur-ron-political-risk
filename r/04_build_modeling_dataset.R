# 04_build_modeling_dataset.R
# goal: build the final modeling dataset by joining FX returns, GARCH volatility, EPU and event windows

library(readr)
library(dplyr)
library(lubridate)

# 1) Load datasets
fx <- read_csv("data_processed/fx_returns.csv", show_col_types = FALSE) %>%
  mutate(date = as.Date(date)) %>%
  arrange(date)

garch <- read_csv("data_processed/garch_volatility.csv", show_col_types = FALSE) %>%
  mutate(date = as.Date(date)) %>%
  arrange(date)

epu <- read_csv("data_raw/epu.csv", show_col_types = FALSE) %>%
  mutate(date = as.Date(date)) %>%
  arrange(date)

events <- read_csv("data_processed/events.csv", show_col_types = FALSE) %>%
  mutate(event_date = as.Date(event_date)) %>%
  mutate(event_month = floor_date(event_date, "month"))

# 2) Create event_window = 0/1 on monthly dates
event_window_df <- data.frame(date = fx$date, event_window = 0)

for(i in 1:nrow(events)) {

  event_month <- events$event_month[i]
  w <- events$window_months[i]

  start_date <- event_month %m-% months(w)
  end_date   <- event_month %m+% months(w)

  event_window_df$event_window[event_window_df$date >= start_date &
                               event_window_df$date <= end_date] <- 1
}

# 3) Join everything into one table
modeling_dataset <- fx %>%
  left_join(garch, by = "date") %>%
  left_join(epu, by = "date") %>%
  left_join(event_window_df, by = "date")

# 4) Define the target variable (high volatility)
# We use vol_14d as the main volatility proxy
# (in your monthly case, vol_14d = rolling volatility on ~6 months)

threshold <- quantile(modeling_dataset$vol_14d, 0.80, na.rm = TRUE)

modeling_dataset$target_high_volatility <-
  ifelse(modeling_dataset$vol_14d > threshold, 1, 0)

# 5) Save final dataset

write_csv(modeling_dataset, "data_processed/modeling_dataset.csv")
cat("saved: data_processed/modeling_dataset.csv\n")
