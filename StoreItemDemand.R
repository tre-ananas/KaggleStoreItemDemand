####################################################################################################
####################################################################################################
# Store Item Demand (Kaggle)                                             ###########################
# Ryan Wolff                                                             ###########################
# 13 November 2023                                                       ###########################
# Data Location and Description:                                         ###########################
# https://www.kaggle.com/competitions/demand-forecasting-kernels-only/overview #####################
####################################################################################################
####################################################################################################

########################################################################
########################################################################
#############################
# Start run in parallel
# cl <- makePSOCKcluster(3)
# registerDoParallel(cl)

# End run in parallel
# stopCluster(cl)
#############################
########################################################################
########################################################################

#################################################################
#################################################################
# EDA                                               #############
#################################################################
#################################################################

library(vroom)
library(tidyverse)
library(timetk)
library(patchwork)

train <- vroom("train.csv")
test <- vroom("test.csv")

# Time Series Plots for 4 different store-item combos
train[train$store == 1 & train$item == 1, ] %>%
  plot_time_series(date, sales, .interactive = FALSE)

train[train$store == 2 & train$item == 2, ] %>%
  plot_time_series(date, sales, .interactive = FALSE)

train[train$store == 3 & train$item == 3, ] %>%
  plot_time_series(date, sales, .interactive = FALSE)

train[train$store == 4 & train$item == 4, ] %>%
  plot_time_series(date, sales, .interactive = FALSE)

# ACF (autocorrelation function plots) for the same
# 4 different store-item combos
acf_1_1 <- train[train$store == 1 & train$item == 1, ] %>%
  pull(sales) %>%
  forecast::ggAcf(.) + ggtitle("ACF for Store 1 Item 1")

acf_2_2 <- train[train$store == 2 & train$item == 2, ] %>%
  pull(sales) %>%
  forecast::ggAcf(.) + ggtitle("ACF for Store 2 Item 2")

acf_3_3 <- train[train$store == 3 & train$item == 3, ] %>%
  pull(sales) %>%
  forecast::ggAcf(.) + ggtitle("ACF for Store 3 Item 3")

acf_4_4 <- train[train$store == 4 & train$item == 4, ] %>%
  pull(sales) %>%
  forecast::ggAcf(.) + ggtitle("ACF for Store 4 Item 4")

# Four-way plot of ACF plots to show differences
# in the autocorrelation structure for the different
# items and justify treating the problem as 200
# different time series problems (one for each
# store-item combo)
fourway_acf <- (acf_1_1 + acf_2_2) / (acf_3_3 + acf_4_4)
fourway_acf


# For later use (making predictions)
####################################################

# Treat the problem as if we have 200
# time series (one for each store-item combo)
nStores <- max(train$store)
nItems <- max(train$item)
for(s in 11:nStores) {
  for(i in 1:nItems) {
    storeItemtrain <- train %>%
      filter(store == s, item == i)
    storeItemTest <- test %>%
      filter(store == s, item == i)
  }
  
  ## Fit storeItem models here
  
  ## Predict storeItem sales
  
  ## Save storeItem predictions
  
  if(s == 1 & i == 1) {
    all_preds <- preds
  } else {
    all_preds <- bind_rows(all_preds, preds)
  }
  
}
