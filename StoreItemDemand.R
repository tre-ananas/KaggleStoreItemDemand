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

#################################################################
#################################################################
# Feature Engineering for a Single Store-Item Combo #############
#################################################################
#################################################################

### Load Data and Packages ###

# Packages
library(vroom)
library(tidyverse)
library(timetk)
library(patchwork)
library(tidyverse)
library(embed)
library(lubridate)
library(parsnip)
library(ranger)
library(workflows)

library(tidyverse)
library(vroom)
library(tidymodels)
library(poissonreg)
library(rpart)
library(stacks)
library(dbarts)
library(xgboost)


# Data
train <- vroom("train.csv")
test <- vroom("test.csv")

# Subset store-item combo w my favorite numbers
storeItem <- train %>%
  filter(store == 4, item == 17)



### Light EDA ###

# Time Series Plot
tsp <- storeItem %>%
  plot_time_series(date, sales, .interactive = FALSE)

# ACF (autocorrelation function plots)
acf <- storeItem %>%
  pull(sales) %>%
  forecast::ggAcf(.) + ggtitle("ACF for Store 4 Item 4")

# ACF-Year
acf_y <- storeItem %>%
  pull(sales) %>%
  forecast::ggAcf(., lag.max = 365) + ggtitle("ACF for Store 4 Item 4")

# ACF-All Time
acf_all <- storeItem %>%
  pull(sales) %>%
  forecast::ggAcf(., lag.max = 5*365) + ggtitle("ACF for Store 4 Item 4")

# Four-Way Plot
fourway <- (tsp + acf) / (acf_y + acf_all)
fourway

# EDA Findings:
# General increase in sales
# Heavy weekly autocorrelation (7 days)
# Biggest sales in the summer
# Smallest sales in the winter



### Recipe ###

# Create Recipe
rec <- recipe(sales ~ ., data = storeItem) %>%
  step_rm(c('store', 'item')) %>%
  step_date(date, features = c('week', 'month', 'quarter')) %>%
  step_mutate_at(all_integer_predictors(), fn = factor) %>%
  step_mutate(season = factor(case_when(
    between(month(date), 3, 5) ~ "Spring",
    between(month(date), 6, 8) ~ "Summer",
    between(month(date), 9, 11) ~ "Fall",
    TRUE ~ "Winter"
  ))) %>%
  step_mutate(cumulative_sales = cumsum(sales)) %>%
  step_dummy(all_nominal_predictors()) # Make nominal predictors into dummy variables

# Prep, Bake, and View Recipe
prepped <- prep(rec)
bake(prepped, storeItem)



### Model: Random Forest ###

# Model
rf_model <- rand_forest(mtry = tune(),
                              min_n = tune(),
                              trees = 500) %>% # Type of Model
  set_engine("ranger") %>% # What R function to use
  set_mode("regression")


# Workflow
rf_workflow <- workflow() %>%
  add_recipe(rec) %>%
  add_model(rf_model)

# Tuning grid
tuning_grid <- grid_regular(mtry(range = c(1, 7)), # Grid of values to tune over
                            min_n(),
                            levels = 5) # levels = L means L^2 total tuning possibilities

folds <- vfold_cv(storeItem, # Split data for CV
                  v = 5, # 5 folds
                  repeats = 1)

# Run CV
cv_results <- rf_workflow %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(smape))

# Find Best Tuning Parameters
best_tune <- cv_results %>%
  select_best("smape")
best_tune
# mtry = 7, min_n = 2

cv_results %>% collect_metrics() %>%
  filter(.metric == "smape")
# mean for best_tune is 18.6




















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
