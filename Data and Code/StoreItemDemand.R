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

# Use these bookends if parallelizing long processes

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

# Load Libraries
library(vroom)
library(tidyverse)
library(timetk)
library(patchwork)

# Load Data
train <- vroom("train.csv")
test <- vroom("test.csv")



# Time Series Plots for 4 different store-item combos because I can't feasibly do all store-item combos
# Store 1 Item 1
train[train$store == 1 & train$item == 1, ] %>%
  plot_time_series(date, sales, .interactive = FALSE)

# Store 2 Item 2
train[train$store == 2 & train$item == 2, ] %>%
  plot_time_series(date, sales, .interactive = FALSE)

# Store 3 Item 3
train[train$store == 3 & train$item == 3, ] %>%
  plot_time_series(date, sales, .interactive = FALSE)

# Store 4 Item 4
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



###################################################################################
###################################################################################
# Feature Engineering and Random Forest for a Single Store-Item Combo #############
###################################################################################
###################################################################################

### Load Data and Packages ###

# Load Packages
library(vroom)
library(tidyverse)
library(timetk)
library(patchwork)
library(embed)
library(lubridate)
library(parsnip)
library(ranger)
library(workflows)
library(tidymodels)
library(poissonreg)
library(rpart)
library(stacks)
library(dbarts)
library(xgboost)

# Load Data
train <- vroom("train.csv")
test <- vroom("test.csv")

# Subset store-item combo w my favorite numbers bc I can't feasibly do all combos
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
  # Remove store and item
  step_rm(c('store', 'item')) %>%
  # Expand date into week, month, and quarter
  step_date(date, features = c('week', 'month', 'quarter')) %>%
  # Turn integer predictors into factors
  step_mutate_at(all_integer_predictors(), fn = factor) %>%
  # Create season factor variable
  step_mutate(season = factor(case_when(
    between(month(date), 3, 5) ~ "Spring",
    between(month(date), 6, 8) ~ "Summer",
    between(month(date), 9, 11) ~ "Fall",
    TRUE ~ "Winter"
  ))) %>%
  # Create a cumulative sales feature
  step_mutate(cumulative_sales = cumsum(sales)) %>%
  # Turn all nominal predictors into dummy variables
  step_dummy(all_nominal_predictors())

# Prep, Bake, and View Recipe
prepped <- prep(rec)
bake(prepped, storeItem)



### Model: Random Forest ###

# Set UP Model
rf_model <- rand_forest(mtry = tune(),
                              min_n = tune(),
                              trees = 500) %>% # 500 trees; tune mtry and min_n
  set_engine("ranger") %>% # Use ranger function
  set_mode("regression") # Regression bc target variable is quantitative

# Set up Workflow
rf_workflow <- workflow() %>%
  add_recipe(rec) %>%
  add_model(rf_model)

# Tuning grid
tuning_grid <- grid_regular(mtry(range = c(1, 7)), # Grid of values to tune over
                            min_n(),
                            levels = 5)

# Split data for CV
folds <- vfold_cv(storeItem,
                  v = 5, # 5 folds
                  repeats = 1)

# Run CV
cv_results <- rf_workflow %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(smape))

# Find Best Tuning Parameters to Optimize SMAPE
best_tune <- cv_results %>%
  select_best("smape")
best_tune
# mtry = 7, min_n = 2

# Look at SMAPE for Best Tuning Parameters
cv_results %>% collect_metrics() %>%
  filter(.metric == "smape")
# mean for best_tune is 18.6



#################################################################
#################################################################
# Exponential Smoothing for a Single Store-Item Combo ###########
#################################################################
#################################################################

# Load Libraries
library(vroom)
library(tidyverse)
library(modeltime)
library(timetk)
library(tidymodels)
library(patchwork)



# Get Data

# Load Data
train <- vroom("train.csv")
test <- vroom("test.csv")

# Subset two store-item combos w my favorite numbers because I can't feasibly/quickly do all combos 
s4_i17 <- train %>%
  filter(store == 4, item == 17)

s6_i13 <- train %>%
  filter(store == 6, item == 13)



# Cross Validation

# CV for store 4 item 17
cv_split_4_17 <- time_series_split(s4_i17,
                                   assess = "3 months",
                                   cumulative = TRUE)
cv_preds_4_17 <- cv_split_4_17 %>%
  tk_time_series_cv_plan() %>% # put into data frame
  plot_time_series_cv_plan(date, sales, .interactive = FALSE)
cv_preds_4_17

# CV for store 6 item 13
cv_split_6_13 <- time_series_split(s6_i13,
                                   assess = "3 months",
                                   cumulative = TRUE)
cv_preds_6_13 <- cv_split_6_13 %>%
  tk_time_series_cv_plan() %>% # put into data frame
  plot_time_series_cv_plan(date, sales, .interactive = FALSE)
cv_preds_6_13



# Exponential Smoothing

# ES for store 4 item 17
es_model_4_17 <- exp_smoothing() %>%
  set_engine('ets') %>%
  fit(sales~date, data = training(cv_split_4_17))

# Cross-validate to tune model
cv_results_4_17 <- modeltime_calibrate(es_model_4_17,
                                       new_data = testing(cv_split_4_17))

# Visualize CV results
cv_results_vis_4_17 <- cv_results_4_17 %>%
  modeltime_forecast(
    new_data = testing(cv_split_4_17),
    actual_data = s4_i17
  ) %>%
  plot_modeltime_forecast(.interactive = FALSE)
cv_results_vis_4_17

# ES for store 6 item 13
es_model_6_13 <- exp_smoothing() %>%
  set_engine('ets') %>%
  fit(sales~date, data = training(cv_split_6_13))

# Cross-validate to tune model
cv_results_6_13 <- modeltime_calibrate(es_model_6_13,
                                       new_data = testing(cv_split_6_13))

# Visualize CV results
cv_results_vis_6_13 <- cv_results_6_13 %>%
  modeltime_forecast(
    new_data = testing(cv_split_6_13),
    actual_data = s6_i13
  ) %>%
  plot_modeltime_forecast(.interactive = FALSE)
cv_results_vis_6_13



# Refit to all data then forecast for store 4 item 17
es_fullfit_4_17 <- cv_results_4_17 %>%
  modeltime_refit(data = s4_i17)

es_preds_4_17 <- es_fullfit_4_17 %>%
  modeltime_forecast(h = '3 months') %>%
  rename(date = .index, sales = .value)%>%
  select(date, sales) %>%
  full_join(., y = test, by = "date") %>%
  select(id, sales)
  
es_fullfit_plot_4_17 <- es_fullfit_4_17 %>%
  modeltime_forecast(h = '3 months', actual_data = s4_i17) %>%
  plot_modeltime_forecast(.interactive = FALSE)
es_fullfit_plot_4_17



# Refit to all data then forecast for store 6 item 13
es_fullfit_6_13 <- cv_results_6_13 %>%
  modeltime_refit(data = s6_i13)

es_preds_6_13 <- es_fullfit_6_13 %>%
  modeltime_forecast(h = '3 months') %>%
  rename(date = .index, sales = .value)%>%
  select(date, sales) %>%
  full_join(., y = test, by = "date") %>%
  select(id, sales)
  
es_fullfit_plot_6_13 <- es_fullfit_6_13 %>%
  modeltime_forecast(h = '3 months', actual_data = s6_i13) %>%
  plot_modeltime_forecast(.interactive = FALSE)
es_fullfit_plot_6_13



# Plots
plotly::subplot(cv_results_vis_4_17, cv_results_vis_6_13, es_fullfit_plot_4_17, es_fullfit_plot_6_13, nrows = 2)

# Four-Way Plot
fourway <- (cv_results_vis_4_17 + cv_results_vis_6_13) / (es_fullfit_plot_4_17 + es_fullfit_plot_6_13)
fourway



#################################################################
#################################################################
# SARIMA for a Couple Store-Item Combos               ###########
#################################################################
#################################################################

# Load Libraries
library(vroom)
library(tidyverse)
library(modeltime)
library(timetk)
library(tidymodels)
library(patchwork)
library(forecast)
library(embed)
library(lubridate)
library(parsnip)
library(workflows)
library(ggplot2)



# Get Data

# Load Data
train <- vroom("train.csv")
test <- vroom("test.csv")

# Subset two store-item combos w my favorite numbers bc I can't quickly work with all combos
s4_i17 <- train %>%
  filter(store == 4, item == 17)

s6_i13 <- train %>%
  filter(store == 6, item == 13)

test_s4_i17 <- test %>%
  filter(store == 4, item == 17)

test_s6_i13 <- test %>%
  filter(store == 6, item == 13)



# Recipe for Linear Model Part

# Create Recipe
rec <- recipe(sales ~ ., data = s4_i17) %>%
  # Remove store and item
  step_rm(c('store', 'item')) %>%
  # Convert all integer predictors into factors
  step_mutate_at(all_integer_predictors(), fn = factor) %>%
  # Create a variable of cumulative sales
  step_mutate(cumulative_sales = cumsum(sales)) %>%
  # Turn nominal predictors into dummy variables
  step_dummy(all_nominal_predictors())

# Prep, Bake, and View Recipe
prepped <- prep(rec)
bake(prepped, s4_i17)



# Cross Validation Splits

# CV for store 4 item 17
cv_split_4_17 <- time_series_split(s4_i17,
                                   assess = "3 months",
                                   cumulative = TRUE)
cv_preds_4_17 <- cv_split_4_17 %>%
  tk_time_series_cv_plan() %>% # put into data frame
  plot_time_series_cv_plan(date, sales, .interactive = FALSE)
cv_preds_4_17

# CV for store 6 item 13
cv_split_6_13 <- time_series_split(s6_i13,
                                   assess = "3 months",
                                   cumulative = TRUE)
cv_preds_6_13 <- cv_split_6_13 %>%
  tk_time_series_cv_plan() %>% # put into data frame
  plot_time_series_cv_plan(date, sales, .interactive = FALSE)
cv_preds_6_13



# ARIMA

# ARIMA Model
arima_model <- arima_reg(seasonal_period = 365,
                         non_seasonal_ar = 5, # default max p to tune
                         non_seasonal_ma = 5, # default max q to tune
                         seasonal_ar = 2, # default max P to tune
                         seasonal_ma = 2, # default max Q to tune
                         non_seasonal_differences = 2, # default max d to tune
                         seasonal_differences = 2) %>% # default max D to tune
  set_engine('auto_arima')

# ARIMA Workflows
arima_wf_4_17 <- workflow() %>%
  add_recipe(rec) %>%
  add_model(arima_model) %>%
  fit(data = training(cv_split_4_17))

arima_wf_6_13 <- workflow() %>%
  add_recipe(rec) %>%
  add_model(arima_model) %>%
  fit(data = training(cv_split_6_13))




# Calibrate/Tune Workflows
cv_results_4_17 <- modeltime_calibrate(arima_wf_4_17,
                                       new_data = testing(cv_split_4_17))

cv_results_6_13 <- modeltime_calibrate(arima_wf_6_13,
                                       new_data = testing(cv_split_6_13))

# Visualize and Evaluate CV Accuracies
cv_results_vis_4_17 <- cv_results_4_17 %>%
  modeltime_forecast(
    new_data = testing(cv_split_4_17),
    actual_data = s4_i17
  ) %>%
  plot_modeltime_forecast(.interactive = FALSE) +
  labs(title = 'CV Predictions and True Obs, Store 4 Item 17')
cv_results_vis_4_17

cv_results_vis_6_13 <- cv_results_6_13 %>%
  modeltime_forecast(
    new_data = testing(cv_split_6_13),
    actual_data = s6_i13
  ) %>%
  plot_modeltime_forecast(.interactive = FALSE) +
  labs(title = 'CV Predictions and True Obs, Store 6 Item 13')
cv_results_vis_6_13



# Refit Best Model to Entire Data and Predict for Each Combo

# S4 I17
arima_fullfit_4_17 <- cv_results_4_17 %>%
  modeltime_refit(data = s4_i17)

arima_forecast_plot_4_17 <- arima_fullfit_4_17 %>%
  modeltime_forecast(
    new_data = test_s4_i17,
    actual_data = s4_i17
  ) %>%
  plot_modeltime_forecast(.interactive = FALSE) +
  labs(title = '3-Month Forecast, Store 4 Item 17')
arima_forecast_plot_4_17

# S6 I13
arima_fullfit_6_13 <- cv_results_6_13 %>%
  modeltime_refit(data = s6_i13)

arima_forecast_plot_6_13 <- arima_fullfit_6_13 %>%
  modeltime_forecast(
    new_data = test_s6_i13,
    actual_data = s6_i13
  ) %>%
  plot_modeltime_forecast(.interactive = FALSE) +
  labs(title = '3-Month Forecast, Store 6 Item 13')
arima_forecast_plot_6_13



# Plots
plotly::subplot(cv_results_vis_4_17, cv_results_vis_6_13, arima_forecast_plot_4_17, arima_forecast_plot_6_13, nrows = 2)

# Four-Way Plot
fourway <- (cv_results_vis_4_17 + cv_results_vis_6_13) / (arima_forecast_plot_4_17 + arima_forecast_plot_6_13)
fourway



#################################################################
#################################################################
# PROPHET a Couple Store-Item Combos                  ###########
#################################################################
#################################################################

# Load Libraries
library(vroom)
library(tidyverse)
library(modeltime)
library(timetk)
library(tidymodels)
library(patchwork)
library(forecast)
library(embed)
library(lubridate)
library(parsnip)
library(workflows)
library(ggplot2)



# Get Data

# Load Data
train <- vroom("train.csv")
test <- vroom("test.csv")

# Subset two store-item combos w my favorite numbers; This is quicker and more manageable than all combos
s4_i17 <- train %>%
  filter(store == 4, item == 17)

s6_i13 <- train %>%
  filter(store == 6, item == 13)

test_s4_i17 <- test %>%
  filter(store == 4, item == 17)

test_s6_i13 <- test %>%
  filter(store == 6, item == 13)



# Cross Validation Splits

# CV for store 4 item 17
cv_split_4_17 <- time_series_split(s4_i17,
                                   assess = "3 months",
                                   cumulative = TRUE)
cv_preds_4_17 <- cv_split_4_17 %>%
  tk_time_series_cv_plan() %>% # put into data frame
  plot_time_series_cv_plan(date, sales, .interactive = FALSE)
cv_preds_4_17

# CV for store 6 item 13
cv_split_6_13 <- time_series_split(s6_i13,
                                   assess = "3 months",
                                   cumulative = TRUE)
cv_preds_6_13 <- cv_split_6_13 %>%
  tk_time_series_cv_plan() %>% # put into data frame
  plot_time_series_cv_plan(date, sales, .interactive = FALSE)
cv_preds_6_13




# PROPHET for S4 I17

# ES for store 4 item 17
prophet_model_4_17 <- prophet_reg() %>%
  set_engine('prophet') %>%
  fit(sales~date, data = training(cv_split_4_17))

# Cross-validate to tune model
cv_results_4_17 <- modeltime_calibrate(prophet_model_4_17,
                                       new_data = testing(cv_split_4_17))

# Visualize CV results
cv_results_vis_4_17 <- cv_results_4_17 %>%
  modeltime_forecast(
    new_data = testing(cv_split_4_17),
    actual_data = s4_i17
  ) %>%
  plot_modeltime_forecast(.interactive = FALSE) +
  labs(title = 'CV Predictions and True Obs, Store 4 Item 17')
cv_results_vis_4_17


# PROPHET for S6 I13

# ES for store 6 item 13
prophet_model_6_13 <- prophet_reg() %>%
  set_engine('prophet') %>%
  fit(sales~date, data = training(cv_split_6_13))

# Cross-validate to tune model
cv_results_6_13 <- modeltime_calibrate(prophet_model_6_13,
                                       new_data = testing(cv_split_6_13))

# Visualize CV results
cv_results_vis_6_13 <- cv_results_6_13 %>%
  modeltime_forecast(
    new_data = testing(cv_split_6_13),
    actual_data = s6_i13
  ) %>%
  plot_modeltime_forecast(.interactive = FALSE) +
  labs(title = 'CV Predictions and True Obs, Store 6 Item 13')
cv_results_vis_6_13




# Refit to all data then forecast for store 4 item 17

prophet_fullfit_4_17 <- cv_results_4_17 %>%
  modeltime_refit(data = s4_i17)

prophet_preds_4_17 <- prophet_fullfit_4_17 %>%
  modeltime_forecast(h = '3 months') %>%
  rename(date = .index, sales = .value)%>%
  select(date, sales) %>%
  full_join(., y = test, by = "date") %>%
  select(id, sales)
  
prophet_fullfit_plot_4_17 <- prophet_fullfit_4_17 %>%
  modeltime_forecast(h = '3 months', actual_data = s4_i17) %>%
  plot_modeltime_forecast(.interactive = FALSE) +
  labs(title = '3-Month Forecast, Store 4 Item 17')
prophet_fullfit_plot_4_17


# Refit to all data then forecast for store 6 item 13
prophet_fullfit_6_13 <- cv_results_6_13 %>%
  modeltime_refit(data = s6_i13)

prophet_preds_6_13 <- prophet_fullfit_6_13 %>%
  modeltime_forecast(h = '3 months') %>%
  rename(date = .index, sales = .value)%>%
  select(date, sales) %>%
  full_join(., y = test, by = "date") %>%
  select(id, sales)
  
prophet_fullfit_plot_6_13 <- prophet_fullfit_6_13 %>%
  modeltime_forecast(h = '3 months', actual_data = s6_i13) %>%
  plot_modeltime_forecast(.interactive = FALSE) +
  labs(title = '3-Month Forecast, Store 6 Item 13')
prophet_fullfit_plot_6_13



# Plots
plotly::subplot(cv_results_vis_4_17, cv_results_vis_6_13, prophet_fullfit_plot_4_17, prophet_fullfit_plot_6_13, nrows = 2)

# Four-Way Plot
fourway <- (cv_results_vis_4_17 + cv_results_vis_6_13) / (prophet_fullfit_plot_4_17 + prophet_fullfit_plot_6_13)
fourway

#################################################################
#################################################################
# BOOSTING                                          #############
#################################################################
#################################################################

# Load Libraries -------------
library(tidyverse)
library(tidymodels)
library(modeltime)
library(timetk)
library(vroom)
library(embed)
library(bonsai)
library(lightgbm)



# Load Data -------------
train <- vroom("train.csv")
test <- vroom("test.csv")



# Recipe -------------

# Create Recipe
rec <- recipe(sales~., data=train) %>%
  # Expand date into dow, month, decimal, doy, year, and quarter
  step_date(date, features=c("dow", "month", "decimal", "doy", "year", "quarter")) %>%
  # Set doy to the range [0, pi]
  step_range(date_doy, min=0, max=pi) %>%
  # Create variables holding values equal to sin(doy) and cos(doy) to account for sine and cosine shaped trends
  step_mutate(sinDOY=sin(date_doy), cosDOY=cos(date_doy)) %>%
  # Create a factor variable for season
  step_mutate(season = factor(case_when(
    between(month(date), 3, 5) ~ "Spring",
    between(month(date), 6, 8) ~ "Summer",
    between(month(date), 9, 11) ~ "Fall",
    TRUE ~ "Winter"
  ))) %>%
  # Target encode all nominal predictors
  step_lencode_mixed(all_nominal_predictors(), outcome=vars(sales)) %>%
  # Remove date, item, and store
  step_rm(date, item, store) %>%
  # Normalize all numeric predictors so they have mean = 0 and SD = 1
  step_normalize(all_numeric_predictors())

# Prep, Bake, and View Recipe
prepped <- prep(rec)
head(bake(prepped, train), 3)



# Randomly Select 5 Store-Item Combos for Testing -------------

# Set seed for reproducibility
set.seed(17)

# Generate 5 random indices
indices <- sample(nrow(train), 5, replace = TRUE)

# Extract 'store' and 'item' values using the random indices
random_pairs <- data.frame(
  store = train$store[indices],
  item = train$item[indices]
)

# Display the random pairs
print(random_pairs)

# Loop through each random pair
for (i in 1:nrow(random_pairs)) {
  pair <- random_pairs[i, ]
  filter_result <- train %>%
    filter(store == pair$store, item == pair$item)
  
  # Create a variable with a dynamic name
  assign(paste0("s", pair$store, "_i", pair$item), filter_result)
}



# General Model Testing---------------------------

# Create a Boost model specification
bst_spec <- boost_tree(trees = 1000, # Keep trees at 1000 and tune tree_depth and learn_rate
                       tree_depth = tune(), 
                       learn_rate = tune()) %>%
  set_engine("lightgbm") %>% # Use function lightgbm
  set_mode("regression") # Regression bc target variable is quantitative

# Create a Boost Workflow
bst_wf <- workflow() %>%
  add_recipe(rec) %>%
  add_model(bst_spec)

# Set up Boost tuning grid
bst_grid <- grid_regular(
  tree_depth(range = c(2, 4)),
  learn_rate(range = c(0, .1), trans=NULL),
  levels = 5
)



# Specific Model Testing-----------------------------------------------------

# Split data for cross-validation (CV) for boost
s1_i8_folds <- vfold_cv(s1_i8, v = 5, repeats = 1) # 5 folds for each CV
s8_i3_folds <- vfold_cv(s8_i3, v = 5, repeats = 1)
s5_i42_folds <- vfold_cv(s5_i42, v = 5, repeats = 1)
s3_i23_folds <- vfold_cv(s3_i23, v = 5, repeats = 1)
s8_i44_folds <- vfold_cv(s8_i44, v = 5, repeats = 1)

# Run cross-validation for all 5 boosted models
s1_i8_cv_results <- bst_wf %>%
  tune_grid(resamples = s1_i8_folds,
            grid = bst_grid,
            metrics = metric_set(smape))
s8_i3_cv_results <- bst_wf %>%
  tune_grid(resamples = s8_i3_folds,
            grid = bst_grid,
            metrics = metric_set(smape))
s5_i42_cv_results <- bst_wf %>%
  tune_grid(resamples = s5_i42_folds,
            grid = bst_grid,
            metrics = metric_set(smape))
s3_i23_cv_results <- bst_wf %>%
  tune_grid(resamples = s3_i23_folds,
            grid = bst_grid,
            metrics = metric_set(smape))
s8_i44_cv_results <- bst_wf %>%
  tune_grid(resamples = s8_i44_folds,
            grid = bst_grid,
            metrics = metric_set(smape))

# Find best tuning parameters to optimize smape for each store-item combo
s1_i8_best_tune <- s1_i8_cv_results %>%
  select_best("smape")
s8_i3_best_tune <- s8_i3_cv_results %>%
  select_best("smape")
s5_i42_best_tune <- s5_i42_cv_results %>%
  select_best("smape")
s3_i23_best_tune <- s3_i23_cv_results %>%
  select_best("smape")
s8_i44_best_tune <- s8_i44_cv_results %>%
  select_best("smape")

# Display Best Tune Parameters to Find Consensus
s1_i8_best_tune
s8_i3_best_tune
s5_i42_best_tune
s3_i23_best_tune
s8_i44_best_tune
# Tree Depth = 2
# Learning Rate = .025

# Now plug these tuning parameters into a Kaggle notebook to submit for all store-item combos
# https://www.kaggle.com/code/rwolff17/store-item-demand-boosted



#################################################################
#################################################################
# BOOSTING w XGBOOST                                #############
#################################################################
#################################################################

# Load Libraries -----------------------
library(tidyverse)
library(tidymodels)
library(modeltime)
library(timetk)
library(vroom)
library(embed)
library(bonsai)
library(lightgbm)
library(xgboost)



# Load Data -----------------------
train <- vroom("train.csv")
test <- vroom("test.csv")



# Recipe -----------------------

# Create Recipe
rec <- recipe(sales~., data=train) %>%
  # Expand date into dow, month, decimal, doy, year, and quarter
  step_date(date, features=c("dow", "month", "decimal", "doy", "year", "quarter")) %>%
  # Set doy to the range [0, pi]
  step_range(date_doy, min=0, max=pi) %>%
  # Create variables holding values equal to sin(doy) and cos(doy) to account for sine and cosine shaped trends
  step_mutate(sinDOY=sin(date_doy), cosDOY=cos(date_doy)) %>%
  # Create a factor variable for season
  step_mutate(season = factor(case_when(
    between(month(date), 3, 5) ~ "Spring",
    between(month(date), 6, 8) ~ "Summer",
    between(month(date), 9, 11) ~ "Fall",
    TRUE ~ "Winter"
  ))) %>%
  # Target encode all nominal predictors
  step_lencode_mixed(all_nominal_predictors(), outcome=vars(sales)) %>%
  # Remove date, item, and store
  step_rm(date, item, store) %>%
  # Normalize all numeric predictors so they have mean = 0 and SD = 1
  step_normalize(all_numeric_predictors())

# Prep, Bake, and View Recipe
prepped <- prep(rec)
head(bake(prepped, train), 3)



# Randomly Generate 5 Store-Item Combos for Testing --------------

# Set seed for reproducibility
set.seed(17)

# Generate 5 random indices
indices <- sample(nrow(train), 5, replace = TRUE)

# Extract 'store' and 'item' values using the random indices
random_pairs <- data.frame(
  store = train$store[indices],
  item = train$item[indices]
)

# Display the random pairs
print(random_pairs)

# Loop through each random pair
for (i in 1:nrow(random_pairs)) {
  pair <- random_pairs[i, ]
  filter_result <- train %>%
    filter(store == pair$store, item == pair$item)
  
  # Create a variable with a dynamic name
  assign(paste0("s", pair$store, "_i", pair$item), filter_result)
}




# General Model Testing---------------------------

# Create a boosted model specification
xgb_spec <- boost_tree(trees = 1000, # Use 1000 trees and tune tree_depth, min_n, and learn_rate
                       tree_depth = tune(), 
                       min_n = tune(), 
                       learn_rate = tune()) %>%
  set_engine("xgboost") %>% # Use xgboost function
  set_mode("regression") # Regression bc target variable is quantitative

# Create a Boost Workflow
xgb_wf <- workflow() %>%
  add_recipe(rec) %>%
  add_model(xgb_spec)

# Set up Boost tuning grid
xgb_grid <- grid_regular(
  tree_depth(range = c(1, 5)),
  min_n(range = c(1, 5)),
  learn_rate(range = c(0.01, 0.1), trans = NULL),
  levels = 5
)

# Specific Model Texting-----------------------------------------------------

# Split data for cross-validation (CV) for each model
s1_i8_folds <- vfold_cv(s1_i8, v = 5, repeats = 1) # 5 folds in CV for each store-item combo
s8_i3_folds <- vfold_cv(s8_i3, v = 5, repeats = 1)
s5_i42_folds <- vfold_cv(s5_i42, v = 5, repeats = 1)
s3_i23_folds <- vfold_cv(s3_i23, v = 5, repeats = 1)
s8_i44_folds <- vfold_cv(s8_i44, v = 5, repeats = 1)

# Run cross-validation for each combo
s1_i8_cv_results <- xgb_wf %>%
  tune_grid(resamples = s1_i8_folds,
            grid = xgb_grid,
            metrics = metric_set(smape))
s8_i3_cv_results <- xgb_wf %>%
  tune_grid(resamples = s8_i3_folds,
            grid = xgb_grid,
            metrics = metric_set(smape))
s5_i42_cv_results <- xgb_wf %>%
  tune_grid(resamples = s5_i42_folds,
            grid = xgb_grid,
            metrics = metric_set(smape))
s3_i23_cv_results <- xgb_wf %>%
  tune_grid(resamples = s3_i23_folds,
            grid = xgb_grid,
            metrics = metric_set(smape))
s8_i44_cv_results <- xgb_wf %>%
  tune_grid(resamples = s8_i44_folds,
            grid = xgb_grid,
            metrics = metric_set(smape))

# Find best tuning parameters to optimize SMAPE for each store-item combo
s1_i8_best_tune <- s1_i8_cv_results %>%
  select_best("smape")
s8_i3_best_tune <- s8_i3_cv_results %>%
  select_best("smape")
s5_i42_best_tune <- s5_i42_cv_results %>%
  select_best("smape")
s3_i23_best_tune <- s3_i23_cv_results %>%
  select_best("smape")
s8_i44_best_tune <- s8_i44_cv_results %>%
  select_best("smape")

# Display Best Tuning Parameters for each Store-Item Combo to find Consensus
s1_i8_best_tune
s8_i3_best_tune
s5_i42_best_tune
s3_i23_best_tune
s8_i44_best_tune
# min_n = 5
# Tree Depth = 2
# Learning Rate = .01

# Now plug these tuning parameters into a Kaggle notebook to submit for all store-item combos
# https://www.kaggle.com/code/rwolff17/store-item-demand-xgboost



#################################################################
#################################################################
# END OF CODE                                       #############
#################################################################
#################################################################