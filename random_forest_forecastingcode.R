# =========================================
# 0. Load required packages
# =========================================
library(caret)      # For machine learning workflow
library(dplyr)      # Data manipulation
library(ranger)     # Fast Random Forest implementation
library(data.table) # Fast CSV reading and manipulation
library(brms)       # Bayesian regression models
library(tidyr)      # Data tidying
library(doParallel) # Parallel computation

# =========================================
# 1. Load and prepare FIA disturbance data
# =========================================
# Read main FIA dataset for disturbance code DSTRBCD1
dstrbcd1 <- fread("FIA_forecasting_data_DSTRBCD1.csv")

# Read and process ecoregion dataset
ecorest <- fread("ecorestFIAforecasting_fixchngtpa_include_PREV_TPA.csv") %>%
  group_by(series) %>%
  sample_n(1) %>%             # Sample one row per series
  ungroup() %>%
  group_by(ECODE_NAME) %>%
  filter(n() > 20) %>%        # Keep ECODE_NAME with sufficient observations
  ungroup() %>%
  mutate(PLT_CN = as.character(PLT_CN)) # Ensure PLT_CN is character

# Read previous TPA per tree type
prevtpa <- fread("FIA_treetype_prevtpa_pivot.csv") %>%
  mutate(PLT_CN = as.character(PLT_CN)) %>%
  rename(Pole_prev_tpa = Pole, Mature_prev_tpa = Mature) # Rename for clarity

# Merge ecoregion data with previous TPA
ecorest_rf <- ecorest %>%
  # Map numeric DSTRBCD1 codes to descriptive disturbance types
  mutate(Disturbance_Type = case_when(
    DSTRBCD1 == 0 ~ "No Disturbance",
    DSTRBCD1 == 10 ~ "Insect Damage",
    DSTRBCD1 == 80 ~ "Human-Caused Damage",
    DSTRBCD1 == 22 ~ "Disease Damage",
    DSTRBCD1 == 30 ~ "Fire (Crown and Ground)",
    DSTRBCD1 == 31 ~ "Ground Fire",
    DSTRBCD1 == 32 ~ "Crown Fire",
    DSTRBCD1 %in% c(46, 40) ~ "Animal Damage/Grazing",
    DSTRBCD1 == 52 ~ "Wind"
  )) %>%
  drop_na(Disturbance_Type) %>%  # Drop rows with undefined disturbance type
  inner_join(prevtpa)             # Join previous TPA by PLT_CN

# Load climate data and merge
climate <- fread("ECODE_NAME_ALLPLOTS_climate.csv") %>%
  mutate(PLT_CN = as.character(PLT_CN))

ecorest_rf <- ecorest_rf %>%
  inner_join(climate) # Merge climate data

# =========================================
# 2. Set up parallel backend for training
# =========================================
cl <- makePSOCKcluster(30) # Use 30 cores
registerDoParallel(cl)

# Cross-validation setup: repeated 10-fold CV, 3 repeats
ctrl <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 3,
  savePredictions = "final",   # Save out-of-fold predictions
  verboseIter = TRUE,
  allowParallel = TRUE
)

# =========================================
# 3. Define Random Forest hyperparameter grid
# =========================================
grid1 <- expand.grid(
  mtry = seq(10, 35, 50),      # Number of variables to possibly split at each node
  splitrule = "variance",       # Splitting rule for regression
  min.node.size = 5             # Minimum node size
)

# =========================================
# 4. Train Random Forest model
# =========================================
rf_model <- train(
  annual_change ~ previous_mean + YEAR + Pole_chng_perc + Mature_chng_perc +
    Pole_prev_tpa + Mature_prev_tpa +
    Disturbance_Type + Forest_Type_Group + mean_temp + total_prec + ECO_NAME,
  data = ecorest_rf %>%
    select(
      annual_change, previous_mean, YEAR, Pole_chng_perc, Mature_chng_perc,
      Pole_prev_tpa, Mature_prev_tpa, Disturbance_Type, Forest_Type_Group,
      mean_temp, total_prec, ECO_NAME
    ) %>%
    drop_na(),  # Ensure no missing data
  method = "ranger",       # Use ranger for fast RF
  trControl = ctrl,        # CV control
  tuneGrid = grid1,        # Grid search
  importance = "permutation", # Feature importance by permutation
  quantreg = TRUE          # Quantile regression enabled
)

# =========================================
# 5. Save trained RF model
# =========================================
save(
  rf_model,
  file = "rf_mode_importance_mtry_quantregtrue_econameincl_treatfix_climateecoregincl_alldata.RData"
)
