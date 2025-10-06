################################################################################
# Harvard Forest Remote Sensing & Field α Estimation Pipeline
# Note:
#  - Assumes helper functions exist in the environment:
#      get_potential_breakpoint_and_kde(),
#      determine_truncation_and_filter(),
#      fit_alpha_model(),
#      estimate_total_trees()
#  - These functions are part of an as-yet unpublished R package. However, the
#     code that these functions are based on can be found in the published paper
#     Eichenwald et al. 2025, Global Ecology and Biogeography, Leveraging Remote Sensing 
#     and Theory to Predict Tree Size Abundance Distributions Across Space
################################################################################

# ---------------------------
# 1. Load required packages
# ---------------------------
library(sf)         # spatial vector handling
library(terra)      # raster handling and zonal stats
library(dplyr)      # data manipulation
# (the following libraries are used later in script; duplicated loads are harmless)
library(purrr)      # map/nest helpers
library(itcSegment) # dbh() estimator from canopy measurements
library(stringr)    # string utilities
library(tibble)     # tibbles
library(data.table) # fast file IO
library(rstan)      # Stan sampling
library(posterior)  # posterior draws helper
library(tidyverse)  # includes ggplot2, tidyr, readr, etc. (used later)
library(brms)       # posterior_epred is used
library(caret)      # dummyVars
library(fs)         # dir_create
library(ranger)     # random forest usage (predict)
library(bayestestR) # overlap metric
library(readr)      # write_csv (if not available)
library(ggplot2)    # plotting

# ---------------------------
# 2. Read segment polygons and CHM
# ---------------------------
# Set working directory for outputs and intermediate files
setwd("C:/Users/adam.local/Documents/ForecastingScaling")

# Read prebuilt Weinsten per-hectare crown segment shapefile for Harvard 2014
# NOTE: file path preserved as requested
harv1 <- read_sf("harv_2014_weinstensegment_hectare.shp")

# Basic cleaning:
# - group_by/ungroup ensures operations that rely on IDhectbest behave consistently
# - filter remove IDhectbest == 0 which likely indicates background or missing
harv1 <- harv1 %>%
  group_by(IDhectbest) %>%
  filter(IDhectbest != 0) %>%
  ungroup()

# Load the NEON Canopy Height Model (CHM) raster for the matching tile
# Keep the original file path exactly
chm <- rast("C:\\Users\\adam.local\\Downloads\\HARV_2014_canopyheight_chm\\NEON_struct-ecosystem\\NEON.D01.HARV.DP3.30015.001.2014-06.basic.20250723T152655Z.RELEASE-2025\\NEON_D01_HARV_DP3_731000_4713000_CHM.tif")

# Convert sf to terra's SpatVector for zonal extraction
harv1 <- vect(harv1)

# Compute maximum CHM (max canopy height) within each crown polygon
# na.rm = TRUE ensures NA pixels don't break the summary
zonalharv <- zonal(chm, harv1, fun = "max", na.rm = TRUE)

# Merge zonal stats (max height) back into the attribute table and rename
harvready <- cbind(
  harv1 %>% data.frame(),
  zonalharv %>%
    rename(Max_Height = NEON_D01_HARV_DP3_731000_4713000_CHM)
)

# Estimate canopy diameter from Perimeter and Area using ellipse approximation
# Then estimate DBH (cm) using itcSegment::dbh which expects H (height, m) and CA
harvready <- harvready %>%
  mutate(
    Diameter = 0.5 * (sqrt(Perimeter^2 - (8 * Area))),     # approximate canopy diameter
    dbh = itcSegment::dbh(H = Max_Height, CA = Diameter)   # estimated DBH (cm)
  )

# ---------------------------
# 3. Prepare for per-hectare α estimation (remote-sensed)
# ---------------------------
# Load additional libraries used below (okay to re-call)
library(purrr)
library(itcSegment)
library(stringr)
library(tibble)
library(data.table)
library(rstan)
library(posterior)

# Initialize lists that will store per-hectare results
alpha_results <- list()      # will store α posterior summary per tile
tree_results <- list()       # will store total tree posterior summary per tile

# Unique hectare IDs to loop over
vector <- harvready$IDhectbest %>% unique()

# Loop over each hectare polygon (remote-sensed crowns)
for (i in vector) {
  site_i <- "HARV"  # site code (static for this dataset)
  # tile_id_i <- tile_info$geo_index[i]  # (legacy / commented out in original)
  
  # --- Get LAI for this site ---
  # Leaf_area_index is assumed to exist in the environment. This pulls the
  # single LAI value for HARV. If none present, next -> warning/skip.
  LAI_i <- Leaf_area_index %>%
    filter(site == site_i) %>%
    pull(Leaf_area_index)
  
  # --- Get prior mean and sd for this site from neon regression outputs ---
  # neonregressionoutput is assumed to exist in the environment
  priors <- neonregressionoutput %>%
    filter(siteID == site_i)
  
  # If priors or LAI missing, warn and skip this hectare
  if (nrow(priors) == 0 || length(LAI_i) == 0) {
    warning("Missing prior or LAI for site: ", site_i)
    next
  }
  
  # Prior mean is hard-coded here (was originally using priors but set to 1.40)
  prior_mean <- 1.40
  prior_sd <- priors$prior_sd
  
  message("Processing: ", site_i, " plot ", i)
  
  # Filter data for this hectare and remove NA dbh values
  df_tile <- harvready %>%
    filter(IDhectbest == i) %>%
    filter(!is.na(dbh))
  
  # Skip hectares with too few observations (less than 25)
  if (nrow(df_tile) < 25) next
  
  # Pull numeric DBH vector for KDE and modeling steps
  df_tile <- df_tile %>% pull(dbh)
  
  # --------------------------------------------------------------------------
  # Step 1: KDE + candidate breakpoint detection
  #    - get_potential_breakpoint_and_kde() is a helper you must have defined
  #    - This should return a list with the KDE and candidate breakpoints
  # --------------------------------------------------------------------------
  kde_output <- tryCatch({
    get_potential_breakpoint_and_kde(df_tile)
  }, error = function(e) {
    message("KDE failed for ", i)
    return(NULL)
  })
  if (is.null(kde_output)) next
  
  # --------------------------------------------------------------------------
  # Step 2: Determine truncation and filter DBH values to the Pareto tail
  #    - determine_truncation_and_filter() should return elements:
  #         bayesian_data (data frame/vector for fitting),
  #         final_breakpoint (numeric)
  # --------------------------------------------------------------------------
  trunc_output <- tryCatch({
    determine_truncation_and_filter(kde_output)
  }, error = function(e) {
    message("Truncation failed for ", i)
    return(NULL)
  })
  if (is.null(trunc_output)) next
  
  # --------------------------------------------------------------------------
  # Step 3: Fit alpha model (Bayesian / Stan)
  #    - fit_alpha_model() is assumed to wrap Stan/brms sampling and return
  #      the fitted object with posterior summaries and stan_fit elements.
  # --------------------------------------------------------------------------
  fit <- tryCatch({
    fit_alpha_model(
      bayesian_data = trunc_output$bayesian_data,
      breakpoint = trunc_output$final_breakpoint,
      LAI = LAI_i,
      prior_mean = prior_mean,
      prior_sd = prior_sd
    )
  }, error = function(e) {
    message("Model failed for ", i)
    return(NULL)
  })
  if (is.null(fit)) next
  
  # --------------------------------------------------------------------------
  # Step 4: Estimate total trees (model that propagates α uncertainty)
  #    - estimate_total_trees() should take the fit and return posterior_summary,
  #      and stan_fit
  # --------------------------------------------------------------------------
  trees <- tryCatch({
    estimate_total_trees(
      fit,
      refresh = 900
    )
  }, error = function(e) {
    message("Model failed for ", i)
    return(NULL)
  })
  if (is.null(trees)) next
  
  # Save results (posterior_summary parts) into lists keyed by "SITE_HECT"
  alpha_results[[paste(site_i, i, sep = "_")]] <- fit$posterior_summary
  tree_results[[paste(site_i, i, sep = "_")]] <- trees$posterior_summary
  
  # Save raw draws for both α and tree totals to CSV files (folder must exist)
  # as_draws_df requires posterior or rstanfit object; we convert to data.frame
  as_draws_df(fit$stan_fit) %>% data.frame() %>%
    fwrite(paste0("alpha_draws_method/", i, "_alpha.csv"))
  
  as_draws_df(trees$stan_fit) %>% data.frame() %>%
    fwrite(paste0("alpha_draws_method/", i, "_treestot.csv"))
}

# ---------------------------
# 4. Combine per-hectare α and tree estimate summaries and save CSVs
# ---------------------------
alpha_summary_df <- bind_rows(alpha_results, .id = "IDhectbest")
tree_summary_df <- bind_rows(tree_results, .id = "IDhectbest")

# Write summary CSVs to working directory
fwrite(alpha_summary_df, "alpha_summary_df_harv2014.csv")
fwrite(tree_summary_df, "tree_summary_df_harv2014.csv")

# Also write the Weinstein-format alpha calculation file (keeps compatibility)
alpha_summary_df %>% fwrite("HARV_2014_weinstein_alphacalc.csv")

# ---------------------------
# 5. Load Harvard Forest field stems and plot coordinates (2014)
# ---------------------------
# Field stems and plot corner coordinates are read from downloads (preserve paths)
harv2014stems <- fread("C:/Users/adam.local/Downloads/hf253-05-stems-2014.csv")
plotcoords <- fread("C:/Users/adam.local/Downloads/hf253-03-coordinates.csv")

# Quick print of plotcoords to visually inspect in an interactive session
plotcoords

# ---------------------------
# 6. Convert tree relative coords to absolute UTM coordinates and save shapefile
# ---------------------------
# Determine SW corner UTM coordinates for each plot in plotcoords table
# This assumes 'corner' column has values like "SW" and easting/northing columns
sw_corner <- plotcoords %>%
  filter(corner == "SW") %>%
  select(easting, northing)

# Convert tree gx/gy (relative) to absolute UTM coordinates using SW corner
trees_sf <- harv2014stems %>%
  filter(!is.na(gx) & !is.na(gy)) %>%
  mutate(
    abs_easting = gx + sw_corner$easting,
    abs_northing = gy + sw_corner$northing
  ) %>%
  st_as_sf(coords = c("abs_easting", "abs_northing"), crs = 32618)  # UTM zone 18N

# Write tree points shapefile for later spatial joins
st_write(trees_sf, "harvard_2014_tree_points.shp", append = FALSE)

# ---------------------------
# 7. Read aggregated per-hectare tree survey shapefile (pre-processed)
# ---------------------------
# (This file should contain a per-hectare tree survey spatial join and dbh column)
setwd("C:/Users/adam.local/Documents/ForecastingScaling")
harv2014stemshectare <- read_sf("harv_2014_treesurvey_hectare.shp")

# ---------------------------
# 8. Define Stan model for Pareto α estimation (field stems)
# ---------------------------
# This Stan code models DBH (x) >= x_min with a Pareto(alpha) tail and logs normal prior
stan_model_code_alpha <- "
data {
  int<lower=0> N;               // Number of observations
  real<lower=0> x_min;          // Minimum DBH
  vector<lower=0>[N] x;         // DBH values
}
parameters {
  real<lower=0, upper=5> alpha; // Pareto density parameter
}
model {
  alpha ~ lognormal(1, 1);      // Prior for alpha (weakly informative)
  x ~ pareto(x_min, alpha);     // Likelihood for Pareto distribution
}
"

# Compile Stan model once for reuse
stan_model_alpha <- stan_model(model_code = stan_model_code_alpha)

# Function to fit Stan model and return mean + sd of posterior α
fit_alpha_for_plot <- function(plot_data) {
  # Prepare data for Stan
  stan_data <- list(
    N = nrow(plot_data),
    x_min = 10,            # lower DBH cutoff used in study
    x = plot_data$dbh      # DBH column inside plot_data
  )
  
  # Run sampling (note: high iterations/warmup in original; be aware of runtime)
  fit <- sampling(
    stan_model_alpha,
    data = stan_data,
    iter = 9000,
    warmup = 6000,
    chains = 4,
    refresh = 0
  )
  
  # Extract posterior mean and sd for alpha
  alpha_est <- summary(fit)$summary["alpha", "mean"]
  alpha_sd <- summary(fit)$summary["alpha", "sd"]
  return(list(alpha = alpha_est, alpha_sd = alpha_sd))
}

# Function that returns raw α draws (vector) for downstream summaries / draws
fit_alpha_draws_for_plot_draws <- function(plot_data) {
  stan_data <- list(
    N = nrow(plot_data),
    x_min = 10,
    x = plot_data$dbh
  )
  
  fit <- sampling(
    stan_model_alpha,
    data = stan_data,
    iter = 9000,
    warmup = 6000,
    chains = 4,
    refresh = 0
  )
  
  # rstan::extract returns a list; $alpha is the vector of draws
  alpha_draws <- rstan::extract(fit, pars = "alpha")$alpha
  return(alpha_draws)
}

# ---------------------------
# 9. Fit α to 2014 field stems (per-hectare)
# ---------------------------
alpha_results_2014 <- harv2014stemshectare %>%
  filter(dbh <= 50 & dbh >= 10) %>%   # restrict to DBH range 10-50 cm
  data.frame() %>%
  group_by(IDhectbest) %>%
  filter(n() > 25) %>%                # only plots with >25 trees
  nest() %>%
  mutate(
    fit_results = map(data, fit_alpha_for_plot),             # fit model to each plot
    alpha_2014 = map_dbl(fit_results, ~ .x$alpha),           # extract mean
    alpha_sd_2014 = map_dbl(fit_results, ~ .x$alpha_sd)      # extract sd
  ) %>%
  select(IDhectbest, alpha_2014, alpha_sd_2014)

# Also get long draws for 2014 (wide format later)
alpha_draws_long_2014 <- harv2014stemshectare %>%
  filter(dbh >= 10, dbh <= 50) %>%
  group_by(IDhectbest) %>%
  filter(n() > 25) %>%
  nest() %>%
  mutate(
    alpha_draws = map(data, fit_alpha_draws_for_plot_draws)  # returns vector of draws
  ) %>%
  select(IDhectbest, alpha_draws) %>%
  unnest(alpha_draws) %>%
  group_by(IDhectbest) %>%
  mutate(draw = row_number()) %>%
  ungroup() %>%
  mutate(year = "alpha_2014")   # tag year for later pivot

# ---------------------------
# 10. Read 2019 per-hectare field data and run the same fits
# ---------------------------
harv2019 <- read_sf("harv_2019_treesurvey_hectare.shp") %>%
  data.frame() %>%
  select(-geometry)

alpha_results_2019 <- harv2019 %>%
  filter(dbh <= 50 & dbh >= 10) %>%
  data.frame() %>%
  group_by(IDhectbest) %>%
  filter(n() > 25) %>%
  nest() %>%
  mutate(
    fit_results = map(data, fit_alpha_for_plot),
    alpha_2019 = map_dbl(fit_results, ~ .x$alpha),
    alpha_sd_2019 = map_dbl(fit_results, ~ .x$alpha_sd)
  ) %>%
  select(IDhectbest, alpha_2019, alpha_sd_2019)

alpha_draws_long_2019 <- harv2019 %>%
  filter(dbh >= 10, dbh <= 50) %>%
  group_by(IDhectbest) %>%
  filter(n() > 25) %>%
  nest() %>%
  mutate(alpha_draws = map(data, fit_alpha_draws_for_plot_draws)) %>%
  select(IDhectbest, alpha_draws) %>%
  unnest(alpha_draws) %>%
  group_by(IDhectbest) %>%
  mutate(draw = row_number()) %>%
  ungroup() %>%
  mutate(year = "alpha_2019")

# ---------------------------
# 11. Combine 2014 & 2019 draws and compute per-hectare α change per year
# ---------------------------
harvforestgeodraws <- alpha_draws_long_2014 %>%
  rbind(alpha_draws_long_2019) %>%
  pivot_wider(names_from = year, values_from = alpha_draws) %>%
  mutate(alpha_change = alpha_2019 - alpha_2014) %>%
  mutate(alpha_change = alpha_change / 5)  # divide by 5 years to get per-year change

# Save combined draws to CSV for reproducibility
fwrite(harvforestgeodraws, "harvforestgeodraws.csv")

# ---------------------------
# 12. Estimate tree counts per hectare per size class (Pole, Mature)
# ---------------------------
treesperhectare <- harv2019 %>%
  mutate(Year = 2019) %>%
  select(dbh, IDhectbest, Year) %>%
  rbind(
    harv2014stemshectare %>%
      mutate(Year = 2014) %>%
      select(dbh, IDhectbest, Year) %>%
      data.frame() %>%
      select(-geometry)
  ) %>%
  mutate(
    Type = ifelse(dbh >= 10 & dbh <= 25.4, "Pole",
                  ifelse(dbh >= 25.4 & dbh <= 50, "Mature", NA))
  ) %>%
  drop_na() %>%
  group_by(IDhectbest, Year, Type) %>%
  summarize(Trees_per_hectare = n(), .groups = "drop") %>%
  ungroup() %>%
  mutate(Trees_per_acre = Trees_per_hectare / 2.47105) %>%  # convert to per acre
  select(-Trees_per_hectare) %>%
  pivot_wider(names_from = Type, values_from = Trees_per_acre) %>%
  filter(Year == 2014)   # keep only 2014 baseline for later predictions

# ---------------------------
# 13. Prepare model input structure for RF-based simulation
# ---------------------------
# The script uses rf_model$trainingData (caret training object) as a template to
# construct predictors and dummy variables. We preserve that behavior.

# Get training data that was used to build the RF (assumed in rf_model object)
train_proc <- rf_model$trainingData

# Create dummy variables model using caret::dummyVars (replicates preprocessing)
dummy_model <- dummyVars(~ ., data = train_proc)

# Apply the dummy model to the training data to get a template for newdata
newdata_proc <- predict(dummy_model, train_proc) %>% as.data.frame()

# Output directory for simulated draws (create if doesn't exist)
output_dir <- "simulated_rf_draws_climateandtreatment_alldata/"
dir_create(output_dir)

# Columns we want set to 1 in categorical dummy columns when synthesizing a scenario
cols_to_set_1 <- c(
  "ECO_NAMELower New England / Northern Piedmont",
  "Disturbance_TypeNo Disturbance",
  "Forest_Type_GroupOak/pine"
)

# Quantile probabilities used for inverse-CDF sampling later (0.001 .. 0.999)
quantile_probs <- seq(0.001, 0.999, by = 0.01)

# Join α draws and tree densities (left join by IDhectbest)
draws_joined <- alpha_draws_long_2014 %>%
  left_join(treesperhectare, by = "IDhectbest")

# Save alias for dplyr::select as original code did
select <- dplyr::select

# ---------------------------
# 14. Loop over plots to simulate RF predictions and create draws
# ---------------------------
# For each plot (IDhectbest) we'll:
#  - build a replicated 'base_template' data.frame with n_draws rows
#  - fill in sampled α draws and other predictors (Pole_prev_tpa, etc.)
#  - use rf_model$finalModel to predict quantiles
#  - invert predicted quantile curves to draw random samples (inverse CDF via approxfun)
#  - save all simulated samples as an .rds file per plot

for (plot_id in unique(draws_joined$IDhectbest)) {
  
  # Subset draws and tree density info for this plot
  draws_plot <- draws_joined %>% filter(IDhectbest == plot_id)
  
  # Use 8000 draws per plot (as in original)
  n_draws <- 8000
  
  # Build base template by replicating the first row of training-processed data
  # - remove .outcome column from caret trainingData
  # - slice(rep(1, n_draws)) creates n_draws identical rows to mutate
  base_template <- newdata_proc[1, ] %>%
    select(-.outcome) %>%
    slice(rep(1, n_draws)) %>%
    mutate(
      # previous_mean is sampled from the per-plot alpha draws (sample with replacement)
      previous_mean = draws_plot$alpha_draws %>% sample(n_draws, replace = TRUE),
      
      # Pole_chng_perc: predicted using brms model posterior_epred for a baseline scenario
      # The brm_model_pole_perc object must be available in the environment.
      # We call posterior_epred with a single-row data.frame and coerce to vector.
      Pole_chng_perc = posterior_epred(brm_model_pole_perc,
                                       data.frame(
                                         Disturbance_Type = "No Disturbance",
                                         PREV_TPA = treesperhectare %>%
                                           filter(Year == 2014 & IDhectbest == plot_id) %>%
                                           mutate(TPA_Total = Pole + Mature) %>%
                                           pull(TPA_Total),
                                         previous_mean = draws_plot$alpha_draws %>% mean(),
                                         previous_sd = draws_plot$alpha_draws %>% sd(),
                                         mean_temp = 6.958333,
                                         total_prec = 1130,
                                         ECO_NAME = "Lower New England / Northern Piedmont"
                                       )) %>%
        as.vector(),
      
      # Mature_chng_perc currently hard-coded to 0 (placeholder/commented)
      Mature_chng_perc = 0,
      
      # Sample previous TPA values for the two size classes (sample with replacement)
      Mature_prev_tpa = draws_plot$Mature %>% sample(n_draws, replace = TRUE),
      Pole_prev_tpa = draws_plot$Pole %>% sample(n_draws, replace = TRUE)
    ) %>%
    # transform the Pole change predicted value using sinh (as in original)
    mutate(Pole_chng_perc = sinh(Pole_chng_perc)) %>%
    # Ensure categorical dummy columns are set according to `cols_to_set_1`
    mutate(across(
      matches("^(Disturbance_Type|ECO_NAME|Forest_Type_Group)"),
      ~ if_else(cur_column() %in% cols_to_set_1, 1, 0)
    ))
  
  # ---------------------------
  # Predict quantiles from the trained random forest model
  # - rf_model$finalModel is expected to be the fitted ranger (or caret wrapper)
  # - 'type = "quantiles"' returns quantile predictions per row consistent with ranger
  # ---------------------------
  pred_quants <- predict(
    rf_model$finalModel,
    data = base_template,
    type = "quantiles",
    quantiles = quantile_probs
  )$predict
  
  # ---------------------------
  # Inverse CDF sampling: for each draw i,
  #   - build an interpolator (approxfun) mapping quantile_probs -> predicted quantiles
  #   - evaluate interpolator at runif(1000) to get 1000 simulated samples per draw
  #   - store these in list; flatten later
  # ---------------------------
  simulated_draws_list <- vector("list", length = n_draws)
  
  # Loop through draws and invert the predicted quantile function
  for (i in seq_len(n_draws)) {
    # Create interpolator; rule = 2 extends the end values linearly for tails
    interpolator <- approxfun(quantile_probs, pred_quants[i, ], rule = 2)
    
    # Create 1000 random samples by inverting the predicted quantiles at uniform(0,1)
    simulated_draws_list[[i]] <- interpolator(runif(1000))
    
    # Progress message every 1000 iterations (and on last iteration)
    if (i %% 1000 == 0 || i == nrow(draws_joined %>% filter(IDhectbest == plot_id))) {
      message(paste0("  Processing scenario ", i, " of ", nrow(draws_joined %>% filter(IDhectbest == plot_id)), "..."))
    }
  }
  
  # Flatten nested list into a single numeric vector (n_draws * 1000 samples)
  all_simulated_samples <- unlist(simulated_draws_list)
  
  # Save simulated draws for this plot to RDS (designed to be reloaded for overlap tests)
  saveRDS(all_simulated_samples, file = file.path(output_dir, paste0("simulated_draws_", plot_id, ".rds")))
  
  message(sprintf("Saved %d x 1000 simulated samples for plot %s", n_draws, plot_id))
}

# ---------------------------
# 15. Compute overlap between observed α-change draws and simulated predictions
# ---------------------------
# Prepare objects to store overlap summary metrics per plot
plot_ids <- unique(harvforestgeodraws$IDhectbest)
sim_draws_path <- output_dir

# Initialize results dataframes
overlap_results <- data.frame(
  IDhectbest = character(),
  overlap_percent = numeric(),
  stringsAsFactors = FALSE
)

# Remove any zero ID if present
plot_ids <- plot_ids[plot_ids != 0]

# Extra stats store for diagnostics (sd/mean observed vs predicted)
overlap_stats <- data.frame(
  Observed_sd = numeric(),
  Observed_mean = numeric(),
  Predicted_sd = numeric(),
  Predicted_mean = numeric()
)

# Loop through each plot and compute kernel overlap between observed α-change draws
# and simulated RF-generated prediction draws
for (plot_id in plot_ids) {
  
  # Construct the filename holding the simulated RF samples for this plot
  sim_file <- file.path(sim_draws_path, paste0("simulated_draws_", plot_id, ".rds"))
  
  if (!file.exists(sim_file)) {
    message("Missing simulated file for plot: ", plot_id, " — skipping.")
    next
  }
  
  # Load predicted samples (vector of simulated α-change predictions)
  sim_samples <- readRDS(sim_file)
  
  # Extract the observed per-plot α-change draws from harvforestgeodraws
  real_draws <- harvforestgeodraws %>%
    filter(IDhectbest == plot_id) %>%
    pull(alpha_change)
  
  # Check we have enough samples on both sides
  if (length(real_draws) < 10 || length(sim_samples) < 10) {
    message("Too few samples for plot: ", plot_id, " — skipping.")
    next
  }
  
  # Compute the overlap between the two distributions using bayestestR::overlap
  # method_density = "kernel" uses KDE-based overlap; precision controls FFT resolution
  overlap_val <- bayestestR::overlap(
    x = real_draws,
    y = sim_samples,
    method_density = "kernel",
    precision = 2^12
  )
  
  # Append percent overlap to overlap_results (multiply by 100 for percent)
  overlap_results <- bind_rows(overlap_results, data.frame(
    IDhectbest = as.character(plot_id),
    overlap_percent = round(as.numeric(overlap_val) * 100, 2)
  ))
  
  # Append summary stats for diagnostics
  overlap_stats <- bind_rows(overlap_stats, data.frame(
    Observed_sd = sd(real_draws),
    Observed_mean = mean(real_draws),
    Predicted_sd = sd(sim_samples),
    Predicted_mean = mean(sim_samples)
  ))
  
  message("Done with plot: ", plot_id, " — Overlap: ", round(as.numeric(overlap_val) * 100, 2), "%")
}

# Quick summary in console (prints summary for overlap_results)
overlap_results %>% summary()

# Save overlap results for downstream analysis / plotting
write_csv(overlap_results, "overlap_results_by_plot_and_climate.csv")

# ---------------------------
# 16. Simple boxplot of overlap percentages for quick visualization
# ---------------------------
ggplot(overlap_results %>% mutate(Overlap = "Overlap"), aes(Overlap, overlap_percent)) +
  geom_boxplot() +
  labs(y = "Overlap percent", x = "") +
  theme_minimal()

################################################################################