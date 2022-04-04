# Import libraries
library("data.table")
library("ggplot2")
library("tidyr")
library("dplyr")

# Configurations
path_results <- "/home/mudomini/projects/ensmic/results/"
datasets <- c("chmnist", "covid", "isic", "drd")
phases <- c("baseline", "augmenting", "stacking", "bagging")

path_eval <- file.path(path_results, "evaluation")
dir.create(path_eval, showWarnings=FALSE)

# Iterate over all phases
for (i in seq(1,4)){
  # Load best score data
  path_file <- file.path(path_results, "eval_tmp", paste(phases[i], ".", "all.csv", sep="", collapse=NULL))
  dt_score <- fread(path_file)
  # Dcast dataframes
  dt_dcast <- dcast(dt_score, dataset + method ~ metric)
  # Select only important metrics
  dt_final <- dt_dcast[, c("dataset", "method", "Accuracy", "F1", "Sensitivity", "ROC_AUC")] 
  # Round metrics
  dt_final <- dt_final %>% mutate_if(is.numeric, round, 2)
  # Write results to disk
  path_out <- file.path(path_eval, paste("table.results.", phases[i], ".csv", sep=""))
  fwrite(dt_final, path_out)
}