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

# Gather results for baseline, augmenting & stacing
for (i in seq(1,3)){
  dt <- data.table()
  
  for (j in seq(1,4)){
    path_dir <- file.path(path_results, paste("phase_", phases[i], ".", datasets[j], sep = "", collapse = NULL))
    path_file <- file.path(path_dir, "evaluation", "results.test.averaged.csv")
    dt_tmp <- fread(path_file)
    dt_tmp[, dataset:=datasets[j]]
    if("ensembler" %in% colnames(dt_tmp))
    {
      setnames(dt_tmp, "ensembler", "model")
    }
    if("architecture" %in% colnames(dt_tmp))
    {
      setnames(dt_tmp, "architecture", "model")
    }
    dt_tmp <- dt_tmp[metric %in% c("Accuracy", "F1", "Sensitivity", "Specificity", "ROC_AUC")]
    dt <- rbind(dt, dt_tmp)
  }
  dt <- dcast(dt, dataset + model ~ metric)
  dt <- dt %>% mutate_if(is.numeric, round, 2)
  path_out <- file.path(path_eval, paste("table.results.", phases[i], ".csv", sep=""))
  fwrite(dt, path_out)
}

# Gather results for bagging
dt <- data.table()
for (j in seq(1,4)){
  path_dir <- file.path(path_results, paste("phase_", phases[4], ".", datasets[j], sep = "", collapse = NULL))
  
  # Identify highest performing architecture
  max_value <- 0
  max_arch <- ""
  for (path_arch in list.dirs(path_dir, recursive=FALSE)){
    dt_tmp <- fread(file.path(path_arch, "evaluation", "results.test.averaged.csv"))
    if(max(dt_tmp[metric=="F1"]$value) > max_value){
      max_value <- max(dt_tmp[metric=="F1"]$value)
      max_arch <- path_arch
    }
  }
  
  # Obtain results
  dt_tmp <- fread(file.path(path_arch, "evaluation", "results.test.averaged.csv"))
  dt_tmp[, dataset:=datasets[j]]
  dt <- rbind(dt, dt_tmp)
  
  print(max_arch)
  print(max_value)
}
dt <- dt[metric %in% c("Accuracy", "F1", "Sensitivity", "Specificity", "ROC_AUC")]
setnames(dt, "ensembler", "model")
dt <- dcast(dt, dataset + model ~ metric)
dt <- dt %>% mutate_if(is.numeric, round, 2)
path_out <- file.path(path_eval, paste("table.results.", phases[4], ".csv", sep=""))
fwrite(dt, path_out)
