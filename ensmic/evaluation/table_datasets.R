# Import libraries
library("data.table")
library("tidyr")
install.packages("rjson")
library("rjson")

# Configurations
path_data <- "/home/mudomini/projects/ensmic/data/"
datasets <- c("chmnist", "covid", "isic", "drd")
sampling <- c("train-model", "val-model", "val-ensemble", "test")

# Create result dir
path_eval <- file.path("/home/mudomini/projects/ensmic", "results", "evaluation")
dir.create(path_eval, showWarnings=FALSE)

# Initialize result table
dt <- list()

# Iterate over datasets
for (ds in datasets){
  # Iterate over samplings 
  for (s in sampling){
    pf <- paste(ds, s, "json", sep=".", collapse=NULL)
    data <- fromJSON(file = file.path(path_data, pf))           # Load sampling json
    n_samples <- length(data)-1                                 # Identify number of samples (minus the legend)
    n_classes <- length(data$legend)                            # Identify number of classes

    dt <- rbind(dt, c(ds, s, as.numeric(n_samples), as.numeric(n_classes)))
  }
}

# Combine to datatable
dt <- as.data.table(dt)
setnames(dt, c("Dataset","sampling", "n_samples", "n_classes"))
dt <- dt[, n_samples:=as.numeric(n_samples)]
dt <- dt[, n_classes:=as.numeric(n_classes)]
# Spread table
dt <- spread(dt, key="sampling", value="n_samples")

# Add hard coded information - Modality
modalities <- c("Histology", "X-Ray", "Dermoscopy", "Ophthalmoscopy")
dt <- cbind(dt, "Modality"=modalities)
# Add hard coded information - Modality
difficulty <- c("Easy", "Easy", "Medium", "Hard")
dt <- cbind(dt, "Difficulty"=difficulty)

# Store to disk
path_out <- file.path(path_eval, "table.dataset.csv")
fwrite(dt, path_out)

