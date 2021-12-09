# Import libraries
library("data.table")
library("ggplot2")
library("tidyr")
library("stringr")

# Configurations
path_eval <- "/home/mudomini/projects/ensmic/results/evaluation/"
datasets <- c("baseline", "augmenting", "bagging", "stacking")

# Gather data
dt <- data.table()
for (ds in datasets){
  path_file <- file.path(path_eval, paste("table", "results", ds, "csv", sep="."))
  dt_tmp <- fread(path_file)
  dt_tmp[, phase:=ds]
  dt <- rbind(dt, dt_tmp)
}

# Identify best methods & merge
dt$dataset <- sapply(dt$dataset, toupper)
dt$phase <- sapply(dt$phase, str_to_title)
dt_proc <- dt[, .(min=min(Accuracy), max=max(Accuracy)), by=list(phase, dataset)]

# Order labels
dt_proc$phase <- factor(dt_proc$phase, levels=c("Baseline", "Augmenting", "Bagging", "Stacking"))
dt$phase <- factor(dt$phase, levels=c("Baseline", "Augmenting", "Bagging", "Stacking"))
dt_proc$dataset <- factor(dt_proc$dataset, levels=c("CHMNIST", "COVID", "ISIC", "DRD"))
dt$dataset <- factor(dt$dataset, levels=c("CHMNIST", "COVID", "ISIC", "DRD"))

# Plot comparison
dodge <- position_dodge(width=0.8)
plot_comparison <- ggplot(dt_proc, aes(x=dataset, y=max, fill=phase)) +
  geom_bar(stat="identity", position=dodge, color="black", width=0.4, alpha=0.4) +
  stat_boxplot(data=dt, aes(x=dataset, y=F1), position=dodge, geom="errorbar") +
  geom_boxplot(data=dt, aes(x=dataset, y=F1), position=dodge, outlier.shape=NA) +
  geom_text(aes(y=0, label=phase), position=dodge, hjust=-0.2, vjust=0.4, angle=90) +
  scale_fill_brewer(palette="Set1") +
  scale_y_continuous(breaks=seq(0, 1, 0.05), limits=c(0, 1)) +
  theme_bw() +
  theme(legend.position = "none") +
  xlab("") +
  ylab("Score (Accuracy / F1)") +
  ggtitle("Comparison of Ensemble Learning Performance Influence on multiple datasets")
png(file.path(path_eval, "figure.comparison.png"), width=2000, height=800, res=180)
plot_comparison
dev.off()

# Compute performance gain for Accuracy
r_datasets <- dt_proc$dataset[1:4]
r_stacking <- ((dt_proc[phase=="Stacking"]$max / dt_proc[phase=="Baseline"]$max)-1) * 100
r_bagging <- ((dt_proc[phase=="Bagging"]$max / dt_proc[phase=="Baseline"]$max)-1) * 100
r_augmenting <- ((dt_proc[phase=="Augmenting"]$max / dt_proc[phase=="Baseline"]$max)-1) * 100

dt_gain <- data.table(dataset=r_datasets, augmenting=r_augmenting, bagging=r_bagging, stacking=r_stacking)
dt_gain <- melt(dt_gain, id.vars="dataset", measure.vars=c("augmenting", "bagging", "stacking"), 
                variable.name="phase", value.name="gain")

# Plot gain for Accuracy
plot_gain_acc <- ggplot(dt_gain, aes(x=dataset, y=gain, fill=phase)) +
  geom_bar(stat="identity", position="dodge", color="black", width=0.4, alpha=0.4) +
  scale_y_continuous(breaks=seq(-10, 10, 1), limits=c(-10, +10)) +
  scale_fill_manual(values=c("#377EB8", "#4DAF4A", "#984EA3")) + 
  coord_flip() +
  theme_bw() +
  theme(legend.position = "none") +
  xlab("") +
  ylab("Performance Gain in %") +
  ggtitle("Accuracy Gain compared to Baseline")

# Compute performance gain for F1
dt_proc <- dt[, .(min=min(F1), max=max(F1)), by=list(phase, dataset)]
dt_proc$phase <- factor(dt_proc$phase, levels=c("Baseline", "Augmenting", "Bagging", "Stacking"))
dt$phase <- factor(dt$phase, levels=c("Baseline", "Augmenting", "Bagging", "Stacking"))
dt_proc$dataset <- factor(dt_proc$dataset, levels=c("CHMNIST", "COVID", "ISIC", "DRD"))
dt$dataset <- factor(dt$dataset, levels=c("CHMNIST", "COVID", "ISIC", "DRD"))
r_datasets <- dt_proc$dataset[1:4]
r_stacking <- ((dt_proc[phase=="Stacking"]$max / dt_proc[phase=="Baseline"]$max)-1) * 100
r_bagging <- ((dt_proc[phase=="Bagging"]$max / dt_proc[phase=="Baseline"]$max)-1) * 100
r_augmenting <- ((dt_proc[phase=="Augmenting"]$max / dt_proc[phase=="Baseline"]$max)-1) * 100

dt_gain <- data.table(dataset=r_datasets, augmenting=r_augmenting, bagging=r_bagging, stacking=r_stacking)
dt_gain <- melt(dt_gain, id.vars="dataset", measure.vars=c("augmenting", "bagging", "stacking"), 
                variable.name="phase", value.name="gain")

# Plot gain for F1
plot_gain_f1 <- ggplot(dt_gain, aes(x=dataset, y=gain, fill=phase)) +
  geom_bar(stat="identity", position="dodge", color="black", width=0.4, alpha=0.4) +
  scale_y_continuous(breaks=seq(-10, 10, 1), limits=c(-10, +10)) +
  scale_fill_manual(values=c("#377EB8", "#4DAF4A", "#984EA3")) + 
  coord_flip() +
  theme_bw() +
  theme(legend.position = "none") +
  xlab("") +
  ylab("Performance Gain in %") +
  ggtitle("F1 Gain compared to Baseline")

###########################################################################################

multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)
  
  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)
  
  numPlots = length(plots)
  
  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  
  if (numPlots==1) {
    print(plots[[1]])
    
  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    
    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}

###########################################################################################

png(file.path(path_eval, "figure.comparison.big.png"), width=2000, height=800, res=170)
multiplot(plot_comparison, plot_gain_f1, layout=matrix(c(1,1,2), nrow=1, ncol=3, byrow=TRUE))
dev.off()


png(file.path(path_eval, "figure.comparison.final.png"), width=2200, height=1000, res=170)
multiplot(plot_comparison, plot_gain_f1, plot_gain_acc, layout=matrix(c(1,1,2,1,1,3), nrow=2, ncol=3, byrow=TRUE))
dev.off()









