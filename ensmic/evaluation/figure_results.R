# Import libraries
library("data.table")
library("ggplot2")
library("tidyr")
library("grid")
library("gridExtra")
library("stringr")

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
  # Preprocess
  dt_score$dataset <- toupper(dt_score$dataset)
  dt_score$dataset <- paste("Dataset:", dt_score$dataset)
  dt_score <- transform(dt_score, dataset=factor(dataset, levels=c("Dataset: CHMNIST","Dataset: COVID",
                                                                   "Dataset: ISIC","Dataset: DRD")))

  # Plot figure A
  plot_a <- ggplot(dt_score[metric=="F1"], aes(method, value, fill=dataset)) +
    geom_bar(stat="identity", position="dodge", col="black", width=0.7) +
    scale_y_continuous(breaks=seq(0, 1, 0.1), limits=c(0, 1)) +
    facet_wrap(. ~ dataset) + 
    coord_flip() +
    theme_bw() +
    theme(legend.position = "none") +
    xlab("") +
    ylab("Metric - F1")
  # png(file.path(path_eval, paste("figure.results.", phases[i], ".A.png", sep="")), width=1200, height=1000, res=180)
  # print(plot_a)
  # dev.off()
  
  # Load roc data
  path_file <- file.path(path_results, "eval_tmp", paste(phases[i], ".", "roc.csv", sep="", collapse=NULL))
  dt_roc <- fread(path_file)
  # Preprocess
  dt_roc$dataset <- toupper(dt_roc$dataset)
  dt_roc$dataset <- paste("Dataset: ", dt_roc$dataset, " - Best Method: ", dt_roc$method, sep="")
  dt_roc <- transform(dt_roc, dataset=factor(dataset, levels=unique(dt_roc$dataset)))
  
  # Plot figure B  
  dt_roc_split <- split(dt_roc, f=dt_roc$dataset)
  p1 <- ggplot(dt_roc_split[[1]], aes(FPR, TPR, color=class)) + 
    geom_line() +
    geom_abline(intercept=0, slope=1, color="black",
                linetype="dashed") +
    facet_wrap(~ dataset, nrow=1, ncol=1) +
    scale_x_continuous(breaks=seq(0, 1, 0.1), limits=c(0, 1)) +
    scale_y_continuous(breaks=seq(0, 1, 0.1), limits=c(0, 1)) +
    scale_color_discrete(name="Classes") +
    theme_bw() +
    xlab("False Positive Rate") +
    ylab("True Positive Rate")
  
  p2 <- p1 %+% dt_roc_split[[2]]
  p3 <- p1 %+% dt_roc_split[[3]]
  p4 <- p1 %+% dt_roc_split[[4]]
  
  g1 = ggplotGrob(p1)
  g2 = ggplotGrob(p2)
  g3 = ggplotGrob(p3)
  g4 = ggplotGrob(p4)
  g1$widths <- g4$widths
  g2$widths <- g4$widths
  g3$widths <- g4$widths
  
  # png(file.path(path_eval, paste("figure.results.", phases[i], ".B.png", sep="")), width=1250, height=800, res=120)
  # plot_b <- arrangeGrob(g1,g2,g3,g4)
  # grid.arrange(plot_b)
  # dev.off()
  
  # Plot figure A and B together
  title <- paste("Results for Ensemble Learning Technique: ", str_to_title(phases[i]), sep="")
  png(file.path(path_eval, paste("figure.results.", phases[i], ".big.png", sep="")), width=2000, height=800, res=110)
  layout <- rbind(c(1,1,1,2,2,3,3),
                  c(1,1,1,4,4,5,5))
  grid.arrange(plot_a,g1,g2,g3,g4, layout_matrix=layout, top=textGrob(title, gp=gpar(fontsize=20,font=1)))
  dev.off()
}


