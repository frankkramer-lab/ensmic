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

# Load data
dt <- fread(file.path(path_results, "eval_tmp", "confusion_matrix.csv"))


# Preprocess
dt$dataset <- paste("Dataset:", dt$dataset)
dt <- transform(dt, dataset=factor(dataset, levels=c("Dataset: CHMNIST","Dataset: COVID",
                                                     "Dataset: ISIC","Dataset: DRD")))
dt$score <- round(dt$score, 1)
dt <- transform(dt, phase=factor(phase, levels=c("Baseline", "Augmenting", "Bagging", "Stacking")))




# Make template plot
dt_cut <- split(dt, f=dt$dataset)
p1 <- ggplot(dt_cut[[1]], aes(pd, gt, fill=score)) +
  geom_tile() +
  geom_text(aes(pd, gt, label=score), color="black", size=3) +  # size=24
  facet_wrap(. ~ phase) +
  xlab("Prediction") +
  ylab("Ground Truth") +
  scale_fill_gradient(low="white", high="royalblue", limits=c(0, 100)) +
  ggtitle(unique(dt$dataset)[1]) +
  theme_bw() + # size=24
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) +
  theme(legend.position="none")

# Plot remaining ones
p2 <- p1 %+% dt_cut[[2]] %+% ggtitle(unique(dt$dataset)[2])
p3 <- p1 %+% dt_cut[[3]] %+% ggtitle(unique(dt$dataset)[3])
p4 <- p1 %+% dt_cut[[4]] %+% ggtitle(unique(dt$dataset)[4])
# Convert & adjust widths+heights
g1 = ggplotGrob(p1)
g2 = ggplotGrob(p2)
g3 = ggplotGrob(p3)
g4 = ggplotGrob(p4)
g1$widths <- g4$widths
g2$widths <- g4$widths
g3$widths <- g4$widths
g1$heights <- g4$heights
g2$heights <- g4$heights
g3$heights <- g4$heights

title <- textGrob("Confusion Matrices of the best Methods for each Dataset & Phase\n", gp=gpar(fontsize=20,font=1))
png(file.path(path_eval, paste("figure.cm.big.png", sep="")), width=3000, height=1000, res=120)
grid.arrange(g1,g2,g3,g4, nrow=1, ncol=4, top=title)
dev.off()
