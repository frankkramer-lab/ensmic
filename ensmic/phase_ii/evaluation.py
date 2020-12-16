#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2020 IT-Infrastructure for Translational Medical Research,    #
#                University of Augsburg                                        #
#                                                                              #
#  This program is free software: you can redistribute it and/or modify        #
#  it under the terms of the GNU General Public License as published by        #
#  the Free Software Foundation, either version 3 of the License, or           #
#  (at your option) any later version.                                         #
#                                                                              #
#  This program is distributed in the hope that it will be useful,             #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
#  GNU General Public License for more details.                                #
#                                                                              #
#  You should have received a copy of the GNU General Public License           #
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.       #
#==============================================================================#
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
import argparse
import os
import pandas
import numpy as np
import json
from plotnine import *
# Internal libraries/scripts
from ensmic.data_loading import IO_Inference
from ensmic.ensemble import ensembler
from ensmic.utils.metrics import compute_metrics
from ensmic.utils.categorical_averaging import macro_averaging
# Experimental
import warnings
warnings.filterwarnings("ignore")

#-----------------------------------------------------#
#                      Argparser                      #
#-----------------------------------------------------#
parser = argparse.ArgumentParser(description="Analysis of COVID-19 Classification via Ensemble Learning")
parser.add_argument("-m", "--modularity", help="Data modularity selection: ['covid', 'isic', 'chmnist', 'drd']",
                    required=True, type=str, dest="seed")
args = parser.parse_args()

#-----------------------------------------------------#
#                    Configurations                   #
#-----------------------------------------------------#
# Initialize configuration dictionary
config = {}
# Path to data directory
config["path_data"] = "data"
# Path to result directory
config["path_results"] = "results"
# Seed (if training multiple runs)
config["seed"] = args.seed

# Load possible classes
path_classdict = os.path.join(config["path_data"],
                              config["seed"] + ".classes.json")
with open(path_classdict, "r") as json_reader:
    config["class_dict"] = json.load(json_reader)
config["class_list"] = sorted(config["class_dict"],
                              key=config["class_dict"].get)

# Ensemble Learning Techniques
config["ensembler_list"] = ensembler

#-----------------------------------------------------#
#          Function: Identify Classification          #
#-----------------------------------------------------#
def identify_class(pred, method="argmax"):
    if method == "argmax":
        return np.argmax(pred)

#-----------------------------------------------------#
#               Function: Preprocessing               #
#-----------------------------------------------------#
def preprocessing(ensembler, config):
    # Get path to phase for correct dataset
    path_phase = os.path.join(config["path_results"], "phase_ii" + "." + \
                              config["seed"])
    # Load ground truth for testing set
    path_gt = os.path.join(path_phase, "phase_i.inference.test.class.csv")
    gt_map = pandas.read_csv(path_gt, header=0, index_col="index").to_dict()
    gt_map = gt_map["Ground_Truth"]

    # Create Inference IO Interface
    path_inf = os.path.join(path_phase, ensembler, "inference.test.json")
    infIO = IO_Inference(config["class_dict"], path=path_inf)
    # Load predictions for samples
    inference = infIO.load_inference()

    # Initialize lists for predictions and ground truth
    id = []
    gt = []
    pd_class = []
    pd_prob = []
    # Iterate over all samples of the testing set
    sample_list = inference.keys()
    for sample in sample_list:
        # Cache ground truth and prediction
        id.append(sample)
        gt.append(gt_map[sample])
        prediction = inference[sample]
        pd_class.append(identify_class(prediction))
        pd_prob.append(prediction)
    # Return parsed information
    return id, gt, pd_class, pd_prob

#-----------------------------------------------------#
#          Function: Results Parsing & Backup         #
#-----------------------------------------------------#
def parse_results(metrics, ensembler, config):
    # Parse metrics to Pandas dataframe
    results = pandas.DataFrame.from_dict(metrics)
    results = results.transpose()
    results.columns = config["class_list"]
    # Backup to disk
    path_ensembler = os.path.join(config["path_results"], "phase_ii" + "." + \
                                  config["seed"], ensembler)
    path_res = os.path.join(path_ensembler, "metrics.csv")
    results.to_csv(path_res, index=True, index_label="metric")
    # Return dataframe
    return results

def collect_results(result_set, verified_ensembler, path_eval, config):
    # Initialize result dataframe
    cols = ["ensembler", "class", "metric", "value"]
    df_results = pandas.DataFrame(data=[], dtype=np.float64, columns=cols)
    # Iterate over each ensembler results
    for i in range(0, len(verified_ensembler)):
        ens_type = verified_ensembler[i]
        ens_df = result_set[i].copy()
        ens_df.drop(index="TP-TN-FP-FN", inplace=True)
        ens_df.drop(index="ROC_FPR", inplace=True)
        ens_df.drop(index="ROC_TPR", inplace=True)
        ens_df = ens_df.astype(float)
        # Parse ensembler result dataframe into desired shape
        ens_df = ens_df.reset_index()
        ens_df.rename(columns={"index":"metric"}, inplace=True)
        ens_df["ensembler"] = ens_type
        ens_df = ens_df.melt(id_vars=["ensembler", "metric"],
                             value_vars=config["class_list"],
                             var_name="class",
                             value_name="value")
        # Reorder columns
        ens_df = ens_df[cols]
        # Merge to global result dataframe
        df_results = df_results.append(ens_df, ignore_index=True)
    # Backup merged results to disk
    path_res = os.path.join(path_eval, "all.results.csv")
    df_results.to_csv(path_res, index=False)
    # Return merged results
    return df_results

#-----------------------------------------------------#
#                    Run Evaluation                   #
#-----------------------------------------------------#
# Create evaluation subdirectory
path_eval = os.path.join(config["path_results"], "phase_ii" + "." + \
                         config["seed"], "evaluation")
if not os.path.exists(path_eval) : os.mkdir(path_eval)

# Initialize result dataframe
result_set = []
verified_ensembler = []

# Run Evaluation for all ensemble learning techniques
for ensembler in config["ensembler_list"]:
    print("Run evaluation for Ensembler:", ensembler)
    try:
        # Preprocess ground truth and predictions
        id, gt, pd, pd_prob = preprocessing(ensembler, config)
        # Compute metrics
        metrics = compute_metrics(gt, pd, pd_prob, config)
        # Backup results
        metrics_df = parse_results(metrics, ensembler, config)
        # Cache dataframe and add ensembler to verification list
        result_set.append(metrics_df)
        verified_ensembler.append(ensembler)
    except Exception as e:
        print("Skipping Ensembler", ensembler, "due to Error:", e)

# Combine results
results_all = collect_results(result_set, verified_ensembler, path_eval, config)

#-----------------------------------------------------#
#         Function: Plot Performance Classwise        #
#-----------------------------------------------------#
# Iterate over each metric
for metric in np.unique(results_all["metric"]):
    # Extract sub dataframe for the current metric
    df = results_all.loc[results_all["metric"] == metric]
    # Sort classification
    df["class"] = pandas.Categorical(df["class"],
                                     categories=config["class_list"],
                                     ordered=True)
    # Plot results individually
    fig = (ggplot(df, aes("ensembler", "value", fill="class"))
                  + geom_col(stat='identity', width=0.6, position = position_dodge(width=0.6))
                  + ggtitle("Ensemble Learning Comparison by " + metric)
                  + xlab("Ensemble Learning Technique")
                  + ylab(metric)
                  + coord_flip()
                  + scale_y_continuous(limits=[0, 1])
                  + scale_fill_discrete(name="Classification")
                  + theme_bw(base_size=28))
    # Store figure to disk
    fig.save(filename="plot.classwise." + metric + ".png", path=path_eval,
             width=18, height=14, dpi=200)
# Plot results together
fig = (ggplot(results_all, aes("ensembler", "value", fill="class"))
           + geom_col(stat='identity', width=0.6,
                      position = position_dodge(width=0.6))
           + ggtitle("Ensemble Learning Comparisons")
           + facet_wrap("metric", nrow=2)
           + xlab("Ensemble Learning Technique")
           + ylab("Score")
           + coord_flip()
           + scale_y_continuous(limits=[0, 1])
           + scale_fill_discrete(name="Classification")
           + theme_bw(base_size=28))
# Store figure to disk
fig.save(filename="plot.classwise.all.png",
      path=path_eval, width=40, height=25, dpi=200, limitsize=False)

#-----------------------------------------------------#
#         Function: Plot Performance Averaged         #
#-----------------------------------------------------#
# Macro Average results
results_averaged = macro_averaging(results_all, "test", path_eval, "ensembler")

# Iterate over each metric
for metric in np.unique(results_averaged["metric"]):
    # Extract sub dataframe for the current metric
    df = results_averaged.loc[results_averaged["metric"] == metric]
    # Plot results individually
    fig = (ggplot(df, aes("ensembler", "value"))
                  + geom_col(stat='identity', width=0.6,
                             position = position_dodge(width=0.6),
                             show_legend=False,
                             color='#F6F6F6', fill="#0C475B")
                  + ggtitle("Ensemble Learning Comparison by " + metric)
                  + xlab("Ensemble Learning Technique")
                  + ylab(metric)
                  + coord_flip()
                  + scale_y_continuous(limits=[0, 1])
                  + theme_bw(base_size=28))
    # Store figure to disk
    fig.save(filename="plot.averaged." + metric + ".png", path=path_eval,
             width=18, height=14, dpi=200)
# Plot results together
fig = (ggplot(results_averaged, aes("ensembler", "value"))
           + geom_col(stat='identity', width=0.6,
                      position = position_dodge(width=0.6),
                      show_legend=False,
                      color='#F6F6F6', fill="#0C475B")
           + ggtitle("Ensemble Learning Comparisons")
           + facet_wrap("metric", nrow=2)
           + xlab("Ensemble Learning Technique")
           + ylab("Score")
           + coord_flip()
           + scale_y_continuous(limits=[0, 1])
           + theme_bw(base_size=28))
# Store figure to disk
fig.save(filename="plot.averaged.all.png",
      path=path_eval, width=40, height=25, dpi=200, limitsize=False)

#-----------------------------------------------------#
#                     ROC Analysis                    #
#-----------------------------------------------------#
# Initialize result dataframe
cols = ["ensembler", "class", "FPR", "TPR"]
results_roc = pandas.DataFrame(data=[], dtype=np.float64, columns=cols)
# Iterate over each ensembler results
for i in range(0, len(verified_ensembler)):
    # Preprocess data into correct format
    ens_df = result_set[i].copy()
    ens_df = ens_df.transpose()
    roc_df = ens_df[["ROC_FPR", "ROC_TPR"]]
    roc_df = roc_df.apply(pandas.Series.explode)
    roc_df["ensembler"] = verified_ensembler[i]
    # Append to result dataframe
    roc_df = roc_df.reset_index()
    roc_df.rename(columns={"index":"class",
                           "ROC_FPR":"FPR",
                           "ROC_TPR":"TPR"},
                  inplace=True)
    # Reorder columns
    roc_df = roc_df[cols]
    # Convert from object to float
    roc_df["FPR"] = roc_df["FPR"].astype(float)
    roc_df["TPR"] = roc_df["TPR"].astype(float)
    # Merge to global result dataframe
    results_roc = results_roc.append(roc_df, ignore_index=True)

# Plot roc results via facet_wrap
fig = (ggplot(results_roc, aes("FPR", "TPR", color="class"))
           + geom_line(size=1.5)
           + geom_abline(intercept=0, slope=1, color="black",
                         linetype="dashed")
           + ggtitle("Ensemble Learning Comparisons by ROC")
           + facet_wrap("ensembler", nrow=3)
           + xlab("False Positive Rate")
           + ylab("True Positive Rate")
           + scale_x_continuous(limits=[0, 1])
           + scale_y_continuous(limits=[0, 1])
           + scale_color_discrete(name="Classification")
           + theme_bw(base_size=28))
# Store figure to disk
fig.save(filename="plot.ROC.individual.png",
         path=path_eval, width=40, height=20, dpi=200, limitsize=False)

# Plot roc results together
try:
    fig = (ggplot(results_roc, aes("FPR", "TPR", color="ensembler"))
            + geom_smooth(method="loess", se=False, size=1.5, span=0.7)
            + geom_abline(intercept=0, slope=1, color="black",
                          linetype="dashed", size=1.5)
            + ggtitle("Ensemble Learning Comparisons by ROC")
            + xlab("False Positive Rate")
            + ylab("True Positive Rate")
            + scale_x_continuous(limits=[0, 1])
            + scale_y_continuous(limits=[0, 1])
            + scale_color_discrete(name="Ensemble LearningTechnique")
            + theme_bw(base_size=40))
    # Store figure to disk
    fig.save(filename="plot.ROC.together.png",
          path=path_eval, width=30, height=20, dpi=200, limitsize=False)
except:
    print("ROC-Together plot was not be able to created!")
