#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2021 IT-Infrastructure for Translational Medical Research,    #
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
import json
import pandas
import numpy as np
from plotnine import *
# Internal libraries/scripts
from ensmic.data_loading import IO_Inference, architecture_list, architecture_params
from ensmic.utils.metrics import compute_metrics, compute_rawCM
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

#-----------------------------------------------------#
#               Function: Preprocessing               #
#-----------------------------------------------------#
def preprocessing(architecture, dataset, config):
    # Load ground truth dictionary
    path_gt = os.path.join(config["path_data"], config["seed"] + \
                           "." + dataset + ".json")
    with open(path_gt, "r") as json_reader:
        gt_map = json.load(json_reader)
    # Get result subdirectory for current architecture
    path_arch = os.path.join(config["path_results"], "phase_i" + "." + \
                             config["seed"], architecture)
    # Create an Inference IO Interface
    path_inf = os.path.join(path_arch, "inference" + "." + dataset + ".json")
    infIO = IO_Inference(None, path=path_inf)
    # Load predictions for samples
    inference = infIO.load_inference()
    # Load class names
    config["class_list"] = infIO.load_inference(index="legend")

    # Initialize lists for predictions and ground truth
    id = []
    gt = []
    pd_class = []
    pd_prob = []
    # Iterate over all samples of the testing set
    sample_list = inference.keys()
    for sample in sample_list:
        # Obtain ground truth and predictions
        id.append(sample)
        gt.append(np.argmax(gt_map[sample]))
        prediction = inference[sample]
        pd_class.append(np.argmax(prediction))
        pd_prob.append(prediction)
    # Return parsed information
    return id, gt, pd_class, pd_prob

#-----------------------------------------------------#
#          Function: Results Parsing & Backup         #
#-----------------------------------------------------#
def parse_results(metrics, architecture, dataset, config):
    # Parse metrics to Pandas dataframe
    results = pandas.DataFrame.from_dict(metrics)
    results = results.transpose()
    results.columns = config["class_list"]
    # Backup to disk
    path_arch = os.path.join(config["path_results"], "phase_i" + "." + \
                             config["seed"], architecture)
    path_res = os.path.join(path_arch, "metrics." + dataset + ".csv")
    results.to_csv(path_res, index=True, index_label="metric")
    # Return dataframe
    return results

def collect_results(result_set, architectures, dataset, path_eval, config):
    # Initialize result dataframe
    cols = ["architecture", "class", "metric", "value"]
    df_results = pandas.DataFrame(data=[], dtype=np.float64, columns=cols)
    # Iterate over each architecture results
    for i in range(0, len(architectures)):
        arch_type = architectures[i]
        arch_df = result_set[i].copy()
        arch_df.drop(index="TP-TN-FP-FN", inplace=True)
        arch_df.drop(index="ROC_FPR", inplace=True)
        arch_df.drop(index="ROC_TPR", inplace=True)
        arch_df = arch_df.astype(float)
        # Parse architecture result dataframe into desired shape
        arch_df = arch_df.reset_index()
        arch_df.rename(columns={"index":"metric"}, inplace=True)
        arch_df["architecture"] = arch_type
        arch_df = arch_df.melt(id_vars=["architecture", "metric"],
                               value_vars=config["class_list"],
                               var_name="class",
                               value_name="value")
        # Reorder columns
        arch_df = arch_df[cols]
        # Merge to global result dataframe
        df_results = df_results.append(arch_df, ignore_index=True)
    # Backup merged results to disk
    path_res = os.path.join(path_eval, "results." + dataset + ".collection.csv")
    df_results.to_csv(path_res, index=False)
    # Return merged results
    return df_results

#-----------------------------------------------------#
#        Compute and Store raw Confusion Matrix       #
#-----------------------------------------------------#
def calc_confusion_matrix(gt, pd, architecture, dataset, config):
    # Compute confusion matrix
    rawcm_np = compute_rawCM(gt, pd, config["class_list"])
    rawcm = pandas.DataFrame(rawcm_np)
    # Tidy dataframe
    rawcm.index = config["class_list"]
    rawcm.columns = config["class_list"]
    # Backup to disk
    path_arch = os.path.join(config["path_results"], "phase_i" + "." + \
                             config["seed"], architecture)
    path_res = os.path.join(path_arch, "confusion_matrix." + dataset + ".csv")
    rawcm.to_csv(path_res, index=True, index_label="metric")
    # Return results
    return rawcm

def plot_confusion_matrix(rawcm, architecture, dataset, config):
    # Preprocess dataframe
    dt = rawcm.div(rawcm.sum(axis=0), axis=1) * 100
    dt = dt.round(decimals=2)
    dt.reset_index(drop=False, inplace=True)
    dt = dt.melt(id_vars=["index"], var_name="gt", value_name="score")
    dt.rename(columns={"index": "pd"}, inplace=True)
    # Plot confusion matrix
    fig = (ggplot(dt, aes("pd", "gt", fill="score"))
                  + geom_tile()
                  + geom_text(aes("pd", "gt", label="score"), color="black", size=28)
                  + ggtitle("Confusion Matrix: " + architecture)
                  + xlab("Prediction")
                  + ylab("Ground Truth")
                  + scale_fill_gradient(low="white", high="royalblue")
                  + theme_bw(base_size=28)
                  + theme(axis_text_x = element_text(angle = 45, vjust = 1, hjust = 1)))
    # Store figure to disk
    path_arch = os.path.join(config["path_results"], "phase_i" + "." + \
                             config["seed"], architecture)
    fig.save(filename="plot." + dataset + ".confusion_matrix." + "png",
             path=path_arch, width=18, height=14, dpi=200)

#-----------------------------------------------------#
#                Function: Plot Results               #
#-----------------------------------------------------#
def plot_categorical_results(results, dataset, eval_path):
    # Iterate over each metric
    for metric in np.unique(results["metric"]):
        # Extract sub dataframe for the current metric
        df = results.loc[results["metric"] == metric]
        # Sort classification
        df["class"] = pandas.Categorical(df["class"],
                                         categories=config["class_list"],
                                         ordered=True)
        # Plot results individually
        fig = (ggplot(df, aes("architecture", "value", fill="class"))
                      + geom_col(stat='identity', width=0.6,
                                 position = position_dodge(width=0.6))
                      + ggtitle("Architecture Comparison by " + metric)
                      + xlab("Architectures")
                      + ylab(metric)
                      + coord_flip()
                      + scale_y_continuous(limits=[0, 1])
                      + scale_fill_discrete(name="Classification")
                      + theme_bw(base_size=28))
        # Store figure to disk
        fig.save(filename="plot." + dataset + ".classwise." + metric + ".png",
                 path=path_eval, width=18, height=14, dpi=200)
    # Plot results together
    fig = (ggplot(results, aes("architecture", "value", fill="class"))
               + geom_col(stat='identity', width=0.6,
                          position = position_dodge(width=0.6))
               + ggtitle("Architecture Comparisons")
               + facet_wrap("metric", nrow=2)
               + xlab("Architectures")
               + ylab("Score")
               + coord_flip()
               + scale_y_continuous(limits=[0, 1])
               + scale_fill_discrete(name="Classification")
               + theme_bw(base_size=28))
    # Store figure to disk
    fig.save(filename="plot." + dataset + ".classwise.all.png",
          path=path_eval, width=40, height=25, dpi=200, limitsize=False)


def plot_averaged_results(results, dataset, eval_path):
    # Iterate over each metric
    for metric in np.unique(results["metric"]):
        # Extract sub dataframe for the current metric
        df = results.loc[results["metric"] == metric]
        # Plot results
        fig = (ggplot(df, aes("architecture", "value"))
                      + geom_col(stat='identity', width=0.6,
                                 position = position_dodge(width=0.6),
                                 show_legend=False,
                                 color='#F6F6F6', fill="#0C475B")
                      + ggtitle("Architecture Comparison by " + metric)
                      + xlab("Architectures")
                      + ylab(metric)
                      + coord_flip()
                      + scale_y_continuous(limits=[0, 1])
                      + theme_bw(base_size=28))
        # Store figure to disk
        fig.save(filename="plot." + dataset + ".averaged." + metric + ".png",
                 path=path_eval, width=18, height=14, dpi=200)
    # Plot results together
    fig = (ggplot(results, aes("architecture", "value"))
               + geom_col(stat='identity', width=0.6,
                          position = position_dodge(width=0.6),
                          show_legend=False,
                          color='#F6F6F6', fill="#0C475B")
               + ggtitle("Architecture Comparisons")
               + facet_wrap("metric", nrow=2)
               + xlab("Architectures")
               + ylab("Score")
               + coord_flip()
               + scale_y_continuous(limits=[0, 1])
               + theme_bw(base_size=28))
    # Store figure to disk
    fig.save(filename="plot." + dataset + ".averaged.all.png",
          path=path_eval, width=40, height=25, dpi=200, limitsize=False)

#-----------------------------------------------------#
#                     ROC Analysis                    #
#-----------------------------------------------------#
def preprocess_roc_data(results, valid_architectures):
    # Initialize result dataframe
    cols = ["architecture", "class", "FPR", "TPR"]
    df_results = pandas.DataFrame(data=[], dtype=np.float64, columns=cols)
    # Iterate over each architecture results
    for i in range(0, len(valid_architectures)):
        # Preprocess data into correct format
        arch_df = result_set[i].copy()
        arch_df = arch_df.transpose()
        roc_df = arch_df[["ROC_FPR", "ROC_TPR"]]
        roc_df = roc_df.apply(pandas.Series.explode)
        roc_df["architecture"] = valid_architectures[i]
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
        df_results = df_results.append(roc_df, ignore_index=True)
    return df_results

def plot_auroc_results(results, dataset, eval_path):
    # Plot roc results via facet_wrap
    fig = (ggplot(results, aes("FPR", "TPR", color="class"))
               + geom_line(size=1.5)
               + geom_abline(intercept=0, slope=1, color="black",
                             linetype="dashed")
               + ggtitle("Architecture Comparisons by ROC")
               + facet_wrap("architecture", nrow=4)
               + xlab("False Positive Rate")
               + ylab("True Positive Rate")
               + scale_x_continuous(limits=[0, 1])
               + scale_y_continuous(limits=[0, 1])
               + scale_color_discrete(name="Classification")
               + theme_bw(base_size=28))
    # Store figure to disk
    fig.save(filename="plot." + dataset + ".ROC.individual.png",
             path=path_eval, width=40, height=20, dpi=200, limitsize=False)

    try:
        # Plot roc results together
        fig = (ggplot(results, aes("FPR", "TPR", color="architecture"))
                + geom_smooth(method="loess", se=False, size=1.5)
                + geom_abline(intercept=0, slope=1, color="black",
                              linetype="dashed", size=1.5)
                + ggtitle("Architecture Comparisons by ROC")
                + xlab("False Positive Rate")
                + ylab("True Positive Rate")
                + scale_x_continuous(limits=[0, 1])
                + scale_y_continuous(limits=[0, 1])
                + scale_color_discrete(name="Architectures")
                + theme_bw(base_size=40))
        # Store figure to disk
        fig.save(filename="plot." + dataset + ".ROC.together.png",
              path=path_eval, width=30, height=20, dpi=200, limitsize=False)
    except:
        print("Skipped ROC-together figure for:", dataset)


#-----------------------------------------------------#
#               Model Parameter Analysis              #
#-----------------------------------------------------#
def preprocess_modelparas(results, path_data):
    # Parse parameter dictionary to Pandas dataframe
    params = pandas.DataFrame.from_dict(architecture_params, orient="index",
                                        columns=["params"])
    params.reset_index(drop=False, inplace=True)
    params.rename(mapper={"index":"architecture"}, axis=1, inplace=True)
    # Merge dataframes
    df_results = results.merge(params, how="inner", on="architecture")
    # Specify on metric: Accuracy
    df_results = df_results[df_results["metric"] == "Accuracy"]
    # Return result dataframe
    return df_results

def plot_modelparas_results(results, dataset, eval_path):
    # Plot roc results via facet_wrap
    fig = (ggplot(results, aes("params", "value", color="architecture",
                                label="architecture"))
               + geom_point(size=10)
               + geom_text(aes(label="architecture"), size=30, nudge_y=-0.002)
               + ggtitle("Architecture Complexity compared to Performance")
               + xlab("Number of Parameters")
               + ylab("Accuracy")
               + scale_color_discrete(name="Architectures")
               + theme_bw(base_size=28))
    # Store figure to disk
    fig.save(filename="plot." + dataset + ".modelparams.png",
             path=path_eval, width=30, height=20, dpi=200, limitsize=False)

#-----------------------------------------------------#
#                Fitting Curve Analysis               #
#-----------------------------------------------------#
def gather_fitting_data(config):
    dt_list = []
    # Gather fitting logs from each architecture
    for architecture in architecture_list:
        try:
            # Get path to fitting logging for current architecture
            path_arch = os.path.join(config["path_results"], "phase_i" + "." + \
                                     config["seed"], architecture)
            path_trainlogs = os.path.join(path_arch, "logs.csv")
            # Load CSV as dataframe
            dt_trainlog = pandas.read_csv(path_trainlogs)
            # Add current architecture to dataframe and add to datatable list
            dt_trainlog["architecture"] = architecture
            dt_list.append(dt_trainlog)
        except:
            print("Skipping architecture", architecture, "for fitting evaluation")
    # Merge to global fitting dataframe
    dt_fitting = pandas.concat(dt_list, ignore_index=True)
    # Melt dataframe into correct format
    dt_fitting_loss = dt_fitting.melt(id_vars=["architecture", "epoch"],
                                      value_vars=["loss", "val_loss"],
                                      var_name="Dataset",
                                      value_name="score")
    dt_fitting_accuracy = dt_fitting.melt(id_vars=["architecture", "epoch"],
                                          value_vars=["categorical_accuracy",
                                                    "val_categorical_accuracy"],
                                          var_name="Dataset",
                                          value_name="score")
    # Return datatable
    return dt_fitting_loss, dt_fitting_accuracy

def plot_fitting(results, metric, eval_path, config):
    if metric == "Loss Function":
        limits = [0, 2]
    else:
        limits = [0, 1]
    # Plot results
    fig = (ggplot(results, aes("epoch", "score", color="factor(Dataset)"))
               #+ geom_smooth(method="gpr", size=1)
               + geom_line(size=1)
               + ggtitle(config["seed"].upper() + \
                         ": Fitting Curve during Training")
               + facet_wrap("architecture", nrow=4, scales="free_x")
               + xlab("Epoch")
               + ylab(metric)
               + scale_y_continuous(limits=limits)
               + scale_colour_discrete(name="Dataset",
                                       labels=["Training", "Validation"])
               + theme_bw(base_size=28))
    # Store figure to disk
    fig.save(filename="plot.fitting_course." + metric + ".png",
          path=path_eval, width=65, height=25, dpi=200, limitsize=False)

#-----------------------------------------------------#
#                    Run Evaluation                   #
#-----------------------------------------------------#
# Create evaluation subdirectory
path_eval = os.path.join(config["path_results"], "phase_i" + "." + \
                         config["seed"], "evaluation")
if not os.path.exists(path_eval) : os.mkdir(path_eval)

# Run Evaluation for validation and test dataset
for ds in ["val-ensemble", "test"]:
    # Initialize result dataframe
    result_set = []
    verified_architectures = []

    # Run Evaluation for all architectures
    print("Run evaluation for dataset:", ds)
    for architecture in architecture_list:
        print("Run evaluation for Architecture:", architecture)
        try:
            # Preprocess ground truth and predictions
            id, gt, pd, pd_prob = preprocessing(architecture, ds, config)
            # Compute metrics
            metrics = compute_metrics(gt, pd, pd_prob, config["class_list"])
            # Backup results
            metrics_df = parse_results(metrics, architecture, ds, config)
            # Cache dataframe and add architecture to verification list
            result_set.append(metrics_df)
            verified_architectures.append(architecture)
            # Compute and store raw Confusion Matrix
            rawcm = calc_confusion_matrix(gt, pd, architecture, ds, config)
            plot_confusion_matrix(rawcm, architecture, ds, config)
        except:
            print("Skipping Architecture", architecture, "due to Error.")

    # Collect results
    results_all = collect_results(result_set, verified_architectures, ds,
                                  path_eval, config)
    # Macro Average results
    results_averaged = macro_averaging(results_all, ds, path_eval)

    # Plot result figure
    plot_categorical_results(results_all, ds, path_eval)
    plot_averaged_results(results_averaged, ds, path_eval)

    # Analyse ROC
    results_roc = preprocess_roc_data(result_set, verified_architectures)
    plot_auroc_results(results_roc, ds, path_eval)

    # Analyse Model Parameters vs Performance
    results_modelparas = preprocess_modelparas(results_averaged,
                                               config["path_results"])
    plot_modelparas_results(results_modelparas, ds, path_eval)

# Analyse fitting curve loggings
dt_fitting_loss, dt_fitting_accuracy = gather_fitting_data(config)
plot_fitting(dt_fitting_loss, "Loss_Function", path_eval, config)
plot_fitting(dt_fitting_accuracy, "Categorical_Accuracy", path_eval, config)
