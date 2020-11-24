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
import pickle
import pandas
import numpy as np
from plotnine import *
# Internal libraries/scripts
from ensmic.data_loading import IO_Inference
from ensmic.architectures import architecture_dict, architectures
# Experimental
import warnings
warnings.filterwarnings("ignore")

#-----------------------------------------------------#
#                      Argparser                      #
#-----------------------------------------------------#
parser = argparse.ArgumentParser(description="Analysis of COVID-19 Classification via Ensemble Learning")
parser.add_argument("-m", "--modularity", help="Data modularity selection: ['x-ray', 'ct']",
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

# Adjust possible classes
if config["seed"] == "x-ray":
    config["class_dict"] = {'NORMAL': 0,
                            'Viral Pneumonia': 1,
                            'COVID-19': 2}
else:
    print("ERROR - Unknwon:", config["seed"])
    pass
config["class_list"] = sorted(config["class_dict"],
                              key=config["class_dict"].get)

# Architectures for Classification
config["architecture_list"] = architectures

#-----------------------------------------------------#
#          Function: Identify Classification          #
#-----------------------------------------------------#
def identify_class(pred, method="argmax"):
    if method == "argmax":
        return np.argmax(pred)

#-----------------------------------------------------#
#               Function: Preprocessing               #
#-----------------------------------------------------#
def preprocessing(architecture, dataset, config):
    # Load ground truth dictionary
    path_gt = os.path.join(config["path_data"], config["seed"] + \
                           ".classes.pickle")
    with open(path_gt, "rb") as pickle_reader:
        gt_map = pickle.load(pickle_reader)
    # Get result subdirectory for current architecture
    path_arch = os.path.join(config["path_results"], "phase_i" + "." + \
                             config["seed"], architecture)
    # Create an Inference IO Interface
    path_inf = os.path.join(path_arch, "inference" + "." + dataset + ".json")
    infIO = IO_Inference(config["class_dict"], path=path_inf)
    # Load predictions for samples
    inference = infIO.load_inference()

    # Initialize lists for predictions and ground truth
    id = []
    gt = []
    pd = []
    # Iterate over all samples of the testing set
    sample_list = inference.keys()
    for sample in sample_list:
        # Load prediction
        prediction = inference[sample]
        id.append(sample)
        gt.append(gt_map[sample])
        pd.append(identify_class(prediction))
    # Return parsed information
    return id, gt, pd

#-----------------------------------------------------#
#             Function: Metric Computation            #
#-----------------------------------------------------#
def compute_metrics(truth, pred, config):
    # Iterate over each class
    metrics = []
    for c in sorted(config["class_dict"].values()):
        mc = {}
        # Compute the confusion matrix
        tp, tn, fp, fn = compute_CM(truth, pred, c)
        mc["TP-TN-FP-FN"] = "-".join([str(tp), str(tn), str(fp), str(fn)])
        # Compute several metrics
        mc["Sensitivity"] = safe_division(tp, tp+fn)
        mc["Specificity"] = safe_division(tn, tn+fp)
        mc["Precision"] = safe_division(tp, tp+fp)
        mc["FPR"] = safe_division(fp, fp+tn)
        mc["FNR"] = safe_division(fn, fn+tp)
        mc["FDR"] = safe_division(fp, fp+tp)
        mc["Accuracy"] = safe_division(tp+tn, tp+tn+fp+fn)
        mc["F1"] = safe_division(2*tp, 2*tp+fp+fn)
        # Append dictionary to metric list
        metrics.append(mc)
    # Return results
    return metrics

# Compute confusion matrix
def compute_CM(gt, pd, c):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(0, len(gt)):
        #print(c, gt[i], pd[i], gt[i] == pd[i])
        if gt[i] == c and pd[i] == c : tp += 1
        elif gt[i] == c and pd[i] != c : fn += 1
        elif gt[i] != c and pd[i] != c : tn += 1
        elif gt[i] != c and pd[i] == c : fp += 1
        else : print("ERROR at confusion matrix", i)
    return tp, tn, fp, fn

# Function for safe division (catch division by zero)
def safe_division(x, y):
    return x / y if y else 0

#-----------------------------------------------------#
#          Function: Results Parsing & Backup         #
#-----------------------------------------------------#
def parse_results(metrics, architecture, config):
    # Parse metrics to Pandas dataframe
    results = pandas.DataFrame.from_dict(metrics)
    results = results.transpose()
    results.columns = config["class_list"]
    # Backup to disk
    path_arch = os.path.join(config["path_results"], "phase_i" + "." + \
                             config["seed"], architecture)
    path_res = os.path.join(path_arch, "metrics.csv")
    results.to_csv(path_res, index=True, index_label="metric")
    # Return dataframe
    return results

def collect_results(result_set, architectures, path_eval):
    # Initialize result dataframe
    cols = ["architecture", "class", "metric", "value"]
    df_results = pandas.DataFrame(data=[], dtype=np.float64, columns=cols)
    # Iterate over each architecture results
    for i in range(0, len(architectures)):
        arch_type = architectures[i]
        arch_df = result_set[i]
        arch_df.drop(index="TP-TN-FP-FN", inplace=True)
        arch_df = arch_df.astype(float)
        # Parse architecture result dataframe into desired shape
        arch_df = arch_df.reset_index()
        arch_df.rename(columns={"index":"metric"}, inplace=True)
        arch_df["architecture"] = arch_type
        arch_df = arch_df.melt(id_vars=["architecture", "metric"],
                               value_vars=["NORMAL", "Viral Pneumonia", "COVID-19"],
                               var_name="class",
                               value_name="value")
        # Reorder columns
        arch_df = arch_df[cols]
        # Merge to global result dataframe
        df_results = df_results.append(arch_df, ignore_index=True)
    # Backup merged results to disk
    path_res = os.path.join(path_eval, "all.results.csv")
    df_results.to_csv(path_res, index=False)
    # Return merged results
    return df_results

#-----------------------------------------------------#
#                Function: Plot Results               #
#-----------------------------------------------------#
def plot_results(results, eval_path):
    # Iterate over each metric
    for metric in np.unique(results["metric"]):
        # Extract sub dataframe for the current metric
        df = results.loc[results["metric"] == metric]
        # Sort classification
        df["class"] = pandas.Categorical(df["class"],
                                         categories=config["class_list"],
                                         ordered=True)
        # Plot results
        fig = (ggplot(df, aes("architecture", "value", fill="class"))
                      + geom_col(stat='identity', width=0.6, position = position_dodge(width=0.6))
                      + ggtitle("Architecture Comparison by " + metric)
                      + xlab("Architectures")
                      + ylab(metric)
                      + coord_flip()
                      + scale_y_continuous(limits=[0, 1])
                      + scale_fill_discrete(name="Classification")
                      + theme_bw(base_size=28))
        # Store figure to disk
        fig.save(filename="plot." + metric + ".png", path=path_eval,
                 width=18, height=14, dpi=300)

#-----------------------------------------------------#
#                    Run Evaluation                   #
#-----------------------------------------------------#
# Create evaluation subdirectory
path_eval = os.path.join(config["path_results"], "phase_i" + "." + \
                         config["seed"], "evaluation")
if not os.path.exists(path_eval) : os.mkdir(path_eval)

# Initialize result dataframe
result_set = []
verified_architectures = []

# Run Evaluation for all architectures
for ds in ["val-ensemble", "test"]:
    print("Run evaluation for dataset:", ds)
    for architecture in config["architecture_list"]:
        print("Run evaluation for Architecture:", architecture)
        try:
            # Preprocess ground truth and predictions
            id, gt, pd = preprocessing(architecture, ds, config)
            # Compute metrics
            metrics = compute_metrics(gt, pd, config)
            # Backup results
            metrics_df = parse_results(metrics, architecture, config)
            # Cache dataframe and add architecture to verification list
            result_set.append(metrics_df)
            verified_architectures.append(architecture)
        except:
            print("Skipping Architecture", architecture, "due to Error.")

# Combine results
results = collect_results(result_set, verified_architectures, path_eval)
# Plot result figure
plot_results(results, path_eval)
