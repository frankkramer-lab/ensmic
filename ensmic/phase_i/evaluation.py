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
import json
import pandas
import numpy as np
from plotnine import *
# Internal libraries/scripts
from ensmic.data_loading import IO_Inference
from ensmic.architectures import architecture_dict, architectures
from ensmic.utils.evaluation import compute_metrics
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

# Load possible classes
path_classdict = os.path.join(config["path_data"],
                              str(self.seed) + ".classes.json")
with open(path_classdict, "r") as json_reader:
    config["class_dict"] = = json.load(json_reader)
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
                           ".class_map.pickle")
    with open(path_gt, "r") as json_reader:
        gt_map = json.load(json_reader)
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

def collect_results(result_set, architectures, path_eval, config):
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
                               value_vars=config["class_list"],
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
results = collect_results(result_set, verified_architectures, path_eval, config)
# Plot result figure
plot_results(results, path_eval)
