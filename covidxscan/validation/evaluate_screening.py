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
import os
import pickle
import pandas
import numpy as np
from plotnine import *
# MIScnn libraries
from miscnn.data_loading.data_io import create_directories
from miscnn.evaluation.cross_validation import load_disk2fold
# Internal libraries/scripts
from covidxscan.data_loading import Inference_IO

#-----------------------------------------------------#
#                    Configurations                   #
#-----------------------------------------------------#
# File structure
path_target = "data"
path_val = "validation.screening"
# Seed (if training multiple runs)
seed = 42
# Adjust possible classes
class_dict = {'NORMAL': 0,
              'Viral Pneumonia': 1,
              'COVID-19': 2}
class_list = sorted(class_dict, key=class_dict.get)
# Architectures for Neural Network
architectures = ["VGG16", "InceptionResNetV2", "Xception", "DenseNet", "ResNeSt"]

#-----------------------------------------------------#
#               Function: Preprocessing               #
#-----------------------------------------------------#
def preprocessing(architecture, path_data, path_val, path_eval):
    # Create evaluation subdirectory for the current architecture
    path_arch = create_directories(path_eval, architecture)
    # Load ground truth dictionary
    path_gt = os.path.join(path_data, str(seed) + ".classes.pickle")
    with open(path_gt, "rb") as pickle_reader:
        gt_map = pickle.load(pickle_reader)
    # Load testing samples
    path_testing = os.path.join(path_val, "testing", "sample_list.json")
    _, testing = load_disk2fold(path_testing)
    # Initialize Inference IO for prediction loading
    path_pd = os.path.join(path_val, "inference", architecture)
    infIO = Inference_IO(class_dict, outdir=path_pd)
    # Initialize lists for predictions and ground truth
    id = []
    gt = []
    pd = []
    # Iterate over all samples of the testing set
    for sample in testing:
        #DEBUG######################################################################################################
        if not os.path.exists(os.path.join(path_pd, sample, ".json")): continue
        # DEBUG######################################################################################################
        # Load prediction
        inf_json = infIO.load_inference(sample)
        inf = class_dict[inf_json["cds_class"]]
        # Store informations in list
        id.append(sample)
        gt.append(gt_map[sample])
        pd.append(inf)
    # Return parsed information
    return id, gt, pd, path_arch

#-----------------------------------------------------#
#             Function: Metric Computation            #
#-----------------------------------------------------#
def compute_metrics(truth, pred):
    # Iterate over each class
    metrics = []
    for c in sorted(class_dict.values()):
        mc = {}
        # Compute the confusion matrix
        tp, tn, fp, fn = compute_CM(truth, pred, c)
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
def parse_results(path_arch, metrics):
    # Parse metrics to Pandas dataframe
    results = pandas.DataFrame.from_dict(metrics)
    results = results.transpose()
    results.columns = class_list
    # Backup to disk
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
    path_res = os.path.join(path_eval, "results.csv")
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
        # Plot results
        fig = (ggplot(df, aes("architecture", "value", fill="class"))
                      + geom_col(stat='identity', position='dodge', size=2)
                      + ggtitle("Architecture Comparison by " + metric)
                      + xlab("Architectures")
                      + ylab(metric)
                      + scale_y_continuous(limits=[0, 1])
                      + scale_fill_discrete(name="Classification",
                                            labels=class_list) # ------------------------------------------- Is this correct??!?!?
                      + theme_bw(base_size=28))
        # Store figure to disk
        fig.save(filename="plot." + metric + ".png", path=path_eval,
                 width=18, height=10, dpi=500)

#-----------------------------------------------------#
#                    Run Evaluation                   #
#-----------------------------------------------------#
# Create evaluation subdirectory
path_eval = create_directories(path_val, "evaluation")
# Iterate over each architecture
result_set = []
for architecture in architectures:
    id, gt, pd, path_arch = preprocessing(architecture, path_target,
                                          path_val, path_eval)
    # Compute metrics
    metrics = compute_metrics(gt, pd)
    # Backup results
    metrics_df = parse_results(path_arch, metrics)
    # Cache dataframe
    result_set.append(metrics_df)
# Combine results
results = collect_results(result_set, architectures, path_eval)
# Plot result figure
plot_results(results, path_eval)
