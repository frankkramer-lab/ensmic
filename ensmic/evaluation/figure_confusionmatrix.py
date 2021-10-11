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
import os
import pandas as pd
import json
from ast import literal_eval
import numpy as np
from plotnine import *
# Internal libraries/scripts
from ensmic.data_loading import IO_Inference
from ensmic.utils.metrics import compute_rawCM

#-----------------------------------------------------#
#                    Configurations                   #
#-----------------------------------------------------#
path_data = "/home/mudomini/projects/ensmic/data"
path_results = "/home/mudomini/projects/ensmic/results"
phases = ["baseline", "augmenting", "stacking", "bagging"]
datasets = ["chmnist", "covid", "isic", "drd"]

#-----------------------------------------------------#
#                     Gather Data                     #
#-----------------------------------------------------#
cols = ["pd", "gt", "score", "dataset", "phase"]
df_results = pd.DataFrame(data=[], columns=cols)
# Iterate over phases
for phase in phases:
    # Iterate over each dataset
    for ds in datasets:
        # Load ground truth dictionary
        path_gt = os.path.join(path_data, ds + ".test.json")
        with open(path_gt, "r") as json_reader:
            gt_map = json.load(json_reader)

        # Get current path
        path_current = os.path.join(path_results, "phase_" + phase + "." + ds)
        # Identify best method
        if phase != "bagging":
            data = pd.read_csv(os.path.join(path_current, "evaluation",
                                            "results.test.averaged.csv"))
            if phase == "stacking":
                data.rename(columns={"ensembler": "method"}, inplace=True)
            else : data.rename(columns={"architecture": "method"}, inplace=True)
            data_f1 = data[data["metric"] == "F1"]
            best_dt = data_f1.iloc[data_f1["value"].argmax()]
            best_method = best_dt["method"]
        # For bagging: Identify best architecture & best method
        else:
            # iterate over all architectures
            best_score = 0
            for walk in os.listdir(path_current):
                if os.path.isfile(os.path.join(path_current, walk)) : continue
                # Load results
                data = pd.read_csv(os.path.join(path_current, walk, "evaluation",
                                                "results.test.averaged.csv"))
                # Obtain best
                data.rename(columns={"ensembler": "method"}, inplace=True)
                data_f1 = data[data["metric"] == "F1"]
                curr_dt = data_f1.iloc[data_f1["value"].argmax()]
                score = curr_dt["value"]
                if best_score > score : continue
                else:
                    best_score = score
                    best_architecture = walk
                    best_method = curr_dt["method"]
            # update path_current with best architecture
            path_current = os.path.join(path_current, best_architecture)

        # Create an Inference IO Interface
        if phase != "bagging" : path_inf = os.path.join(path_current,
                                            best_method, "inference.test.json")
        else : path_inf = os.path.join(path_current, "inference",
                            "inference." + best_method + ".pred.json")
        infIO = IO_Inference(None, path=path_inf)
        # Load predictions for samples
        inference = infIO.load_inference()
        # Load class names
        class_list = infIO.load_inference(index="legend")

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

        #----------------------------#
        #         CM Analysis        #
        #----------------------------#
        # Compute confusion matrix
        rawcm_np = compute_rawCM(gt, pd_class, class_list)
        rawcm = pd.DataFrame(rawcm_np)
        # Tidy dataframe
        rawcm.index = class_list
        rawcm.columns = class_list
        # Preprocess dataframe
        dt = rawcm.div(rawcm.sum(axis=0), axis=1) * 100
        dt = dt.round(decimals=2)
        dt.reset_index(drop=False, inplace=True)
        dt = dt.melt(id_vars=["index"], var_name="gt", value_name="score")
        dt.rename(columns={"index": "pd"}, inplace=True)
        # Add meta information
        dt["dataset"] = ds
        dt["phase"] = phase
        # Append to final dataframe
        df_results = df_results.append(dt, ignore_index=True)

df_results["dataset"] = df_results["dataset"].str.upper()
df_results["phase"] = df_results["phase"].str.capitalize()

# Store data
path_res = os.path.join(path_results, "eval_tmp", "confusion_matrix.csv")
df_results.to_csv(path_res, index=False)

# # Plot figure
# print(df_results)
# fig = (ggplot(df_results[df_results["dataset"]=="chmnist"], aes("pd", "gt", fill="score"))
#               + geom_tile()
#               + geom_text(aes("pd", "gt", label="score"), color="black", size=24)
#               + facet_wrap("~ phase")
#               + ggtitle("Dataset: " + "CHMNIST")
#               + xlab("Prediction")
#               + ylab("Ground Truth")
#               + scale_fill_gradient(low="white", high="royalblue", limits=[0, 100])
#               + theme_bw(base_size=24)
#               + theme(legend_position="none")
#               + theme(axis_text_x = element_text(angle = 45, vjust = 1, hjust = 1)))
# # # Store figure to disk
# # path_arch = os.path.join(config["path_results"], "phase_baseline" + "." + \
# #                          config["seed"], architecture)
# fig.save(filename="test.png",
#          path=path_results, width=18, height=14, dpi=200)
