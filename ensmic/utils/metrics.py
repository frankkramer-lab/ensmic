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
from sklearn.metrics import roc_curve, auc
import numpy as np
# Internal libraries/scripts

#-----------------------------------------------------#
#             Function: Metric Computation            #
#-----------------------------------------------------#
def compute_metrics(truth, pred, pd_prob, config):
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
        # Compute AUROC
        truth_class = (np.array(truth) == c).astype(int)
        pdprob_class = np.asarray(pd_prob)[:, c]
        mc["ROC_FPR"], mc["ROC_TPR"], _ = roc_curve(truth_class, pdprob_class)
        mc["ROC_AUC"] = auc(mc["ROC_FPR"], mc["ROC_TPR"])
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
        if gt[i] == c and pd[i] == c : tp += 1
        elif gt[i] == c and pd[i] != c : fn += 1
        elif gt[i] != c and pd[i] != c : tn += 1
        elif gt[i] != c and pd[i] == c : fp += 1
        else : print("ERROR at confusion matrix", i)
    return tp, tn, fp, fn

# Compute raw confusion matrix
def compute_rawCM(gt, pd, config):
    class_list = sorted(config["class_dict"].values())
    rawcm = np.zeros((len(class_list), len(class_list)))
    for i in range(0, len(gt)):
        rawcm[gt[i]][pd[i]] += 1
    return rawcm

# Function for safe division (catch division by zero)
def safe_division(x, y):
    return x / y if y else 0
