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
# Mean Approaches
from ensmic.ensemble.mean_unweighted import ELM_MeanUnweighted
from ensmic.ensemble.mean_weighted import ELM_MeanWeighted
# Majority Voting Approaches
from ensmic.ensemble.majorityvote_hard import ELM_MajorityVote_Hard
from ensmic.ensemble.majorityvote_soft import ELM_MajorityVote_Soft
# Machine Learning Approaches
from ensmic.ensemble.decision_tree import ELM_DecisionTree
from ensmic.ensemble.logistic_regression import ELM_LogisticRegression
from ensmic.ensemble.k_neighbors import ELM_kNearestNeighbors
from ensmic.ensemble.naive_bayes import ELM_NaiveBayes
from ensmic.ensemble.support_vector_machine import ELM_SupportVectorMachine
from ensmic.ensemble.gaussian_process import ELM_GaussianProcess
# Other Approaches
from ensmic.ensemble.global_argmax import ELM_GlobalArgmax
from ensmic.ensemble.best_model import ELM_BestModel

# Ensembler Dictionary
ensembler_dict = {"BestModel":ELM_BestModel,
                  "MeanUnweighted":ELM_MeanUnweighted,
                  "MeanWeighted":ELM_MeanWeighted,
                  "MajorityVoting_Hard":ELM_MajorityVote_Hard,
                  "MajorityVoting_Soft":ELM_MajorityVote_Soft,
                  "GlobalArgmax":ELM_GlobalArgmax,
                  "DecisionTree":ELM_DecisionTree,
                  "LogisticRegression":ELM_LogisticRegression,
                  "k-NearestNeighbors":ELM_kNearestNeighbors,
                  "NaiveBayes":ELM_NaiveBayes,
                  "SupportVectorMachine":ELM_SupportVectorMachine,
                  "GaussianProcess":ELM_GaussianProcess,
                  }
# List of implemented Ensemblers
ensembler = list(ensembler_dict.keys())
