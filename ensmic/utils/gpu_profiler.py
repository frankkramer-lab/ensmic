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
import tensorflow as tf

#-----------------------------------------------------#
#                     GPU Profiler                    #
#-----------------------------------------------------#
# Author: y.selivonchyk
# Source: https://stackoverflow.com/questions/57915007/how-to-measure-manually-how-much-of-my-gpu-memory-is-used-available
def get_max_memory_usage(sess):
    """Might be unprecise. Run after training"""
    if sess is None: sess = tf.get_default_session()
    max_mem = int(sess.run(tf.contrib.memory_stats.MaxBytesInUse()))
    print("Contrib max memory: %f G" % (max_mem / 1024. / 1024. / 1024.))
    return max_mem
