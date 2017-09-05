"""Metrics about TensorFlow graphs.
"""
"""
Copyright 2017 Parag K. Mital.  See also NOTICE.md.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import locale
import tensorflow as tf
from tensorflow.python.framework import ops


def print_stat(prefix, statistic_type, value):
    """Summary

    Parameters
    ----------
    prefix : TYPE
        Description
    statistic_type : TYPE
        Description
    value : TYPE
        Description
    """
    if value is None:
        friendly_value = "None"
    else:
        friendly_value = locale.format("%d", value, grouping=True)
    print("%s%s=%s" % (prefix, statistic_type, friendly_value))


def calculate_graph_metrics(graph_def, statistic_types, input_layer,
                            input_shape_override, batch_size):
    """Looks at the performance statistics of all nodes in the graph.

    Parameters
    ----------
    graph_def : TYPE
        Description
    statistic_types : TYPE
        Description
    input_layer : TYPE
        Description
    input_shape_override : TYPE
        Description
    batch_size : TYPE
        Description

    Returns
    -------
    TYPE
        Description

    Raises
    ------
    ValueError
        Description
    """
    tf.import_graph_def(graph_def, name="")
    total_stats = {}
    node_stats = {}
    for statistic_type in statistic_types:
        total_stats[statistic_type] = ops.OpStats(statistic_type)
        node_stats[statistic_type] = {}
    # Make sure we get pretty-printed numbers with separators.
    locale.setlocale(locale.LC_ALL, "")
    with tf.Session() as sess:
        input_tensor = sess.graph.get_tensor_by_name(input_layer)
        input_shape_tensor = input_tensor.get_shape()
        if input_shape_tensor:
            input_shape = input_shape_tensor.as_list()
        else:
            input_shape = None
        if input_shape_override:
            input_shape = input_shape_override
        if input_shape is None:
            raise ValueError("""No input shape was provided on the command line,"""
                             """ and the input op itself had no default shape, so"""
                             """ shape inference couldn't be performed. This is"""
                             """ required for metrics calculations.""")
        input_shape[0] = batch_size
        input_tensor.set_shape(input_shape)
        for node in graph_def.node:
            # Ensure that the updated input shape has been fully-propagated before we
            # ask for the statistics, since they may depend on the output size.
            op = sess.graph.get_operation_by_name(node.name)
            ops.set_shapes_for_outputs(op)
            for statistic_type in statistic_types:
                current_stats = ops.get_stats_for_node_def(sess.graph, node,
                                                           statistic_type)
                node_stats[statistic_type][node.name] = current_stats
                total_stats[statistic_type] += current_stats
    return total_stats, node_stats


def stats(graph_def, input_layer, batch_size):
    """Summary

    Parameters
    ----------
    graph_def : TYPE
        Description
    input_layer : TYPE
        Description
    batch_size : TYPE
        Description
    """
    statistic_types = ['flops']
    input_shape_override = False
    total_stats, node_stats = calculate_graph_metrics(
        graph_def, statistic_types, input_layer, input_shape_override,
        batch_size)
    for node in graph_def.node:
        for statistic_type in statistic_types:
            current_stats = node_stats[statistic_type][node.name]
            print_stat(node.name + "(" + node.op + "): ", statistic_type,
                       current_stats.value)
    for statistic_type in statistic_types:
        value = total_stats[statistic_type].value
        print_stat("Total: ", statistic_type, value)
