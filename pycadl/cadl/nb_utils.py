"""Utility for displaying Tensorflow graphs.
"""
"""
From
https://github.com/tensorflow/tensorflow/blob/master/tensorflow
/examples/tutorials/deepdream/deepdream.ipynb

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
import tensorflow as tf
import numpy as np
from IPython.display import display, HTML


def show_graph(graph_def):
    """Summary

    Parameters
    ----------
    graph_def : TYPE
        Description
    """
    # Helper functions for TF Graph visualization
    def _strip_consts(graph_def, max_const_size=32):
        """Strip large constant values from graph_def.

        Parameters
        ----------
        graph_def : TYPE
            Description
        max_const_size : int, optional
            Description

        Returns
        -------
        TYPE
            Description
        """
        strip_def = tf.GraphDef()
        for n0 in graph_def.node:
            n = strip_def.node.add()
            n.MergeFrom(n0)
            if n.op == 'Const':
                tensor = n.attr['value'].tensor
                size = len(tensor.tensor_content)
                if size > max_const_size:
                    tensor.tensor_content = "<stripped {} bytes>".format(size).encode()
        return strip_def

    def _rename_nodes(graph_def, rename_func):
        """Summary

        Parameters
        ----------
        graph_def : TYPE
            Description
        rename_func : TYPE
            Description

        Returns
        -------
        TYPE
            Description
        """
        res_def = tf.GraphDef()
        for n0 in graph_def.node:
            n = res_def.node.add()
            n.MergeFrom(n0)
            n.name = rename_func(n.name)
            for i, s in enumerate(n.input):
                n.input[i] = rename_func(s) if s[0] != '^' else '^' + rename_func(s[1:])
        return res_def

    def _show_entire_graph(graph_def, max_const_size=32):
        """Visualize TensorFlow graph.

        Parameters
        ----------
        graph_def : TYPE
            Description
        max_const_size : int, optional
            Description
        """
        if hasattr(graph_def, 'as_graph_def'):
            graph_def = graph_def.as_graph_def()
        strip_def = _strip_consts(graph_def, max_const_size=max_const_size)
        code = """
            <script>
              function load() {{
                document.getElementById("{id}").pbtxt = {data};
              }}
            </script>
            <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
            <div style="height:600px">
              <tf-graph-basic id="{id}"></tf-graph-basic>
            </div>
        """.format(data=repr(str(strip_def)), id='graph' + str(np.random.rand()))

        iframe = """
            <iframe seamless style="width:800px;height:620px;border:0" srcdoc="{}"></iframe>
        """.format(code.replace('"', '&quot;'))
        display(HTML(iframe))
    # Visualizing the network graph. Be sure expand the "mixed" nodes to see their
    # internal structure. We are going to visualize "Conv2D" nodes.
    tmp_def = _rename_nodes(graph_def, lambda s: "/".join(s.split('_', 1)))
    _show_entire_graph(tmp_def)
