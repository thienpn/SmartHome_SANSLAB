import graphsurgeon as gs
import tensorflow as tf

Input = gs.create_node("Input",
    op="Placeholder",
    dtype=tf.float32,
    shape=[1, 3, 300, 300])
PriorBox = gs.create_plugin_node(name="GridAnchor", op="GridAnchor_TRT",
    numLayers=6,
    minSize=0.2,
    maxSize=0.95,
    aspectRatios=[1.0, 2.0, 0.5, 3.0, 0.33],
    variance=[0.1,0.1,0.2,0.2],
    featureMapShapes=[19, 10, 5, 3, 2, 1])
NMS = gs.create_plugin_node(name="NMS", op="NMS_TRT",
    shareLocation=1,
    varianceEncodedInTarget=0,
    backgroundLabelId=0,
    confidenceThreshold=1e-8,
    nmsThreshold=0.6,
    topK=100,
    keepTopK=100,
    numClasses=91,
    ###########################################
    #inputOrder=[0, 2, 1],
    inputOrder=[1, 0, 2],
    ###########################################
    confSigmoid=1,
    isNormalized=1,
    scoreConverter="SIGMOID")
concat_priorbox = gs.create_node(name="concat_priorbox", op="ConcatV2", dtype=tf.float32, axis=2)
concat_box_loc = gs.create_plugin_node("concat_box_loc", op="FlattenConcat_TRT", dtype=tf.float32, axis=1, ignoreBatch=0)
concat_box_conf = gs.create_plugin_node("concat_box_conf", op="FlattenConcat_TRT", dtype=tf.float32, axis=1, ignoreBatch=0)

namespace_plugin_map = {
    "MultipleGridAnchorGenerator": PriorBox,
    "Postprocessor": NMS,
    "Preprocessor": Input,
    "ToFloat": Input,
    "image_tensor": Input,
#   "MultipleGridAnchorGenerator/Concatenate": concat_priorbox,
    "Concatenate": concat_priorbox,
    "concat": concat_box_loc,
    "concat_1": concat_box_conf
}

def preprocess(dynamic_graph):
    all_assert_nodes = dynamic_graph.find_nodes_by_op("Assert")
    dynamic_graph.remove(all_assert_nodes, remove_exclusive_dependencies=True)

    all_identity_nodes = dynamic_graph.find_nodes_by_op("Identity")
    dynamic_graph.forward_inputs(all_identity_nodes)

    dynamic_graph.collapse_namespaces(namespace_plugin_map)
    dynamic_graph.remove(dynamic_graph.graph_outputs, remove_exclusive_dependencies=False)
    dynamic_graph.find_nodes_by_op("NMS_TRT")[0].input.remove("Input")
