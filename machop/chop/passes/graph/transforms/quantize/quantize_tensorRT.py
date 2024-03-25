from copy import copy, deepcopy
import logging
import torch
from chop.passes.graph.interface.save_and_load import load_mase_graph_interface_pass
import onnx
import tensorrt as trt
from cuda import cudart
import time
import numpy as np

from ...utils import (
    deepcopy_mase_graph,
    get_mase_op,
    get_mase_type,
    get_node_actual_target,
    get_parent_name,
    get_similar_node_actual_target,
    match_a_pattern,
    get_node_target_by_name,
)

from .modify import create_new_fn, create_new_module
from .modify_tensorRT import create_new_module_tensorRT,create_new_module_tensorRT_real
from .quant_parsers import parse_node_config, relink_node_meta, update_quant_meta_param
from .summary import graph_iterator_compare_nodes, graph_iterator_node_histogram

logger = logging.getLogger(__name__)


###########################################################
# Read me!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
#
# The file have Quantization pass and calibration pass
# suitable path is: mase/machop/chop/passes/graph/transforms/quantize/quantize_tensorRT.py
############################################################

QUANTIZEABLE_OP = (
    "add",
    "bmm",
    "conv1d",
    "conv2d",
    "matmul",
    "mul",
    "linear",
    "relu",
    "sub",
    "max_pool2d"
)


def get_config(config: dict, name: str):
    if name in config:
        return config[name]["config"]
    else:
        return config["default"]["config"]

def graph_iterator_quantize_by_type_tensorRT_type(graph, config: dict):
    # Some modules might need information from two graphs to be initilized
    if (
        config.get("baseline_weight_path") is not None
        and config.get("load_type") == "mz"
    ):
        bl_graph = deepcopy_mase_graph(graph)
        bl_graph = load_mase_graph_interface_pass(
            bl_graph, pass_args=config.get("baseline_weight_path")
        )
    else:
        bl_graph = None
    for node in graph.fx_graph.nodes:
        if get_mase_op(node) not in QUANTIZEABLE_OP:
            continue
        node_config = get_config(config, get_mase_op(node))
        if node_config["name"] is None:
            continue
        node_config = parse_node_config(node_config, get_mase_op(node))
        # if get_mase_type(node) == "module":
        if node.op == "call_module":
            ori_module = get_node_actual_target(node)
            successor_module = get_similar_node_actual_target(
                bl_graph, node.next
            )  # Certain modules will require information about their successor module to complete the initialization process. (For LogicNets, activation functions are needed.)
            bl_module = get_similar_node_actual_target(bl_graph, node)
            if config[get_mase_op(node)]['fake'] == "True":
                new_module = create_new_module_tensorRT(
                    get_mase_op(node),
                    ori_module,
                    node_config,
                    node.meta,
                    bl_module,
                    successor_module,
                )
            else:
                new_module = create_new_module_tensorRT_real(
                    get_mase_op(node),
                    ori_module,
                    node_config,
                    node.meta,
                    bl_module,
                    successor_module,
                )
            parent_name, name = get_parent_name(node.target)
            setattr(graph.modules[parent_name], name, new_module)
            # update precision and type in meta.parameters["common"]
            update_quant_meta_param(node, node_config, get_mase_op(node))
        elif get_mase_type(node) in [
            "builtin_func",
            "module_related_func",
        ]:
            new_f, args, kwargs = create_new_fn(node, node_config)
            with graph.fx_graph.inserting_before(node):
                new_node = graph.fx_graph.call_function(new_f, args, kwargs)
                new_node.name = node.name
                new_node.meta["mase"] = copy(node.meta["mase"])
                # new_node.meta["mase"].node -> new_node
                relink_node_meta(new_node, model=graph.model)
                update_quant_meta_param(new_node, node_config, get_mase_op(node))
                node.replace_all_uses_with(new_node)
            graph.fx_graph.erase_node(node)
    return graph


def graph_iterator_quantize_by_type_tensorRT_name(graph, config: dict):
    # Some modules might need information from two graphs to be initilized
    if (
        config.get("baseline_weight_path") is not None
        and config.get("load_type") == "mz"
    ):
        bl_graph = deepcopy_mase_graph(graph)
        bl_graph = load_mase_graph_interface_pass(
            bl_graph, pass_args=config.get("baseline_weight_path")
        )
    else:
        bl_graph = None
    for node in graph.fx_graph.nodes:
        if get_mase_op(node) not in QUANTIZEABLE_OP:
            continue
        node_config = get_config(config, node.name)
        if node_config["name"] is None:
            continue
        # node_config = parse_node_config(node_config, node.name)
        if node.op == "call_module":
            ori_module = get_node_actual_target(node)
            successor_module = get_similar_node_actual_target(
                bl_graph, node.next
            )  # Certain modules will require information about their successor module to complete the initialization process. (For LogicNets, activation functions are needed.)
            bl_module = get_similar_node_actual_target(bl_graph, node)
            if config[node.name]['fake'] == "True":
                new_module = create_new_module_tensorRT(
                    get_mase_op(node),
                    ori_module,
                    node_config,
                    node.meta,
                    bl_module,
                    successor_module,
                )
            else:
                new_module = create_new_module_tensorRT_real(
                    get_mase_op(node),
                    ori_module,
                    node_config,
                    node.meta,
                    bl_module,
                    successor_module,
                )

            parent_name, name = get_parent_name(node.target)
            setattr(graph.modules[parent_name], name, new_module)
            # update precision and type in meta.parameters["common"]
            update_quant_meta_param(node, node_config, get_mase_op(node))
        elif get_mase_type(node) in [
            "builtin_func",
            "module_related_func",
        ]:
            new_f, args, kwargs = create_new_fn(node, node_config)
            with graph.fx_graph.inserting_before(node):
                new_node = graph.fx_graph.call_function(new_f, args, kwargs)
                new_node.name = node.name
                new_node.meta["mase"] = copy(node.meta["mase"])
                # new_node.meta["mase"].node -> new_node
                relink_node_meta(new_node, model=graph.model)
                update_quant_meta_param(new_node, node_config, get_mase_op(node))
                node.replace_all_uses_with(new_node)
            graph.fx_graph.erase_node(node)
    return graph

def tensorRT_quantize_pass(graph, pass_args=None):
    """_summary_

    Args:
        graph (Mase Graph): Input Mase Graph
        pass_args (dic, optional): _description_. Defaults to None.
        fake (bool, optional): _description_. Defaults to False.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    print("hello world")
    by = pass_args["by"]
    match by:
        case "type":
            graph = graph_iterator_quantize_by_type_tensorRT_type(graph, pass_args)
        case "name":
            graph = graph_iterator_quantize_by_type_tensorRT_name(graph, pass_args)
        case _:
            raise ValueError(f'Unsupported quantize "by": {by}')

    # link the model with graph
    graph.model = torch.fx.GraphModule(graph.model, graph.fx_graph)
    return graph, {}

def calibration_pass(graph, pass_args=None,data_module=None,batch_size=8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph.model.to(device)
    for name in graph.modules.keys():
        if name.endswith('_quantizer'):
            graph.modules[name].disable_quant()  # Use full precision data to calibrate
            graph.modules[name].enable_calib()
                
    count = 0

    if count <= 1:
        for inputs in data_module.train_dataloader():
            xs, ys = inputs
            xs, ys = xs.to(device), ys.to(device)
            if xs.shape[0] != batch_size:
                continue
            graph.model(xs)
            count += 1

    for name in graph.modules.keys():
        if name.endswith('_quantizer'):
            print(f"Loading calibration data for {name}")
            graph.modules[name].load_calib_amax()
            graph.modules[name].disable_calib()
            graph.modules[name].enable_quant()
            print(f"Max absolute value for {name}: {graph.modules[name].amax}")
    
    graph.model.to(device)

    return graph, {}

def export_to_onnx_pass(mg, dummy_in, input_generator, onnx_model_path):
    dummy_in = next(iter(input_generator))['x']
    dummy_in = dummy_in.cuda()
    testdevice = torch.device('cpu')
    torch.onnx.export(mg.model.to(testdevice),  dummy_in.to(testdevice), onnx_model_path, export_params=True, opset_version=13, do_constant_folding=True, \
                        input_names = ['input'], output_names = ['output'], dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}})
    mg.meta["onnx_model_path"] = onnx_model_path
    return  mg, {}

def generate_tensorrt_string_pass(mg, TR_output_path):
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    onnx_model_path = mg.meta["onnx_model_path"]

    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_model_path, 'rb') as model:
        print("parser.parse(model.read()): ",str(parser.parse(model.read())))
        for error in range(parser.num_errors):
            print(parser.get_error(error))

    inputTensor = network.get_input(0)
    profile.set_shape(inputTensor.name, (1,) + inputTensor.shape[1:], (8,) + inputTensor.shape[1:], (32,) + inputTensor.shape[1:])
    config.add_optimization_profile(profile)
    
    engineString = builder.build_serialized_network(network, config)
    mg.meta["tensorRT_string"] = engineString
    builder.build_engine(network, config)

    with open(TR_output_path, 'wb') as f:
        f.write(engineString)

    return mg,{}

def run_tensorrt_pass(mg, dataloader):
    logger = trt.Logger(trt.Logger.ERROR)
    engine = trt.Runtime(logger).deserialize_cuda_engine(mg.meta["tensorRT_string"])
    context = engine.create_execution_context()

    nIO = engine.num_io_tensors
    lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
    nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)

    dataiter = iter(dataloader)
    input, labels = next(dataiter)
    input_shape = input.shape
    context.set_input_shape(lTensorName[0], input_shape)

    execute_time = []
    accuracy = []

    for inputs in dataloader:
        data, label = inputs
        bufferH = []
        bufferH.append(np.ascontiguousarray(data))
        for i in range(nInput, nIO):
            bufferH.append(np.zeros(context.get_tensor_shape(lTensorName[i]), dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[i]))))
        bufferD = []
        for i in range(nIO):
            bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

        for i in range(nInput):
            cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

        for i in range(nIO):
            context.set_tensor_address(lTensorName[i], int(bufferD[i]))

        start_time = time.time()
        context.execute_async_v3(0)
        end_time = time.time()
        execute_time.append(end_time - start_time)

        for i in range(nInput, nIO):
            cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
            categories = np.argmax(bufferH[nInput], axis=1)
            acc = np.sum(categories == np.array(label)) / len(label)
            accuracy.append(acc)

            for b in bufferD:
                cudart.cudaFree(b)

    print("Succeeded running model in TensorRT!")
    print("Average execute time for one batch: %.2fms" % (sum(execute_time) / len(execute_time) * 1000))
    print("Total accuracy: %.2f%%" % (sum(accuracy) / len(accuracy) * 100))
    return (sum(accuracy) / len(accuracy) * 100),(sum(execute_time) / len(execute_time) * 1000)