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
from .modify_tensorRT import create_new_module_tensorRT
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
            import pdb; pdb.set_trace()
            ori_module = get_node_actual_target(node)
            successor_module = get_similar_node_actual_target(
                bl_graph, node.next
            )  # Certain modules will require information about their successor module to complete the initialization process. (For LogicNets, activation functions are needed.)
            bl_module = get_similar_node_actual_target(bl_graph, node)
            new_module = create_new_module_tensorRT(
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
            import pdb; pdb.set_trace()
            ori_module = get_node_actual_target(node)
            successor_module = get_similar_node_actual_target(
                bl_graph, node.next
            )  # Certain modules will require information about their successor module to complete the initialization process. (For LogicNets, activation functions are needed.)
            bl_module = get_similar_node_actual_target(bl_graph, node)
            new_module = create_new_module_tensorRT(
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

def tensorRT_quantize_pass(graph, pass_args=None,fake = False):
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
    
    # criterion = nn.CrossEntropyLoss()
    # with torch.no_grad():
    #     evaluate(model, criterion, data_loader_test, device="cuda", print_freq=20)
    graph.model.to(device)

    return graph, {}


def export_onnx_to_tensorRT_engine_pass(mg,dummy_in,input_generator,onnx_model_path,TR_output_path):
    dummy_in = next(iter(input_generator))['x']
    dummy_in = dummy_in.cuda()
    torch.onnx.export(mg.model.cuda(),  dummy_in.cuda(), onnx_model_path, export_params=True, opset_version=13, do_constant_folding=True, \
                        input_names = ['input'], output_names = ['output'], dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}})
    
    TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    ## ONLY test for the DEMO
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    # config.set_flag(trt.BuilderFlag.INT8)
    # Parse the ONNX model
    with open(onnx_model_path, 'rb') as model:
        print("parser.parse(model.read()): ",str(parser.parse(model.read())))
        for error in range(parser.num_errors):
            print(parser.get_error(error))

    # print(parser.parse(onnx_model.SerializeToString()))

    profile = builder.create_optimization_profile()
    inputTensor = network.get_input(0)
    profile.set_shape(inputTensor.name, (1,) + inputTensor.shape[1:], (8,) + inputTensor.shape[1:], (32,) + inputTensor.shape[1:])
    config.add_optimization_profile(profile)

    engineString = builder.build_serialized_network(network, config)

    with open(TR_output_path, 'wb') as f:
        f.write(engineString)

    return mg, {}, builder.build_engine(network, config)

def test_quantize_tensorrt_transform_pass(dataloader, engine_str_path,model_name,ONNX_engine):
    """
    Test the quantize_tensorrt_transform_pass function.
 
    :param pass_args: A dictionary of arguments for the pass.
    :type pass_args: dict
    """
    import pdb; pdb.set_trace()
    # with open(engine_str_path, 'rb') as f:
    #     engine_str = f.read()

    logger = trt.Logger(trt.Logger.ERROR)
    with open(engine_str_path, "rb") as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())

    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    # engine = runtime.deserialize_cuda_engine(engine_str)
    context = engine.create_execution_context()

    logger = trt.Logger(trt.Logger.ERROR)

    nIO = engine.num_io_tensors
    lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
    nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)

    context = engine.create_execution_context()
    dataiter = iter(dataloader)
    input, labels = next(dataiter)
    # if model_name == "vgg7":
    #     input.requires_grad_(True)
    input_shape = input.shape
    context.set_input_shape(lTensorName[0], input_shape)
    for i in range(nIO):
        print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]), engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])

    execute_time = []
    accuracy = []

    for data, label in dataloader:
        bufferH = []
        bufferH.append(np.ascontiguousarray(data))
        for i in range(nInput, nIO):
            ####记得改成np.zeros
            bufferH.append(np.empty(context.get_tensor_shape(lTensorName[i]), dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[i]))))
        bufferD = []
        for i in range(nIO):
            bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

        # for i in range(nInput):
        #     cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

        for i in range(nInput):
            cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

        for i in range(nIO):
            context.set_tensor_address(lTensorName[i], int(bufferD[i]))

    start_time = time.time()
    context.execute_async_v3(0)
    execute_time.append(time.time() - start_time)

    import pdb; pdb.set_trace()

    for i in range(nInput, nIO):
        import pdb; pdb.set_trace()
        cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
        categories = np.argmax(bufferH[nInput], axis=1)
        # print(categories, label)
        acc = np.sum(categories == np.array(labels)) / len(labels)
        # print("Accuracy: %.2f%%" % (acc * 100))
        accuracy.append(acc)

        for b in bufferD:
            cudart.cudaFree(b)


    print("Succeeded running model in TensorRT!")
    print("Average execute time for one batch: %.2fms" % (sum(execute_time) / len(execute_time) * 1000))
    print("Total accuracy: %.2f%%" % (sum(accuracy) / len(accuracy) * 100))


def DEMO_combine_ONNX_and_Engine(mg,dummy_in,dataloader,input_generator,onnx_model_path,TR_output_path):
    dummy_in = next(iter(input_generator))['x']
    dummy_in = dummy_in.cuda()
    torch.onnx.export(mg.model.cuda(),  dummy_in.cuda(), onnx_model_path, export_params=True, opset_version=13, do_constant_folding=True, \
                        input_names = ['input'], output_names = ['output'], dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}})
    
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    ## ONLY test for the DEMO
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    parser = trt.OnnxParser(network, logger)
    # config.set_flag(trt.BuilderFlag.INT8)
    # Parse the ONNX model
    with open(onnx_model_path, 'rb') as model:
        print("parser.parse(model.read()): ",str(parser.parse(model.read())))
        for error in range(parser.num_errors):
            print(parser.get_error(error))

    inputTensor = network.get_input(0)
    profile.set_shape(inputTensor.name, (1,) + inputTensor.shape[1:], (8,) + inputTensor.shape[1:], (32,) + inputTensor.shape[1:])
    config.add_optimization_profile(profile)

    engineString = builder.build_serialized_network(network, config)
    builder.build_engine(network, config)

    with open(TR_output_path, 'wb') as f:
        f.write(engineString)

    with open(TR_output_path, "rb") as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    nIO = engine.num_io_tensors
    lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
    nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)

    dataiter = iter(dataloader)
    input, labels = next(dataiter)
    input_shape = input.shape
    context.set_input_shape(lTensorName[0], input_shape)
    for i in range(nIO):
        print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]), engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])

    execute_time = []
    accuracy = []

    for inputs in dataloader:
        data, label = inputs
        bufferH = []
        bufferH.append(np.ascontiguousarray(data))
        for i in range(nInput, nIO):
            ####记得改成np.zeros
            bufferH.append(np.empty(context.get_tensor_shape(lTensorName[i]), dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[i]))))
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
            # print(categories, label)
            acc = np.sum(categories == np.array(label)) / len(label)
            # print("Accuracy: %.2f%%" % (acc * 100))
            accuracy.append(acc)

            for b in bufferD:
                cudart.cudaFree(b)


    print("Succeeded running model in TensorRT!")
    print("Average execute time for one batch: %.2fms" % (sum(execute_time) / len(execute_time) * 1000))
    print("Total accuracy: %.2f%%" % (sum(accuracy) / len(accuracy) * 100))
    return (sum(accuracy) / len(accuracy) * 100),(sum(execute_time) / len(execute_time) * 1000)


def DEMO_combine_ONNX_and_Engine_run_all_files(mg,dummy_in,dataloader,input_generator,onnx_model_path,TR_output_path):
    dummy_in = next(iter(input_generator))['x']
    dummy_in = dummy_in.cuda()
    torch.onnx.export(mg.model.cuda(),  dummy_in.cuda(), onnx_model_path, export_params=True, opset_version=13, do_constant_folding=True, \
                        input_names = ['input'], output_names = ['output'], dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}})
    
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    ## ONLY test for the DEMO
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    parser = trt.OnnxParser(network, logger)
    # config.set_flag(trt.BuilderFlag.INT8)
    # Parse the ONNX model
    with open(onnx_model_path, 'rb') as model:
        print("parser.parse(model.read()): ",str(parser.parse(model.read())))
        for error in range(parser.num_errors):
            print(parser.get_error(error))

    inputTensor = network.get_input(0)
    profile.set_shape(inputTensor.name, (1,) + inputTensor.shape[1:], (8,) + inputTensor.shape[1:], (32,) + inputTensor.shape[1:])
    config.add_optimization_profile(profile)

    engineString = builder.build_serialized_network(network, config)
    builder.build_engine(network, config)

    with open(TR_output_path, 'wb') as f:
        f.write(engineString)

    with open(TR_output_path, "rb") as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    nIO = engine.num_io_tensors
    lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
    nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)

    dataiter = iter(dataloader)
    input, labels = next(dataiter)
    input_shape = input.shape
    context.set_input_shape(lTensorName[0], input_shape)
    for i in range(nIO):
        print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]), engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])

    execute_time = []
    accuracy = []

    for data, label in dataloader:
        bufferH = []
        bufferH.append(np.ascontiguousarray(data))
        for i in range(nInput, nIO):
            ####记得改成np.zeros
            bufferH.append(np.empty(context.get_tensor_shape(lTensorName[i]), dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[i]))))
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
        # print(categories, label)
        acc = np.sum(categories == np.array(label)) / len(label)
        # print("Accuracy: %.2f%%" % (acc * 100))
        accuracy.append(acc)

        for b in bufferD:
            cudart.cudaFree(b)


    print("Succeeded running model in TensorRT!")
    print("Average execute time for one batch: %.2fms" % (sum(execute_time) / len(execute_time) * 1000))
    print("Total accuracy: %.2f%%" % (sum(accuracy) / len(accuracy) * 100))
    return (sum(accuracy) / len(accuracy) * 100),(sum(execute_time) / len(execute_time) * 1000)


#########################################################################################
# Change the DEMO into ELEGANT code below
#########################################################################################
    
def export_onnx_and_tensorrt_engine_pass(mg, dummy_in, input_generator, onnx_model_path,TR_output_path):
    dummy_in = next(iter(input_generator))['x']
    dummy_in = dummy_in.cuda()
    torch.onnx.export(mg.model.cuda(),  dummy_in.cuda(), onnx_model_path, export_params=True, opset_version=13, do_constant_folding=True, \
                        input_names = ['input'], output_names = ['output'], dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}})
    
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

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
    builder.build_engine(network, config)

    with open(TR_output_path, 'wb') as f:
        f.write(engineString)

    return builder, network, config, profile, logger

def execute_tensorrt_engine(builder, network, config, profile, logger, TR_output_path, dataloader):
    with open(TR_output_path, "rb") as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    nIO = engine.num_io_tensors
    lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
    nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)

    dataiter = iter(dataloader)
    input, labels = next(dataiter)
    input_shape = input.shape
    context.set_input_shape(lTensorName[0], input_shape)

    # ... rest of the code for executing the TensorRT engine ...