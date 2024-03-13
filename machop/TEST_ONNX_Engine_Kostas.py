import sys
import logging
import os
from pathlib import Path
from pprint import pprint as pp
import time
from cuda import cudart


###########################################################
# Read me!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
#
# The file is used for test the written pass for the final Lab
# suitable path is : mase/machop/USEDforTEST_lab3.py
############################################################


# figure out the correct path
# machop_path = Path(".").resolve().parent.parent /"machop"
machop_path = Path(".").resolve()
assert machop_path.exists(), "Failed to find machop at: {}".format(machop_path)
sys.path.append(str(machop_path))

from chop.dataset import MaseDataModule, get_dataset_info
from chop.tools.logger import get_logger

from chop.passes.graph.analysis import (
    report_node_meta_param_analysis_pass,
    profile_statistics_analysis_pass,
)
from chop.passes.graph import (
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    add_software_metadata_analysis_pass,
)
from chop.tools.get_input import InputGenerator
from chop.ir.graph.mase_graph import MaseGraph

from chop.models import get_model_info, get_model

from chop.passes.graph.transforms import (
    quantize_transform_pass,
    summarize_quantization_analysis_pass,
)

from chop.passes.graph.transforms.quantize.quantize_tensorRT import tensorRT_quantize_pass,calibration_pass

from lab2_op_floppass import flop_calculator_pass,modlesize_calculator_pass
from chop.tools.checkpoint_load import load_model
from chop.plt_wrapper import get_model_wrapper

logger = get_logger("chop")
logger.setLevel(logging.INFO)

batch_size = 16
model_name = "vgg7"
dataset_name = "cifar10"


data_module = MaseDataModule(
    name=dataset_name,
    batch_size=batch_size,
    model_name=model_name,
    num_workers=0,
    # custom_dataset_cache_path="../../chop/dataset"
)
data_module.prepare_data()
data_module.setup()

CHECKPOINT_PATH = "./VGG_checkPoint/test-accu-0.9332.ckpt"


model_info = get_model_info(model_name)

model = get_model(
    model_name,
    task="cls",
    dataset_info=data_module.dataset_info,
    pretrained=False)

model = load_model(load_name=CHECKPOINT_PATH, load_type="pl", model=model)

input_generator = InputGenerator(
    data_module=data_module,
    model_info=model_info,
    task="cls",
    which_dataloader="train",
)

dummy_in = next(iter(input_generator))
_ = model(**dummy_in)

# generate the mase graph and initialize node metadata
mg = MaseGraph(model=model)


###########################################################
#                   Define a Search Space                   #
###########################################################
pass_args = {
"by": "type",
"default": {"config": {"name": None}},
"linear": {
        "config": {
            "name": "integer",
            # data
            "data_in_width": 8,
            "data_in_frac_width": 4,
            # weight
            "weight_width": 8,
            "weight_frac_width": 4,
            # bias
            "bias_width": 8,
            "bias_frac_width": 4,
        }
},}

###########################################################
#              Define a Search Strategy                   #
###########################################################

import torch
from torchmetrics.classification import MulticlassAccuracy
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
metric = MulticlassAccuracy(num_classes=5)
metric = metric.to(device)
num_batchs = 5

####################
# A test Pass
####################
def run_model(mg, device, data_module, metric, num_batches):
    j = 0
    accs, losses = [], []
    mg.model = mg.model.to(device)
    inputs_tuple = ()
    for inputs in data_module.train_dataloader():
        xs, ys = inputs
        xs, ys = xs.to(device), ys.to(device)
        preds = mg.model(xs)
        loss = torch.nn.functional.cross_entropy(preds, ys)
        acc = metric(preds, ys)
        accs.append(acc)
        losses.append(loss)
        if j > num_batches:
            break
        j += 1
        # for name in mg.modules.keys():
        #     if name.endswith('_quantizer'):
        #         print(mg.modules[name].weight())
        inputs_tuple += (inputs,)
    acc_avg = sum(accs) / len(accs)
    loss_avg = sum(losses) / len(losses)
    return acc_avg, loss_avg, inputs_tuple

import pytorch_quantization

#########################################################

import onnx
import tensorrt as trt
import pdb 
import numpy as np
from cuda import cudart

# Create execution context
with open('testdemo_engine.trt', 'rb') as f:
    engine_str = f.read()

runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
engine = runtime.deserialize_cuda_engine(engine_str)
context = engine.create_execution_context()

"""
Test the quantize_tensorrt_transform_pass function.

:param pass_args: A dictionary of arguments for the pass.
:type pass_args: dict
"""

logger = trt.Logger(trt.Logger.ERROR)

nIO = engine.num_io_tensors
lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)

context = engine.create_execution_context()

dataloader = data_module.train_dataloader()
dataiter = iter(dataloader)
input, labels = next(dataiter)
input_shape = input.shape
context.set_input_shape(lTensorName[0], input_shape)
for i in range(nIO):
    print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]), engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])

execute_time = []
accuracy = []

## prepare the cuda
# cuda.init()
# Get the first CUDA device
device = cuda.Device(0)
# Create a CUDA context for the device
cuda_context = device.make_context()

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

for i in range(nInput, nIO):
    cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
    pdb.set_trace()
    categories = np.argmax(bufferH[nInput], axis=1)
    # print(categories, label)
    acc = np.sum(categories == np.array(labels)) / len(labels)
    # print("Accuracy: %.2f%%" % (acc * 100))
    accuracy.append(acc)
    
    # for i in range(nIO):
    #     print(lTensorName[i])
    #     print(bufferH[i])
    #     print(categories, label)

    for b in bufferD:
        cudart.cudaFree(b)


print("Succeeded running model in TensorRT!")
print("Average execute time for one batch: %.2fms" % (sum(execute_time) / len(execute_time) * 1000))
print("Total accuracy: %.2f%%" % (sum(accuracy) / len(accuracy) * 100))


def test_quantize_tensorrt_transform_pass(dataloader, engine):
    """
    Test the quantize_tensorrt_transform_pass function.
 
    :param pass_args: A dictionary of arguments for the pass.
    :type pass_args: dict
    """
 
    # Load engineString from file
    logger = trt.Logger(trt.Logger.ERROR)
    #     print("engine.__len__() = %d" % len(engine))
    #     print("engine.__sizeof__() = %d" % engine.__sizeof__())
    #     print("engine.__str__() = %s" % engine.__str__())
 
    #     print("\nEngine related ========================================================")
    
    inspector = engine.create_engine_inspector()
    print("inspector.execution_context=", inspector.execution_context)
    print("inspector.error_recorder=", inspector.error_recorder)  # ErrorRecorder can be set into EngineInspector, usage of ErrorRecorder refer to 02-API/ErrorRecorder
 
    print("Engine information:")  # engine information is equivalent to put all layer information together
    print(inspector.get_engine_information(trt.LayerInformationFormat.ONELINE))  # .txt format
    #print(inspector.get_engine_information(trt.LayerInformationFormat.JSON))  # .json format
 
    print("Layer information:")
    for i in range(engine.num_layers):
        print(inspector.get_layer_information(i, trt.LayerInformationFormat.ONELINE))
    
    nIO = engine.num_io_tensors
    lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
    nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)
 
    context = engine.create_execution_context()
 
    dataiter = iter(dataloader())
    input, labels = next(dataiter)
    input_shape = input.shape
    context.set_input_shape(lTensorName[0], input_shape)
    for i in range(nIO):
        print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]), engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])
 
    execute_time = []
    accuracy = []
    for data, label in dataloader():
        bufferH = []
        bufferH.append(np.ascontiguousarray(data))
        for i in range(nInput, nIO):
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
        execute_time.append(time.time() - start_time)
    
        for i in range(nInput, nIO):
            cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
            
            categories = np.argmax(bufferH[nInput], axis=1)
            # print(categories, label)
            acc = np.sum(categories == np.array(label)) / len(label)
            # print("Accuracy: %.2f%%" % (acc * 100))
            accuracy.append(acc)
        
        # for i in range(nIO):
        #     print(lTensorName[i])
        #     print(bufferH[i])
        #     print(categories, label)
 
        for b in bufferD:
            cudart.cudaFree(b)


    print("Succeeded running model in TensorRT!")
    print("Average execute time for one batch: %.2fms" % (sum(execute_time) / len(execute_time) * 1000))
    print("Total accuracy: %.2f%%" % (sum(accuracy) / len(accuracy) * 100))

# Allocate device memory
# inputs = []
# outputs = []
# bindings = []
# for binding in engine:
#     size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
#     dtype = trt.nptype(engine.get_binding_dtype(binding))
#     # Allocate device memory for inputs.
#     input = np.random.random(size).astype(dtype)
#     input = input.ravel()
#     inputs.append(input)
#     # Allocate device memory for outputs.
#     output = np.empty(size, dtype=dtype)
#     outputs.append(output)
#     # Append the device buffer to device bindings.
#     bindings.append(int(input.data_ptr()))
#     bindings.append(int(output.data_ptr()))

# # Assume input_data is a numpy array holding your input data
# input_data = next(iter(input_generator))['x']
# input_data = input_data.cuda()
# labels = next(iter(input_generator))['y']
# labels = labels.cuda()

# np.copyto(inputs[0].host_mem, input_data.ravel())

# # Transfer input data to the GPU.
# cuda.memcpy_htod(inputs[0].device_mem, inputs[0].host_mem)

# # Run inference
# context.execute_async_v3(0)

# # Transfer predictions back from the GPU.
# cuda.memcpy_dtoh(outputs[0].host_mem, outputs[0].device_mem)

# # Compute accuracy
# predictions = np.argmax(outputs[0].host_mem, axis=1)
# accuracy = np.mean(predictions == labels)

# print("Accuracy: ", accuracy)



# def excute_tensorRT_engine(engineString,input_,label):
#     import onnx
#     import tensorrt as trt
#     import pdb 
#     import numpy as np
#     import pycuda.driver as cuda
#     runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
#     engine = runtime.deserialize_cuda_engine(engineString)

#     # Create execution context
#     context = engine.create_execution_context()

#     # Allocate device memory
#     inputs = []
#     outputs = []
#     bindings = []
#     for binding in engine:
#         size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
#         dtype = trt.nptype(engine.get_binding_dtype(binding))
#         # Allocate device memory for inputs.
#         input = np.random.random(size).astype(dtype)
#         input = input.ravel()
#         inputs.append(input)
#         # Allocate device memory for outputs.
#         output = np.empty(size, dtype=dtype)
#         outputs.append(output)
#         # Append the device buffer to device bindings.
#         bindings.append(int(input.data_ptr()))
#         bindings.append(int(output.data_ptr()))

#     # Assume input_data is a numpy array holding your input data
#     input_data = np.random.random_sample(inputs[0].host_mem.shape)
#     np.copyto(inputs[0].host_mem, input_data.ravel())

#     # Transfer input data to the GPU.
#     cuda.memcpy_htod(inputs[0].device_mem, inputs[0].host_mem)

#     # Run inference
#     context.execute(batch_size=1, bindings=bindings)

#     # Transfer predictions back from the GPU.
#     cuda.memcpy_dtoh(outputs[0].host_mem, outputs[0].device_mem)

#     # Compute accuracy
#     predictions = np.argmax(outputs[0].host_mem, axis=1)
#     labels = ...  # You need to provide the true labels here
#     accuracy = np.mean(predictions == labels)

#     print("Accuracy: ", accuracy)

