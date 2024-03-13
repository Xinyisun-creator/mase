import sys
import logging
import os
from pathlib import Path
from pprint import pprint as pp
import time
import onnx
import tensorrt as trt
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

from chop.passes.graph.transforms.quantize.quantize_tensorRT import tensorRT_quantize_pass,calibration_pass,export_onnx_to_tensorRT_engine_pass,test_quantize_tensorrt_transform_pass

from lab2_op_floppass import flop_calculator_pass,modlesize_calculator_pass

logger = get_logger("chop")
logger.setLevel(logging.INFO)

batch_size = 8
model_name = "jsc-tiny"
dataset_name = "jsc"


data_module = MaseDataModule(
    name=dataset_name,
    batch_size=batch_size,
    model_name=model_name,
    num_workers=0,
    # custom_dataset_cache_path="../../chop/dataset"
)
data_module.prepare_data()
data_module.setup()

model_info = get_model_info(model_name)
model = get_model(
    model_name,
    task="cls",
    dataset_info=data_module.dataset_info,
    pretrained=False,
    checkpoint = None)

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
mg, _ = init_metadata_analysis_pass(mg, None)
mg, _ = add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in})
mg, _ = add_software_metadata_analysis_pass(mg, None)
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

# def Export_ONNX_pass_TensorRT(graph, input,pass_args=None):
#     with pytorch_quantization.enable_onnx_export():
#         # enable_onnx_checker needs to be disabled. See notes below.
#         torch.onnx.export(mg.model.cuda(), dummy_input = input, onnxFile, export_params=True, opset_version=13, do_constant_folding=True, \
#                       input_names = ['input'], output_names = ['output'], dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}})

##
# New passes used for the final LAB in ADL
##

#########################################################
#      Original Maze Graph
#########################################################

engine_str_path = './testdemo_engine_ori.trt'
mg, _ = export_onnx_to_tensorRT_engine_pass(mg,dummy_in,input_generator,onnx_model_path = './testdemo_ori.onnx',TR_output_path=engine_str_path)
test_quantize_tensorrt_transform_pass(data_module.train_dataloader(), engine_str_path)

##
# ask Aron about the following
##

###
# - replace the original layers by the quantized layers (what we do at now)
###

# calibration_pass: calibrate the quantized layers with some inputs
mg, _ = calibration_pass(mg, pass_args,data_module)

# tensorRT_quantize_pass: modify some layers from network into fake quantized layers
mg, _ = tensorRT_quantize_pass(mg, pass_args,fake = True) 

engine_str_path = './testdemo_engine.trt'
# export_onnx_to_tensorRT_engine_pass: export the quantized model into tensorRT engine
mg, _ = export_onnx_to_tensorRT_engine_pass(mg,dummy_in,input_generator,onnx_model_path = './testdemo.onnx',TR_output_path=engine_str_path)
# test_quantize_tensorrt_transform_pass: test the quantized model with the tensorRT engine
test_quantize_tensorrt_transform_pass(data_module.train_dataloader(), engine_str_path)

# acc_avg, loss_avg, inputs_tuple = run_model(mg, device, data_module, metric, num_batchs)
# Export_ONNX_pass_TensorRT(mg, inputs_tuple)
#########################################################
#       ONNX EXPORT
#########################################################
