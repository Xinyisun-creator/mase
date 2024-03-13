"""
Some of the functions here are taken from the Modifier class we had before
"""

from typing import Dict

import torch
from chop.tools.utils import copy_weights, init_LinearLUT_weight, init_Conv2dLUT_weight
from torch import nn
import numpy as np

from .quantized_funcs import quantized_func_map
from .quantized_modules import quantized_module_map

from torch import nn

from pytorch_quantization import tensor_quant
import pytorch_quantization.nn as quant_nn

from pytorch_quantization import tensor_quant
from cuda import cudart

import onnx
import tensorrt as trt

type_to_name_map = {
    nn.Linear: "linear",
    nn.Conv1d: "conv1d",
    nn.Conv2d: "conv2d",
}

###########################################################
# Read me!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
#
# The file is used for modify layers/modules in the maze graph
# suitable path is : mase/machop/chop/passes/graph/transforms/quantize/modify_tensorRT.py
############################################################


def create_new_module_tensorRT(
    mase_op: str,
    original_module: nn.Module,
    config: dict,
    node_meta: dict,
    baseline_module: nn.Module = None,
    successor_module: nn.Module = None,
    input_layers=None,
    output_layers=None,
):
    original_module_cls = type(original_module)
    quant_name = config.get("name")

    if mase_op == "linear":
        print("Start to modify the linear layer to quantized layers")
        new_module_cls = quantized_module_map[f"linear_{quant_name}"]
        use_bias = original_module.bias is not None
        QUANTIZE_DESC_INPUT_LAYER = tensor_quant.ScaledQuantDescriptor(num_bits=config["data_in_width"], axis=(0))
        QUANTIZE_DESC_WEIGHT_LAYER = tensor_quant.ScaledQuantDescriptor(num_bits=config['weight_width'], axis=(0))
        new_module = quant_nn.Linear(
        in_features = original_module.in_features,
        out_features = original_module.out_features,
        bias=True,
        quant_desc_input=QUANTIZE_DESC_INPUT_LAYER,
        quant_desc_weight=QUANTIZE_DESC_WEIGHT_LAYER)

        copy_weights(original_module.weight, new_module.weight)
        if use_bias:
            copy_weights(original_module.bias, new_module.bias)

        print("Quantized Linear Layers are created successfully.")
        
    elif mase_op == "conv2d":
        print("Start to modify the Conv layer to quantized layers")
        new_module_cls = quantized_module_map[f"conv2d_{quant_name}"]
        use_bias = original_module.bias is not None
        QUANTIZE_DESC_INPUT_LAYER = tensor_quant.ScaledQuantDescriptor(num_bits=config["data_in_width"], axis=(0))
        QUANTIZE_DESC_WEIGHT_LAYER = tensor_quant.ScaledQuantDescriptor(num_bits=config['weight_width'], axis=(0))

        new_module = quant_nn.Conv2d(
            in_channels = original_module.in_channels,
            out_channels = original_module.out_channels,
            kernel_size = original_module.kernel_size,
            stride = original_module.stride,
            padding = original_module.padding,
            dilation = original_module.dilation,
            groups = original_module.groups,
            bias = True,
            quant_desc_input=QUANTIZE_DESC_INPUT_LAYER,
            quant_desc_weight=QUANTIZE_DESC_WEIGHT_LAYER)
        
        copy_weights(original_module.weight, new_module.weight)
        if use_bias:
            copy_weights(original_module.bias, new_module.bias)
        
        print("Quantized conv2d Layers are created successfully.")

    # elif mase_op == "max_pool2d": ## not sure yet

    #     print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!pay attension!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    #     import pdb; pdb.set_trace()
    #     print("Start to modyfy the maxpool2d layer to quantized layers")
    #     new_module_cls = quantized_module_map[f"conv2d_{quant_name}"]
    #     use_bias = original_module.bias is not None
    #     QUANTIZE_DESC_INPUT_LAYER = tensor_quant.ScaledQuantDescriptor(num_bits=config["data_in_width"], axis=(0))
    #     QUANTIZE_DESC_WEIGHT_LAYER = tensor_quant.ScaledQuantDescriptor(num_bits=config['weight_width'], axis=(0))

    #     new_module = quant_nn.QuantMaxPool2d(
    #         kernel_size = original_module.kernel_size, 
    #         stride = original_module.stride, 
    #         padding = original_module.padding, 
    #         dilation=original_module.dilation,
    #         return_indices=False, 
    #         ceil_mode=False)
        
    #     print("maxpool2d conv2d Layers are created successfully.")

    else:
        raise NotImplementedError(
            f"Unsupported module class {original_module_cls} to modify"
        )
    return new_module


def create_new_fn(node, config: Dict):
    mase_op = node.meta["mase"].parameters["common"]["mase_op"]
    quant_name = config.get("name")
    func_name = f"{mase_op}_{quant_name}"
    new_func = quantized_func_map[func_name]
    args, kwargs = node.args, node.kwargs | {"config": config}
    return new_func, args, kwargs
