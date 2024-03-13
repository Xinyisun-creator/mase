import sys
import logging
import os
from pathlib import Path
from pprint import pprint as pp

from chop.dataset import MaseDataModule, get_dataset_info
from chop.tools.logger import set_logging_verbosity

from chop.passes.graph import (
    save_node_meta_param_interface_pass,
    report_node_meta_param_analysis_pass,
    profile_statistics_analysis_pass,
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    add_software_metadata_analysis_pass,
)
from chop.tools.get_input import InputGenerator
from chop.tools.checkpoint_load import load_model
from chop.ir import MaseGraph

from chop.models import get_model_info, get_model

from chop.passes.graph.transforms import (
    quantize_transform_pass,
    summarize_quantization_analysis_pass,
)
from chop.ir.graph.mase_graph import MaseGraph

from lab2_op_floppass import flop_calculator_pass,BitOPs_calculator_pass

set_logging_verbosity("info")

# batch_size = 8
# model_name = "jsc-lab1"
# dataset_name = "jsc"

batch_size = 256
model_name = "jsc-tiny"
dataset_name = "jsc"


data_module = MaseDataModule(
    name=dataset_name,
    batch_size=batch_size,
    model_name=model_name,
    num_workers=0,
)
data_module.prepare_data()
data_module.setup()

# CHECKPOINT_PATH = "../mase_output/jsc-lab1_classification_jsc_2024-01-23/software/training_ckpts/best.ckpt"
CHECKPOINT_PATH = "../mase_output/jsc-tiny_classification_jsc_2024-01-24/software/training_ckpts/best.ckpt"
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

# a demonstration of how to feed an input value to the model
dummy_in = next(iter(input_generator))
_ = model(**dummy_in)

# generate the mase graph and initialize node metadata
mg = MaseGraph(model=model)

mg, _ = init_metadata_analysis_pass(mg, None)
mg, _ = add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in})
mg, _ = add_software_metadata_analysis_pass(mg, None)

flops_linear,flops_relu,flops_norm1d = flop_calculator_pass(mg, None)
# flops_linear,flops_relu,flops_norm1d = BitOPs_calculator_pass(mg, None)
print(flops_linear)
print(flops_relu)
print(flops_norm1d)