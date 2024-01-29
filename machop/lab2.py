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

set_logging_verbosity("info")

batch_size = 8
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

CHECKPOINT_PATH = "../mase_output/jsc-tiny_classification_jsc_2024-01-25/software/training_ckpts/best-v1.ckpt"

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

#######################################################################

# # report graph is an analysis pass that shows you the detailed information in the graph
# from chop.passes.graph import report_graph_analysis_pass
# _ = report_graph_analysis_pass(mg)

#Running another Analysis pass: Profile statistics

########################################################################
# pass_args = {
#     "by": "type",                                                            # collect statistics by node name
#     "target_weight_nodes": ["linear"],                                       # collect weight statistics for linear layers
#     "target_activation_nodes": ["relu"],                                     # collect activation statistics for relu layers
#     "weight_statistics": {
#         "variance_precise": {"device": "cpu", "dims": "all"},                # collect precise variance of the weight
#     },
#     "activation_statistics": {
#         "range_quantile": {"device": "cpu", "dims": "all", "quantile": 0.97} # collect 97% quantile of the activation range
#     },
#     "input_generator": input_generator,                                      # the input generator for feeding data to the model
#     "num_samples": 32,                                                       # feed 32 samples to the model
# }

# mg, _ = profile_statistics_analysis_pass(mg, pass_args)

# mg, _ = report_node_meta_param_analysis_pass(mg, {"which": ("software",)})

########################################################################

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
    },
}

pass_args_repo = {
    "by": "type",                                                            # collect statistics by node name
    "target_weight_nodes": ["linear"],                                       # collect weight statistics for linear layers
    "target_activation_nodes": ["relu"],                                     # collect activation statistics for relu layers
    "weight_statistics": {
        "variance_precise": {"device": "cpu", "dims": "all"},                # collect precise variance of the weight
    },
    "activation_statistics": {
        "range_quantile": {"device": "cpu", "dims": "all", "quantile": 0.97} # collect 97% quantile of the activation range
    },
    "input_generator": input_generator,                                      # the input generator for feeding data to the model
    "num_samples": 32,                                                       # feed 32 samples to the model
}

from chop.passes.graph.transforms import (
    quantize_transform_pass,
    summarize_quantization_analysis_pass,
)
from chop.ir.graph.mase_graph import MaseGraph


ori_mg = MaseGraph(model=model)
ori_mg, _ = init_metadata_analysis_pass(ori_mg, None)
ori_mg, _ = add_common_metadata_analysis_pass(ori_mg, {"dummy_in": dummy_in})

mg, _ = quantize_transform_pass(mg, pass_args)
summarize_quantization_analysis_pass(ori_mg, mg, save_dir="quantize_summary")

## Write some code to traverse both `mg` and `ori_mg`,
## check and comment on the nodes in these two graphs. 
## You might find the source code for the implementation of `summarize_quantization_analysis_pass` useful.

print("===============================================\n")
print("Graph Report for MG IR \n ")
print("===============================================")

mg, _ = profile_statistics_analysis_pass(mg, pass_args_repo)
mg, _ = report_node_meta_param_analysis_pass(mg, {"which": ("software",)})

print("===============================================\n")
print("Graph Report for ori_mg IR \n ")
print("===============================================")

ori_mg, _ = add_software_metadata_analysis_pass(mg, None)
ori_mg, _ = profile_statistics_analysis_pass(ori_mg, pass_args_repo)
ori_mg, _ = report_node_meta_param_analysis_pass(ori_mg, {"which": ("software",)})