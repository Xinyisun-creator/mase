
# grid search
import logging

import torch
from torchmetrics.classification import MulticlassAccuracy
import copy
from torch import nn
from chop.passes.graph.utils import get_parent_name
from os import PathLike
from chop.tools.get_input import InputGenerator
from chop.tools.checkpoint_load import load_model
from chop.ir import MaseGraph

from chop.passes.graph import (
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    add_software_metadata_analysis_pass,
)

from chop.models import get_model_info, get_model

from ...tools.config_load import load_config

from os import PathLike

#################################################
#                Initial Config                 #
#################################################

# pass_config = {
# "by": "name",
# "default": {"config": {"name": None}},
# "seq_blocks_2": {
#     "config": {
#         "name": "output_only",
#         # weight
#         "channel_multiplier": 2,
#         }
# },

# "seq_blocks_4": {
#     "config": {
#         "name": "both",
#         # weight
#         "channel_multiplier": 2,
#         }
# },

# "seq_blocks_6": {
#     "config": {
#         "name": "input_only",
#         # weight
#         "channel_multiplier": 2,
#         }
# }
# }

#################################################
#            Transformation pass                #
#################################################

def instantiate_linear(in_features, out_features, bias):
    if bias is not None:
        bias = True
    return nn.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=bias)

def redefine_linear_transform_pass(graph, pass_args=None):
    main_config = pass_args.pop('config')
    default = main_config.pop('default', None)
    import pdb
    pdb.set_trace()
    if default is None:
        raise ValueError(f"default value must be provided.")
    i = 0
    for node in graph.fx_graph.nodes:
        i += 1
        # if node name is not matched, it won't be tracked
        config = main_config.get(node.name, default)['config']
        name = config.get("name", None)
        if name is not None:
            ori_module = graph.modules[node.target]
            # import pdb
            # pdb.set_trace()
            in_features = ori_module.in_features
            out_features = ori_module.out_features
            bias = ori_module.bias
            if name == "output_only":
                out_features = out_features * config["channel_multiplier"]
            elif name == "both":
                in_features = in_features * config["channel_multiplier"]
                out_features = out_features * config["channel_multiplier"]
            elif name == "input_only":
                in_features = in_features * config["channel_multiplier"]
            new_module = instantiate_linear(in_features, out_features, bias)
            parent_name, name = get_parent_name(node.target)
            setattr(graph.modules[parent_name], name, new_module)
    return graph, {}

def parse_search_config(search_config):
    """
    Parse search config from a dict or a toml file and do sanity check.

    ---
    The search config must consist of two parts: strategy and search_space.
    """
    if not isinstance(search_config, dict):
        search_config = load_config(search_config)
    default_pass_config = search_config["default_pass_config"]  # the actual config for action search
    changed_seq_blocks = search_config["changed_seq_blocks"]
    config_list = search_config["config_list"]

    return default_pass_config, changed_seq_blocks,config_list

#################################################
#                search space                   #
#################################################

def search_space_set(pass_config,config_list,changed_seq_blocks):
    search_spaces = []
    for i in config_list:
        for seq_blcok in changed_seq_blocks:
            pass_config[seq_blcok]['config']['channel_multiplier'] = i
        search_spaces.append(copy.deepcopy(pass_config))
    return search_spaces

#################################################
#                 grid search                   #
#################################################

def gridsearch(
    model:torch.nn.Module,
    search_config: dict | PathLike,
    model_info,
    task: str,
    data_module,
    load_name: PathLike = None,
    load_type: str = None,
):
    ### initial set up
    pass_config, changed_seq_blocks,config_list = parse_search_config(search_config)
    search_spaces = search_space_set(pass_config,config_list,changed_seq_blocks)
    metric = MulticlassAccuracy(num_classes=5)
    num_batchs = 5
    recorded_accs = []
    data_module.prepare_data()
    data_module.setup()
    model = load_model(load_name=load_name, load_type=load_type, model=model)

    # input_generator = InputGenerator(
    # data_module=data_module,
    # model_info=model_info,
    # task=task,
    # which_dataloader="train",)

    # dummy_in = {"x": next(iter(data_module.train_dataloader()))[0]}
    # _ = model(**dummy_in)

    mg = MaseGraph(model=model)
    # import pdb
    # pdb.set_trace()

    for i, config in enumerate(search_spaces):
        import pdb
        pdb.set_trace()
        # if i== 6:
        #     import pdb
        #     pdb.set_trace()
        mg, _ = redefine_linear_transform_pass(
        graph=mg, pass_args={"config": config})
        j = 0

        # this is the inner loop, where we also call it as a runner.
        acc_avg, loss_avg = 0, 0
        accs, losses = [], []
        for inputs in data_module.train_dataloader():
            # import pdb
            # pdb.set_trace()
            xs, ys = inputs
            preds = mg.model(xs)
            loss = torch.nn.functional.cross_entropy(preds, ys)
            acc = metric(preds, ys)
            accs.append(acc)
            losses.append(loss)
            if j > num_batchs:
                break
            j += 1
        acc_avg = sum(accs) / len(accs)
        loss_avg = sum(losses) / len(losses)
        recorded_accs.append(acc_avg)

    print("recorded_accs",recorded_accs,"\n")