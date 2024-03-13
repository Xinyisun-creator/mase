# This is the search space for mixed-precision post-training-quantization quantization search on mase graph.
from copy import deepcopy
from torch import nn
from ..base import SearchSpaceBase
from .....passes.graph.transforms.quantize import (
    QUANTIZEABLE_OP,
    quantize_transform_pass,
)
from .....ir.graph.mase_graph import MaseGraph
from .....passes.graph import (
    init_metadata_analysis_pass,
    add_common_metadata_analysis_pass,
)
from .....passes.graph.utils import get_mase_op, get_mase_type
from ..utils import flatten_dict, unflatten_dict
from collections import defaultdict
import copy
from chop.passes.graph.utils import get_parent_name

from chop.passes.graph.analysis import (
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    report_graph_analysis_pass,
)




DEFAULT_QUANTIZATION_CONFIG = {
    "config": {
        "name": "integer",
        "bypass": True,
        "bias_frac_width": 5,
        "bias_width": 8,
        "data_in_frac_width": 5,
        "data_in_width": 8,
        "weight_frac_width": 3,
        "weight_width": 8,
    }
}

DEFAULT_BLOCKSIZE_CONFIG = {
"by": "name",
"default": {"config": {"name": None}},
"seq_blocks_2": {
    "config": {
        "name": "output_only",
        # weight
        "channel_multiplier": 2,
        }
},

"seq_blocks_4": {
    "config": {
        "name": "both",
        # weight
        "channel_multiplier": 2,
        }
},

"seq_blocks_6": {
    "config": {
        "name": "input_only",
        # weight
        "channel_multiplier": 2,
        }
}
}

def instantiate_linear(in_features, out_features, bias):
    if bias is not None:
        bias = True
    return nn.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=bias)

def search_space_set(pass_config,config_list,changed_seq_blocks):
    search_spaces = []
    for i in config_list:
        for seq_blcok in changed_seq_blocks:
            pass_config[seq_blcok]['config']['channel_multiplier'] = i
        search_spaces.append(copy.deepcopy(pass_config))
    return search_spaces

def in_out_size_check(input_features,output_features,ori_module,i):
    check = False
    if i <= 1:
        input_features = ori_module.in_features
        out_features = ori_module.out_features
        if input_features == out_features:
            check = True
    else:
        if input_features == output_features:
            check = True
    return check


def redefine_linear_transform_pass(graph, pass_args=None):
    main_config = pass_args
    i = 0
    for node in graph.fx_graph.nodes:
        # if node name is not matched, it won't be tracked
        if node.name == 'x' or node.name == 'output' or node.name == 'seq_blocks_0':
            continue
        elif node.name == 'seq_blocks_1':
            channel_multiplier = main_config.get(node.name)['config']["channel_multiplier"]
        else:
            ## sanity check with input size and output size:
            config = main_config.get(node.name)['config']
            name = config.get("name_redefine", None)
            if name is not None:
                i += 1
                if i == 1:
                    out_features=None
                ori_module = graph.modules[node.target]
                bias = ori_module.bias
                if name == "output_only":
                    in_features = ori_module.in_features
                    out_features = ori_module.out_features
                    out_features = out_features * channel_multiplier
                elif name == "both":
                    in_features = ori_module.in_features
                    in_features = in_features * channel_multiplier
                    out_features = ori_module.out_features
                    out_features = out_features * channel_multiplier
                elif name == "input_only":
                    in_features = ori_module.in_features
                    in_features = in_features * channel_multiplier
                    out_features = ori_module.out_features
                # import pdb
                # pdb.set_trace()
                new_module = instantiate_linear(in_features, out_features, bias)
                parent_name, name = get_parent_name(node.target)
                setattr(graph.modules[parent_name], name, new_module)
    return graph, {}


class LAB4_GraphSearchSpaceMixedPrecisionPTQ(SearchSpaceBase):
    """
    Post-Training quantization search space for mase graph.
    """

    def _post_init_setup(self):
        # print(self.model)
        # import pdb
        # pdb.set_trace()
        self.model.to("cpu")  # save this copy of the model to cpu
        self.mg = None
        self._node_info = None
        self.default_config = DEFAULT_QUANTIZATION_CONFIG
        self.default_pass_config = DEFAULT_BLOCKSIZE_CONFIG

        # quantize the model by type or name
        assert (
            "by" in self.config["setup"]
        ), "Must specify entry `by` (config['setup']['by] = 'name' or 'type')"

    def rebuild_model(self, sampled_config, is_eval_mode: bool = True):
        # set train/eval mode before creating mase graph

        self.model.to(self.accelerator)
        if is_eval_mode:
            self.model.eval()
        else:
            self.model.train()

        if self.mg is None:
            assert self.model_info.is_fx_traceable, "Model must be fx traceable"
            mg = MaseGraph(self.model)
            mg, _ = init_metadata_analysis_pass(mg, None)
            mg, _ = add_common_metadata_analysis_pass(
                mg, {"dummy_in": self.dummy_input, "force_device_meta": False}
            )
            self.mg = mg
        if sampled_config is not None:
            mg, _ = quantize_transform_pass(self.mg, sampled_config)
            mg, _ = redefine_linear_transform_pass(self.mg, sampled_config)

        mg.model.to(self.accelerator)
        return mg

    def _build_node_info(self):
        """
        Build a mapping from node name to mase_type and mase_op.
        """

    def build_search_space(self):
        """
        Build the search space for the mase graph (only quantizeable ops)
        """
        # Build a mapping from node name to mase_type and mase_op.
        # linear_config_1 = [1,2,4,6,8,10]

        mase_graph = self.rebuild_model(sampled_config=None, is_eval_mode=True)
        node_info = {}
        for node in mase_graph.fx_graph.nodes:
            node_info[node.name] = {
                "mase_type": get_mase_type(node),
                "mase_op": get_mase_op(node),
            }

        # Build the search space
        choices = {}
        seed = self.config["seed"]

        match self.config["setup"]["by"]:
            case "name":
                # iterate through all the quantizeable nodes in the graph
                # if the node_name is in the seed, use the node seed search space
                # else use the default search space for the node
                for n_name, n_info in node_info.items():
                    if n_info["mase_op"] in QUANTIZEABLE_OP:
                        if n_name in seed:
                            choices[n_name] = deepcopy(seed[n_name])
                        else:
                            choices[n_name] = deepcopy(seed["default"])
                            

            case _:
                raise ValueError(
                    f"Unknown quantization by: {self.config['setup']['by']}"
                )
        # flatten the choices and choice_lengths
        flatten_dict(choices, flattened=self.choices_flattened)
        self.choice_lengths_flattened = {
            k: len(v) for k, v in self.choices_flattened.items()
        }


    def flattened_indexes_to_config(self, indexes: dict[str, int]):
        """
        Convert sampled flattened indexes to a nested config which will be passed to `rebuild_model`.

        ---
        For example:
        ```python
        >>> indexes = {
            "conv1/config/name": 0,
            "conv1/config/bias_frac_width": 1,
            "conv1/config/bias_width": 3,
            ...
        }
        >>> choices_flattened = {
            "conv1/config/name": ["integer", ],
            "conv1/config/bias_frac_width": [5, 6, 7, 8],
            "conv1/config/bias_width": [3, 4, 5, 6, 7, 8],
            ...
        }
        >>> flattened_indexes_to_config(indexes)
        {
            "conv1": {
                "config": {
                    "name": "integer",
                    "bias_frac_width": 6,
                    "bias_width": 6,
                    ...
                }
            }
        }
        """
        flattened_config = {}
        for k, v in indexes.items():
            flattened_config[k] = self.choices_flattened[k][v]

        config = unflatten_dict(flattened_config)
        config["default"] = self.default_config
        config["by"] = self.config["setup"]["by"]
        return config
