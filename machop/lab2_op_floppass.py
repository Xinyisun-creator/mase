import logging

logger = logging.getLogger(__name__)


def flop_calculator_pass(graph,pass_args: dict,relu = True):
    ##Initiate the FLOP lists
    flops_linear = []
    flops_relu = []
    flops_norm1d = []
    for node in graph.fx_graph.nodes:
        ## Read parameters
        if node.meta["mase"].parameters["common"]["mase_op"] == "linear":
            # continue
            input_shape = node.meta["mase"].parameters["common"]["args"]["data_in_0"]["shape"]
            output_shape = node.meta["mase"].parameters["common"]["results"]["data_out_0"]["shape"]
            bias_shape = node.meta["mase"].parameters["common"]["args"]["bias"]["shape"]
            batch_size = input_shape[0]
            flops_linear.append(batch_size * ((2*input_shape[1]-1) * output_shape[1]+bias_shape[0]))
        elif node.meta["mase"].parameters["common"]["mase_op"] == "relu":
            if relu == True:
                ## If the comparison between inputs and 0 could be considered as Float Operation
                input_shape = node.meta["mase"].parameters["common"]["args"]["data_in_0"]["shape"]
                flops_relu.append(input_shape[1])
            else:
                ## else: ignore
                continue
        elif node.meta["mase"].parameters["common"]["mase_op"] == "batch_norm1d":
            input_shape = node.meta["mase"].parameters["common"]["args"]["data_in_0"]["shape"]
            flops_norm1d.append(input_shape[0] * input_shape[1] * 4)
    return flops_linear,flops_relu,flops_norm1d

def modlesize_calculator_pass(graph,pass_args: dict):
    ##Initiate the FLOP lists
    linear_size = []
    norm1d_size = []
    relu_size = []
    for node in graph.fx_graph.nodes:
        ## Read parameters
        if node.meta["mase"].parameters["common"]["mase_op"] == "linear":
            # continue
            input_shape = node.meta["mase"].parameters["common"]["args"]["data_in_0"]["shape"]
            output_shape = node.meta["mase"].parameters["common"]["results"]["data_out_0"]["shape"]
            bias_shape = node.meta["mase"].parameters["common"]["args"]["bias"]["shape"]
            linear_size.append((input_shape[1] * output_shape[1]+bias_shape[0]))
        elif node.meta["mase"].parameters["common"]["mase_op"] == "relu":
            input_shape = node.meta["mase"].parameters["common"]["args"]["data_in_0"]["shape"]
            relu_size.append(input_shape[1])
        elif node.meta["mase"].parameters["common"]["mase_op"] == "batch_norm1d":
            input_shape = node.meta["mase"].parameters["common"]["args"]["data_in_0"]["shape"]
            norm1d_size.append(input_shape[1] * 2)
    return linear_size,norm1d_size,relu_size