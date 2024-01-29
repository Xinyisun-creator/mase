import logging

logger = logging.getLogger(__name__)


def flop_calculator_pass(graph,pass_args: dict,relu = True):
    ##Target 1: get input and output size from mase graph 
    for node in graph.fx_graph.nodes:
        ## Read parameters
        if node.meta["mase"].parameters["common"]["mase_op"] == "linear":
            # continue
            input_shape = node.meta["mase"].parameters["common"]["args"]["data_in_0"]["shape"]
            output_shape = node.meta["mase"].parameters["common"]["results"]["data_out_0"]["shape"]
            bias_shape = node.meta["mase"].parameters["common"]["args"]["bias"]["shape"]
            batch_size = input_shape[0]
            flops_linear = batch_size * ((2*input_shape[1]-1) * output_shape[1]+bias_shape[0])
            print("flops_linear = ",flops_linear)
        elif node.meta["mase"].parameters["common"]["mase_op"] == "relu":
            if relu == True:
                ## If the comparison between inputs and 0 could be considered as Float Operation
                input_shape = node.meta["mase"].parameters["common"]["args"]["data_in_0"]["shape"]
                flops_relu = input_shape[1]
                print("flops_relu = ",flops_relu)
            else:
                ## else: ignore
                continue
        elif node.meta["mase"].parameters["common"]["mase_op"] == "batch_norm1d":
            input_shape = node.meta["mase"].parameters["common"]["args"]["data_in_0"]["shape"]
            flops_norm1d = input_shape[0] * input_shape[1] * 4
            print("flops_norm1d",flops_norm1d)
        
    ## linear layer
    ## activation layer
    ## BatchNorm 1d layer
