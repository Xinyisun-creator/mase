import json
import matplotlib.pyplot as plt

# Load the data from the JSON file
with open('./Pytorch_Quantization_Experiment_result/results_name_20240327153441.json', 'r') as f:
    data = json.load(f)

# Load the data from the first JSON file
with open('./Pytorch_Quantization_Experiment_result/results_name_20240327145720.json', 'r') as f:
    data1 = json.load(f)

# Load the data from the second JSON file
with open('./Pytorch_Quantization_Experiment_result/results_name_20240327153441.json', 'r') as f:
    data2 = json.load(f)

# Extract the data from the first file
widths1 = data1['widths'][1:] # Exclude the first element "Original Graph"
tensorRT_accuracy1 = data1['TensorRT Accuracy'][1:]
tensorRT_latency1 = data1['TensorRT Latency'][1:]
run_model_accuracy1 = data1['Run Model Accuracy'][1:]  # Exclude the first element
run_model_accuracy1 = [i * 100 for i in run_model_accuracy1]
run_model_latency1 = data1['Run Model Latency'][1:]  # Exclude the first element

# Extract the data from the second file
widths2 = data2['widths'][1:] # Exclude the first element "Original Graph"
tensorRT_accuracy2 = data2['TensorRT Accuracy'][1:]
tensorRT_latency2 = data2['TensorRT Latency'][1:]
run_model_accuracy2 = data2['Run Model Accuracy'][1:]  # Exclude the first element
run_model_accuracy2 = [i * 100 for i in run_model_accuracy2]
run_model_latency2 = data2['Run Model Latency'][1:]  # Exclude the first element

# Extract the accuracy and latency of the original graph
original_graph_run_model_accuracy = data['Run Model Accuracy'][0] * 100  # Multiply by 100 to get percentage
original_graph_tensorRT_accuracy = data['TensorRT Accuracy'][0]
original_graph_run_model_latency = data['Run Model Latency'][0]
original_graph_tensorRT_latency = data['TensorRT Latency'][0]

# Create a new widths list that includes 'Original Graph'
new_widths = ['Original Graph'] + widths1

# Create a new widths list with integers
new_widths_int = list(range(len(new_widths)))

# Plot the accuracy
plt.figure(figsize=(7, 5))
plt.plot(new_widths_int[1:], tensorRT_accuracy1, label='TensorRT Accuracy')
plt.plot(new_widths_int[1:], run_model_accuracy1, label='Run Model Accuracy')
plt.plot(new_widths_int[1:], tensorRT_accuracy2, label='TensorRT Accuracy with True Quant')
plt.plot(new_widths_int[1:], run_model_accuracy2, label='Run Model Accuracy with True Quant')
plt.scatter(new_widths_int[0], original_graph_tensorRT_accuracy, color='blue', label='Original Graph TensorRT Accuracy')
plt.scatter(new_widths_int[0], original_graph_run_model_accuracy, color='orange', label='Original Graph Run Model Accuracy')
plt.xticks(new_widths_int, new_widths)  # Set the x-axis labels to the new widths list
plt.title("Accuracy when Classifier.0(Linear) is Quantized among Different Widths")
plt.grid(True)
plt.xlabel('Widths')
plt.ylabel('Accuracy(%)')
plt.legend()
plt.show()

# Create a subplot for TensorRT Latency
plt.figure(figsize=(7, 5))
plt.plot(new_widths_int[1:], tensorRT_latency1, label='TensorRT Latency')
plt.plot(new_widths_int[1:], tensorRT_latency2, label='TensorRT Latency with True Quant')
plt.scatter(new_widths_int[0], original_graph_tensorRT_latency, color='blue', label='Original Graph TensorRT Latency')
plt.xlabel('Widths')
plt.ylabel('Latency(ms)')
plt.xticks(new_widths_int, new_widths)  # Set the x-axis labels to the new widths list
plt.title("TensorRT Latency when Classifier.0(Linear) is Quantized among Different Widths")
plt.legend()
plt.grid(True)
plt.show()

# Create a subplot for Run Model Latency
plt.figure(figsize=(7, 5))
plt.plot(new_widths_int[1:], run_model_latency1, label='TensorRT Latency')
plt.plot(new_widths_int[1:], run_model_latency2, label='TensorRT Latency with True Quant')
plt.scatter(new_widths_int[0], original_graph_run_model_latency, color='blue', label='Original Graph TensorRT Latency')
# plt.plot(new_widths_int[1:], run_model_latency, label='Run Model Latency')
plt.xlabel('Widths')
plt.ylabel('Latency(ms)')
plt.xticks(new_widths_int, new_widths)  # Set the x-axis labels to the new widths list
plt.title("The Run-Model Latency when Classifier.0(Linear) is Quantized among Different Widths")
plt.legend()
plt.grid(True)

# Show the plots
plt.show()