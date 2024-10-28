# Fluid Data to Graph Data Conversion

This project provides a Python script that converts fluid simulation data into graph format, compatible with the PyTorch Geometric library. The code is designed to process data from the x-component of the velocity vector, computed for various Reynolds numbers using the Finite Element Method (FEM) solutions to the Navier-Stokes equation. 

## Overview

The code performs the following steps:
1. Loads x-component velocity data for 4000 time points across different Reynolds numbers.
2. Reshapes and formats the data to align with the PyTorch Geometric data structure.
3. Converts the fluid data into a graph format and saves it in `vel_x_graph.pt` for easy loading and processing in graph-based machine learning models.

## Data Structure

Each generated graph object has the following attributes:

- **data.x**: Node feature matrix with shape `[num_nodes, num_node_features]`.
- **data.edge_index**: Graph connectivity in COO format with shape `[2, num_edges]`, type `torch.long`.
- **data.y**: Reynolds number associated with each data point, with shape `[1]` and type `torch.long`.
- **data.pos**: Node position matrix with shape `[num_nodes, num_dimensions]`.

## Code Walkthrough

### Libraries and Data Loading

The following libraries are used:

- **numpy**: For numerical operations and data manipulation.
- **torch**: Core library for tensor operations and saving/loading models.
- **torch_geometric**: For constructing graph-based data structures compatible with PyTorch.

### Data Loading and Configuration

The code loads velocity data from `.npy` files and specifies Reynolds numbers in two ranges: `20-40` and `100-120`. This allows processing of datasets with different Reynolds numbers based on the configuration.

### Graph Conversion Function

The main function, `load_flow_graph`, converts the fluid data into graph format by:
1. Loading edges from `refined_edges` and positions from `position.npy`.
2. Iterating over time points, converting each data snapshot into a graph.
3. Appending each graph to `data_list`.

Finally, it saves the processed graph data list to `vel_x_graph.pt`.

## Usage

1. **Prepare the dataset**: Ensure that `vel_x.npy`, `refined_edges`, and `position.npy` are in the `./datasets` directory.
2. **Run the code**: Execute the script to generate the graph data.
3. **Load the graph data**: The processed graph data is saved to `vel_x_graph.pt` and can be loaded using `torch.load('vel_x_graph.pt')`.

### Example Code

```python
import torch

# Load the graph data
flow_list = torch.load('vel_x_graph.pt')

# Example access
print(flow_list[0])  # Print the first graph object
