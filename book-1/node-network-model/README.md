## Overview

### 1. **BasicGraphConvolutionLayer**
A custom implementation of a graph convolution layer that performs the following operations:
- Applies linear transformations to node features.
- Aggregates messages from neighboring nodes using the adjacency matrix.
- Combines the transformed features with bias to produce the output.

### 2. **NodeNetwork**
A simple GNN model that stacks two `BasicGraphConvolutionLayer` layers followed by fully connected layers. This model is designed for graph classification tasks.

### 3. **global_sum_pool(X, batch_mat)**
Implements global sum pooling, which aggregates node features into graph-level representations. It supports both single graph and batch processing scenarios.

### 4. **get_batch_tensor(graph_sizes)**
Generates a batch matrix to help identify node membership within different graphs in a batch. Essential for efficiently processing multiple graphs together.

### 5. **collate_graphs(batch)**
Collates multiple graphs into a batch-level representation by constructing block diagonal adjacency matrices, concatenating node features, and assembling batch matrices.
