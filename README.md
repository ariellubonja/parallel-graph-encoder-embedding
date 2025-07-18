# Graph Encoder Embedding

Fast Python implementation of Graph Encoder Embedding from **"One-Hot Graph Encoder Embedding"** by C. Shen, Q. Wang, and C. E. Priebe (2022, arXiv:2109.13098).

For ultra-fast C++ implementation using Ligra, see [parallel-graph-encoder-embedding-ligra](https://github.com/ariellubonja/parallel-graph-encoder-embedding-ligra).

## Performance

- **100x speedup** over original Python implementation using Numba
- **1000x speedup** available with Ligra C++ implementation

## Quick Start

Basic usage with edge list graphs:
```python
from src.DataPreprocess import numba_graph_encoder_embed
import numpy as np

# Load your graph as edge list: [source, target, weight]
edges = np.loadtxt("your_graph.csv", delimiter=",")
labels = np.random.randint(0, 5, size=(n_nodes, 1))  # Your node labels
n_nodes = int(max(edges[:, :2].max(), edges[:, :2].max())) + 1

# Get embeddings
Z, W = numba_graph_encoder_embed(edges, labels, n_nodes)
```

## Command Line Usage

Run embedding on any graph file:
```bash
python src/Evaluation.py path/to/graph.csv --weighted --laplacian
```

The script generates random labels and saves embeddings to `embeddings.csv`.

## Input Formats

**Edge list** (recommended): CSV with columns `[source, target, weight]`
```
0,1,1.0
1,2,0.5
```

**Adjacency matrix**: Square matrix where `A[i,j]` is edge weight between nodes i and j.

## Key Functions

- `numba_graph_encoder_embed()` - Fast embedding computation
- `DataPreprocess()` - Handles input preprocessing and format conversion
- `Clustering()` - Node clustering using embeddings
- `GNN()` - Neural network classification on embeddings

## Options

**Graph preprocessing**:
- `Laplacian=True` - Use normalized Laplacian instead of adjacency matrix
- `DiagA=True` - Add self-loops (diagonal augmentation)
- `Correlation=True` - L2-normalize embedding rows

**Learning tasks**:
- Clustering: Unknown cluster assignments
- Semi-supervised: Some nodes have known labels
- Supervised: Train/test split with fully labeled data

## Examples

**Clustering** with unknown number of clusters:
```python
from src.Run import Run
from utils.create_test_case import Case

case = Case(n=1000).case_10_cluster()  # Generate test graph
Run(case, "c", Y=[2,3,4,5])  # Try 2-5 clusters
```

**Semi-supervised learning** with 95% unlabeled nodes:
```python
case = Case(n=1000).case_10()  # 95% nodes unlabeled
Run(case, "se", Learner=2)  # Neural network learning
```

## Dependencies

```bash
pip install numpy numba sklearn tensorflow
```

For maximum performance, use the [Ligra implementation](https://github.com/ariellubonja/parallel-graph-encoder-embedding-ligra) for large graphs (>100k nodes).
