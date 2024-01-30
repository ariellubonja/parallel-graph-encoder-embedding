import os
import numpy as np
import DataPreprocess
import timeit

base_dir = "/home/ubuntu/prog/erdos-new"

erdos_10_degree_graphs_npy_path = os.path.join(base_dir, "NPY")
erdos_labels_path = os.path.join(base_dir, "Ys")

def setup_gee(graph_name):
    # Run this every time to not cache results
    # print("Loading", graph_name)

    G_edgelist = np.load(os.path.join(erdos_10_degree_graphs_npy_path , graph_name))
    # G_edgelist = np.loadtxt("../../../Thesis-Graph-Data/twitch-SNAP-bidir-manually", delimiter=" ", dtype=np.int32)
    
    G_edgelist = G_edgelist[G_edgelist[:, 0].argsort()] # Sort by first column
    
    # Add column of ones - weights
    G_edgelist = np.hstack((G_edgelist, np.ones((G_edgelist.shape[0], 1))))#.astype(np.int32)
    # Make sure G_edgelist isn't restricted to int-s
    
    n = int(np.max(G_edgelist[:,1]) + 1) # Nr. vertices

    return G_edgelist, n


def run_gee(erdos_labels_path , graph_name, G_edgelist, n):
    Y = np.load(os.path.join(erdos_labels_path , graph_name)) # For Ligra fairness - ligra cannot preload this
    _ = DataPreprocess.graph_encoder_embed(G_edgelist, Y, n, Correlation = False, Laplacian = False)



# Get all .npy files in the directory
graph_files = [f for f in os.listdir(erdos_10_degree_graphs_npy_path) if f.endswith('.npy')]

# Sort files by number of nodes (optional, for structured progression)
graph_files.sort(key=lambda x: int(x.split('-')[0].split('_')[0]))

# Loop over each graph file
for graph_name in graph_files:
    with open("runtime_results.txt", "a") as result_file:
        result_file.write(f"\n\n{graph_name}\n\n")

    print(f"\n\nRunning experiments for {graph_name}\n")

    for i in range(7):
        # Setup GEE (outside of timing)
        G_edgelist, n = setup_gee(graph_name)

        # Time the run_gee function
        runtime = timeit.timeit(lambda: run_gee(erdos_labels_path, graph_name, G_edgelist, n), number=1)
        result_string = f"Experiment {i+1} for {graph_name}: {runtime} seconds"

        # Print and write the result to runtime_results.txt
        print(result_string)
        with open("runtime_results.txt", "a") as result_file:
            result_file.write(str(runtime)+'\n')
