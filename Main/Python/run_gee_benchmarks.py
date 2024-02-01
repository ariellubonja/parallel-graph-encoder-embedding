import os
import numpy as np
import DataPreprocess
import timeit
import argparse

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


def run_gee(erdos_labels_path, graph_name, G_edgelist, n, numba='Parallel'):
    Y = np.load(os.path.join(erdos_labels_path , graph_name)) # For Ligra fairness - ligra cannot preload this

    os.environ['NUMBA'] = numba

    if numba == 'None':
        _ = DataPreprocess.graph_encoder_embed(G_edgelist, Y, n, Correlation = False, Laplacian = False)
    else: # if numba == 'Parallel' or 'Serial'
        _ = DataPreprocess.numba_graph_encoder_embed(G_edgelist, Y, n, Correlation = False, Laplacian = False)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test GEE Runtime results, with or without numba')
    parser.add_argument('--machine_name', type=str, choices=['m5.12xlarge', 'm5.metal'],
                        help="Name of AWS machine we're currently running on")
    parser.add_argument('--numba', type=str, choices=['None', 'Parallel', 'Serial'], default='None',
                        help="Whether to use numba for GEE or not, and if so, Serial or Parallel")
    parser.add_argument('--graphs_base_dir', type=str, default='/home/ubuntu/prog/erdos-renyi-10-degree/',)
    parser.add_argument('--nr_experiments', type=int, default=7,)

    args = parser.parse_args()

    base_dir = args.graphs_base_dir

    erdos_10_degree_graphs_npy_path = os.path.join(base_dir, "NPY")
    erdos_labels_path = os.path.join(base_dir, "Ys")

    # Get all .npy files in the directory
    graph_files = [f for f in os.listdir(erdos_10_degree_graphs_npy_path) if f.endswith('.npy')]

    # Sort files by number of nodes (optional, for structured progression)
    graph_files.sort(key=lambda x: int(x.split('-')[0].split('_')[0]))

    # Loop over each graph file
    for graph_name in graph_files:

        if args.machine_name == 'm5.12xlarge':
            result_file_name = "runtime_results/m5_12xlarge.txt"
        else:  # if args.machine_name == 'm5.metal':
            result_file_name = "runtime_results/m5_metal.txt"

        with open(result_file_name, "a") as result_file:
            result_file.write(f"\n\n{graph_name}\n\n")

        print(f"\n\nRunning experiments for {graph_name}\n")

        for i in range(args.nr_experiments):
            # Setup GEE (outside of timing)
            G_edgelist, n = setup_gee(graph_name)

            # Time the run_gee function
            runtime = timeit.timeit(lambda: run_gee(erdos_labels_path, graph_name, G_edgelist, n), number=1)
            result_string = f"Experiment {i+1} for {graph_name}: {runtime} seconds"

            # Print and write the result to runtime_results.txt
            print(result_string)
            with open(result_file_name, "a") as result_file:
                result_file.write(str(runtime)+'\n')
