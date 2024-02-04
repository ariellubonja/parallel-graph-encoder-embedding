import os
import numpy as np
from src import DataPreprocess
import timeit
import argparse


def setup_gee(graph_name):
    # Run this every time to not cache results
    # print("Loading", graph_name)

    G_edgelist = np.load(os.path.join(graphs_npy_path, graph_name))
    # G_edgelist = np.loadtxt("../../../Thesis-Graph-Data/twitch-SNAP-bidir-manually", delimiter=" ", dtype=np.int32)

    G_edgelist = G_edgelist[G_edgelist[:, 0].argsort()]  # Sort by first column

    # Add column of ones - weights
    G_edgelist = np.hstack((G_edgelist, np.ones((G_edgelist.shape[0], 1))))  # .astype(np.int32)
    # Make sure G_edgelist isn't restricted to int-s

    n = int(np.max(G_edgelist[:, 1]) + 1)  # Nr. vertices

    return G_edgelist, n


def run_gee(erdos_labels_path, graph_name, G_edgelist, n, numba):
    Y = np.load(os.path.join(erdos_labels_path, graph_name))  # For Ligra fairness - ligra cannot preload this

    # os.environ['NUMBA'] = numba   # This fails with Numba compiled

    if not numba:
        _ = DataPreprocess.graph_encoder_embed(G_edgelist, Y, n, Correlation=False, Laplacian=False)
    else:  # if numba == 'Parallel' or 'Serial' - deprecated. Will only experiment with basic GEE and Numba Serial
        _ = DataPreprocess.numba_graph_encoder_embed(G_edgelist, Y, n, Correlation=False, Laplacian=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test GEE Runtime results, with or without numba. File names should '
                                                 'match in NPY/ and Ys/')
    parser.add_argument('--machine_name', type=str, choices=['m5.12xlarge', 'm5.metal'], required=True,
                        help="Name of AWS machine we're currently running on")
    parser.add_argument('--numba', action='store_true', default=False,
                        help="Whether to use numba for GEE or not. If Numba used, Serial only")
    parser.add_argument('--graphs_base_dir', type=str, default='/home/ubuntu/prog/erdos-renyi-10-degree/', )
    parser.add_argument('--nr_experiments', type=int, default=7, )

    args = parser.parse_args()

    base_dir = args.graphs_base_dir

    graphs_npy_path = os.path.join(base_dir, "NPY")
    labels_path = os.path.join(base_dir, "Ys")

    # Get all .npy files in the directory
    graph_files = [f for f in os.listdir(graphs_npy_path) if f.endswith('.npy')]

    if args.machine_name == 'm5.12xlarge':
        result_file_name = "runtime_results/m5_12xlarge.txt"
    else:
        result_file_name = "runtime_results/m5_metal.txt"

    with open(result_file_name, "a") as result_file:
        if args.numba:
            result_file.write(f"\n\nGEE Stock\n\n")
        else:
            result_file.write(f"\n\nNumba Serial\n\n")

    # Loop over graphs in folder
    for graph_name in graph_files:
        with open(result_file_name, "a") as result_file:
            result_file.write(f"\n\n{graph_name}\n\n")

        print(f"\n\nRunning experiments for {graph_name}\n")

        for i in range(args.nr_experiments):
            # Setup GEE (outside of timing)
            G_edgelist, n = setup_gee(graph_name)

            # Time the run_gee function
            runtime = timeit.timeit(lambda: run_gee(labels_path, graph_name, G_edgelist, n, numba=args.numba),
                                    number=1)
            result_string = f"Running time {runtime}"

            # Print and write the result to runtime_results.txt
            print(result_string)
            with open(result_file_name, "a") as result_file:
                result_file.write(str(runtime) + '\n')
