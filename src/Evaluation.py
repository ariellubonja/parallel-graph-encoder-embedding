from sklearn import metrics
from sklearn.metrics import adjusted_rand_score
# from tensorflow.keras.utils import to_categorical
import numpy as np
from utils.create_test_case import Case
from DataPreprocess import numba_graph_encoder_embed
import argparse

class Evaluation:
    def GNN_supervise_test(self, gnn, z_test, y_test):
        """
          test the accuracy for GNN_direct
        """
        y_test_one_hot = to_categorical(y_test)
        # set verbose to 0 to silent the output
        test_loss, test_acc = gnn.model.evaluate(z_test,  y_test_one_hot, verbose=0)

        return test_acc

    def LDA_supervise_test(self, lda, z_test, y_test):
        """
          test the accuracy for LDA_learner
        """
        test_acc = lda.model.score(z_test, y_test)

        return test_acc

    def GNN_semi_supervised_learn_test(self,Y_result, Y_original):
        """
          test accuracy for semi-supervised learning
        """
        test_acc = metrics.accuracy_score(Y_result, Y_original)

        return test_acc

    def GNN_semi_supervised_not_learn_test(self, gnn, Dataset, case):
        """
          test accuracy for semi-supervised learning
        """

        ind_unlabel = Dataset.ind_unlabel
        z_unlabel =  Dataset.z_unlabel
        y_unlabel_ori = case.Y_ori[ind_unlabel, 0]
        y_unlabel_ori_one_hot = to_categorical(y_unlabel_ori)
        test_loss, test_acc = gnn.model.evaluate(z_unlabel, y_unlabel_ori_one_hot, verbose=0)

        return test_acc


    def clustering_test(self, Y_result, Y_original):
        """
          test accuracy for semi-supervised learning
        """
        ari = adjusted_rand_score(Y_result, Y_original.reshape(-1,))

        return ari


# Code to test functions
class Encoder_case:
    def __init__(self, A,Y,n):
        Encoder_case.X = A
        Encoder_case.Y = Y
        Encoder_case.n = n



def loadGraph(filepath, weighted, randomY=True, yPath=None):
    print("Loading " + filepath)

    if 'npy' in filepath:
        G_edgelist = np.load(filepath)
    else:
        G_edgelist = np.loadtxt(filepath, delimiter=",")
    
    if not weighted:
	# Add column of ones - weights
    	G_edgelist = np.hstack((G_edgelist, np.ones((G_edgelist.shape[0], 1))))
    
    n = int(np.max(G_edgelist[:,1]) + 1) # Nr. vertices
    
    if randomY: # Efficient Vector operations
        n_nodes = int(max(np.max(G_edgelist[:,1]), np.max(G_edgelist[:,0])))
    	# create some ground truth
        labels = np.random.randint(low=0, high=50, size=(n_nodes,1))
    	# Remove 90% of them by assigning -1
        bern = np.random.binomial(1, 0.1, size=(n_nodes,1))
        Y = labels * bern
        # Subtract 1
        Y -= 1
    
        # Make sure to assign 1 value to be maxvalue i.e. 49 (50-1) in Y - GEE uses max(Y) to set embedding size!
        Y[0]=49
    else:
        Y = np.load(yPath)
    
    return G_edgelist, Y, n

    


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script to load a graph.')

    # Add the command line arguments
    parser.add_argument('filepath', type=str, help='Path to the graph file')
    parser.add_argument('--weighted', action='store_true', help='Specify if the graph is weighted', default=False)
    parser.add_argument('--randomY', action='store_true', help='Specify if random Y values should be used', default=False)
    parser.add_argument('--yPath', type=str, help='Path to the Y labels (Required if --randomY=False)')

    parser.add_argument('--laplacian', action='store_true', default=False, help='Use Graph Laplacian or Adjacency matrix?')
    parser.add_argument('--correlation', action='store_true', default=False)
    
    # Parse the command line arguments
    args = parser.parse_args()
    
    G_edgelist, Y, n = loadGraph(args.filepath, args.weighted, args.randomY, args.yPath)


    print("Running GraphEncoderEmbed( laplacian =", args.laplacian, ")")

    Z, W = numba_graph_encoder_embed(G_edgelist, Y, n, Correlation = False, Laplacian = args.laplacian)

    print("Saving Embedding to embeddings.csv")
    np.savetxt("embeddings.csv", Z, fmt="%f")
    # np.save("Z_CorrectResults.npy", Z)

