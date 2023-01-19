from sklearn import metrics
from sklearn.metrics import adjusted_rand_score
# from tensorflow.keras.utils import to_categorical
import numpy as np
from utils.create_test_case import Case
from DataPreprocess import graph_encoder_embed

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



# def generateYlabelsSBM():
    


if __name__ == '__main__':
    
    # https://www.tutorialspoint.com/python/python_command_line_arguments.htm
    import argparse

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-fn", "--filename", help="Input File name. .npy Edgelist required")
    
    args = argParser.parse_args()

    print("args.name=%s" % args.filename)
    
    # A = np.ones((5,5))
    # A[0,4] = 0
    # A[4,0] = 0
    # np.fill_diagonal(A, 0)
    #
    # Y = np.array([[0,0,0,1,1]]).reshape((5,1))
    #
    # # print(A)
    # # print(Y)
    #
    # Encoder_case5 = Encoder_case(A,Y,5)
    #
    # from Main.Python.DataPreprocess import DataPreprocess
    #
    # Dataset = DataPreprocess(Encoder_case5, Laplacian = False, DiagA = False)
    # # print(Dataset.X)
    # # print(Dataset.Y)
    # # print(Dataset.n)

    # case = Case(n)
    # case_10 = case.case_10() # This is O(n^2)
    # case_10.summary()

    #
    # print("Running graph_encoder_embed()")
    #
    # Z, W = graph_encoder_embed(Dataset.X[0], Dataset.Y, Dataset.n, Correlation = False)
    # print("Z:\n", Z)
    # # print(W)


    # print("Loading LiveJournal graph - 1GB")
    #
    # G_edgelist = np.load("../../../Thesis-Graph-Data/NPY-graphs/liveJournalWgh.npy")
    #
    # # G_edgelist = np.load("../../../Thesis-Graph-Data/NPY-graphs/liveJournal.npy")
    # # Add column of ones - weights
    # # G_edgelist = np.hstack((G_edgelist, np.ones((G_edgelist.shape[0], 1))))
    #
    # n = int(np.max(G_edgelist[:,1]) + 1) # Nr. vertices
    #
    # Y = np.load("../../../Thesis-Graph-Data/Ys/liveJournal-Y50.npy")

    # print("Loading Orkut graph - 1.8GB")
    #
    # G_edgelist = np.load("../../../Thesis-Graph-Data/NPY-graphs/orkutWgh.npy")
    #
    # # Add column of ones - weights
    # # G_edgelist = np.hstack((G_edgelist, np.ones((G_edgelist.shape[0], 1))))
    #
    # n = int(np.max(G_edgelist[:, 1]) + 1)  # Nr. vertices
    #
    # Y = np.load("../../../Thesis-Graph-Data/Ys/orkut-Y50.npy")


    print("Loading Orkut-User2Group graph - 5.1GB")

    # G_edgelist = np.load("../../../Thesis-Graph-Data/NPY-graphs/orkut-groupsUnw.npy")
    # # Add column of ones - weights
    # G_edgelist = np.hstack((G_edgelist, np.ones((G_edgelist.shape[0], 1))))

    G_edgelist = np.load("../../../Thesis-Graph-Data/NPY-graphs/orkut-groupsWgh.npy")


    n = int(np.max(G_edgelist[:, 1])) + 1  # Nr. vertices +1 bcs we index G[v] - 0-indexed

    Y = np.load("../../../Thesis-Graph-Data/Ys/orkut-groups-Y40.npy")

    # print("Loading Twitch graph")
    #
    # G_edgelist = np.load("../../../Thesis-Graph-Data/NPY-graphs/twitchWgh.npy")
    # # G_edgelist = np.loadtxt("../../../Thesis-Graph-Data/twitch-SNAP-bidir-manually", delimiter=" ", dtype=np.int32)
    #
    # G_edgelist = G_edgelist[G_edgelist[:, 0].argsort()] # Sort by first column
    #
    # # Add column of ones - weights
    # # G_edgelist = np.hstack((G_edgelist, np.ones((G_edgelist.shape[0], 1))))#.astype(np.int32)
    # # Make sure G_edgelist isn't restricted to int-s
    #
    # n = int(np.max(G_edgelist[:,1]) + 1) # Nr. vertices
    #
    # Y = np.load("../../../Thesis-Graph-Data/Ys/twitch-Y20.npy")


    # print("Loading Pokec graph - 400MB")
    #
    # G_edgelist = np.load("../../../Thesis-Graph-Data/NPY-graphs/pokecWgh.npy")
    #
    # # Add column of ones - weights
    # # G_edgelist = np.hstack((G_edgelist, np.ones((G_edgelist.shape[0], 1))))
    #
    # n = int(np.max(G_edgelist[:,1]) + 1) # Nr. vertices
    #
    # # Load Y from file
    # Y = np.load("../../../Thesis-Graph-Data/Ys/pokec-Y50.npy")


    # print("Loading Friendster graph - 31GB")
    #
    # G_edgelist = np.load("../../../Thesis-Graph-Data/NPY-graphs/friendsterWgh.npy")
    #
    # # Add column of ones - weights
    # # G_edgelist = np.hstack((G_edgelist, np.ones((G_edgelist.shape[0], 1))))
    #
    # n = int(np.max(G_edgelist[:,1]) + 1) # Nr. vertices
    #
    # # Load Y from file
    # Y = np.load("../../../Thesis-Graph-Data/Ys/friendster-Y50.npy")


    laplacian=False
    print("Running GraphEncoderEmbed( laplacian =", laplacian, ")")

    Z, _ = graph_encoder_embed(G_edgelist.astype(np.float64), Y, n, Correlation = False, Laplacian = laplacian)

    print("Saving Embedding to file")
    # np.savetxt("Z_CorrectResults.csv", Z, fmt="%f")
    np.save("Z_CorrectResults.npy", Z)

