import unittest

import numpy as np


class TestAdjacency(unittest.TestCase):
    files_dir = "../../../Experiments/Compare-Results/Adjacency/"

    def test_Twitch(self):
        graph_path = "Twitch/"
        Z_Correct = np.load(self.files_dir + graph_path + "Z_CorrectResults.npy")
        Z_to_check = np.loadtxt(self.files_dir + graph_path + "Z_to_check.csv")

        comparison = np.isclose(Z_Correct, Z_to_check, atol=1e-06) # Atol for ligra - csv loses precision

        self.assertTrue(np.all(comparison))

    def test_Pokec(self):
        graph_path = "Pokec/"
        Z_Correct = np.load(self.files_dir + graph_path + "Z_CorrectResults.npy")
        Z_to_check = np.loadtxt(self.files_dir + graph_path + "Z_to_check.csv")

        comparison = np.isclose(Z_Correct, Z_to_check, atol=1e-06) # Abs tolerance for ligra - csv loses precision

        self.assertTrue(np.all(comparison))

    def test_LiveJournal(self):
        graph_path = "LiveJournal/"
        Z_Correct = np.load(self.files_dir + graph_path + "Z_CorrectResults.npy")
        Z_to_check = np.loadtxt(self.files_dir + graph_path + "Z_to_check.csv")

        comparison = np.isclose(Z_Correct, Z_to_check, atol=1e-06)

        self.assertTrue(np.all(comparison))

    def test_Orkut(self):
        graph_path = "Orkut/"
        Z_Correct = np.load(self.files_dir + graph_path + "Z_CorrectResults.npy")
        Z_to_check = np.loadtxt(self.files_dir + graph_path + "Z_to_check.csv")

        comparison = np.isclose(Z_Correct, Z_to_check, atol=1e-06)

        self.assertTrue(np.all(comparison))


    def test_Orkut_Groups(self):
        graph_path = "Orkut-groups/"
        Z_Correct = np.load(self.files_dir + graph_path + "Z_CorrectResults.npy")
        Z_to_check = np.load(self.files_dir + graph_path + "Z_to_check.npy")

        comparison = np.isclose(Z_Correct, Z_to_check, atol=1e-06)

        self.assertTrue(np.all(comparison))


    def test_Friendster(self):
        graph_path = "Friendster/"
        Z_Correct = np.load(self.files_dir + graph_path + "Z_CorrectResults.npy")
        Z_to_check = np.load(self.files_dir + graph_path + "Z_to_check.npy")

        comparison = np.isclose(Z_Correct, Z_to_check, atol=1e-05)

        self.assertTrue(np.all(comparison))


class TestLaplacian(unittest.TestCase):
    files_dir = "../../../Experiments/Compare-Results/Laplacian/"

    def test_Twitch(self):
        graph_path = "Twitch/"
        Z_Correct = np.load(self.files_dir + graph_path + "Z_CorrectResults.npy")
        Z_to_check = np.loadtxt(self.files_dir + graph_path + "Z_to_check.csv")

        comparison = np.isclose(Z_Correct, Z_to_check, atol=1e-06)  # Atol for ligra - csv loses precision

        self.assertTrue(np.all(comparison))

    def test_Pokec(self):
        graph_path = "Pokec/"
        Z_Correct = np.load(self.files_dir + graph_path + "Z_CorrectResults.npy")
        Z_to_check = np.loadtxt(self.files_dir + graph_path + "Z_to_check.csv")

        comparison = np.isclose(Z_Correct, Z_to_check, atol=1e-06)  # Abs tolerance for ligra - csv loses precision

        self.assertTrue(np.all(comparison))

    def test_LiveJournal(self):
        graph_path = "LiveJournal/"
        Z_Correct = np.load(self.files_dir + graph_path + "Z_CorrectResults.npy")
        Z_to_check = np.loadtxt(self.files_dir + graph_path + "Z_to_check.csv")

        comparison = np.isclose(Z_Correct, Z_to_check, atol=1e-06)

        self.assertTrue(np.all(comparison))

    def test_Orkut(self):
        graph_path = "Orkut/"
        Z_Correct = np.load(self.files_dir + graph_path + "Z_CorrectResults.npy")
        Z_to_check = np.loadtxt(self.files_dir + graph_path + "Z_to_check.csv")

        comparison = np.isclose(Z_Correct, Z_to_check, atol=1e-06)

        self.assertTrue(np.all(comparison))

    def test_Orkut_Groups(self):
        graph_path = "Orkut-groups/"
        Z_Correct = np.load(self.files_dir + graph_path + "Z_CorrectResults.npy")
        Z_to_check = np.loadtxt(self.files_dir + graph_path + "Z_to_check.csv")

        comparison = np.isclose(Z_Correct, Z_to_check, atol=1e-05)

        self.assertTrue(np.all(comparison))


    def test_Friendster(self):
        graph_path = "Friendster/"
        Z_Correct = np.load(self.files_dir + graph_path + "Z_CorrectResults.npy")
        Z_to_check = np.load(self.files_dir + graph_path + "Z_to_check.npy")

        comparison = np.isclose(Z_Correct, Z_to_check, atol=1e-05)

        self.assertTrue(np.all(comparison))
    

if __name__ == '__main__':
    unittest.main()

