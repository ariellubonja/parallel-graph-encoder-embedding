import unittest

import numpy as np


class TestStringMethods(unittest.TestCase):

    def test_Twitch(self):
        files_dir = "../../../Experiments/Compare-Results/"
        graph_path = "Twitch/"
        Z_Correct = np.load(files_dir + graph_path + "Z_CorrectResults.npy")
        Z_to_check = np.loadtxt(files_dir + graph_path + "Z_to_check.csv")

        comparison = np.isclose(Z_Correct, Z_to_check)

        self.assertTrue(np.all(comparison))

    def test_Pokec(self):
        files_dir = "../../../Experiments/Compare-Results/"
        graph_path = "Pokec/"
        Z_Correct = np.load(files_dir + graph_path + "Z_CorrectResults.npy")
        Z_to_check = np.loadtxt(files_dir + graph_path + "Z_to_check.csv")

        comparison = np.isclose(Z_Correct, Z_to_check)

        self.assertTrue(np.all(comparison))

    def test_LiveJournal(self):
        files_dir = "../../../Experiments/Compare-Results/"
        graph_path = "LiveJournal/"
        Z_Correct = np.load(files_dir + graph_path + "Z_CorrectResults.npy")
        Z_to_check = np.loadtxt(files_dir + graph_path + "Z_to_check.csv")

        comparison = np.isclose(Z_Correct, Z_to_check)

        self.assertTrue(np.all(comparison))

    def test_Orkut(self):
        files_dir = "../../../Experiments/Compare-Results/"
        graph_path = "Orkut/"
        Z_Correct = np.load(files_dir + graph_path + "Z_CorrectResults.npy")
        Z_to_check = np.loadtxt(files_dir + graph_path + "Z_to_check.csv")

        comparison = np.isclose(Z_Correct, Z_to_check)

        self.assertTrue(np.all(comparison))


    def test_Orkut_Groups(self):
        files_dir = "../../../Experiments/Compare-Results/"
        graph_path = "Orkut-groups/"
        Z_Correct = np.load(files_dir + graph_path + "Z_CorrectResults.npy")
        Z_to_check = np.loadtxt(files_dir + graph_path + "Z_to_check.csv")

        comparison = np.isclose(Z_Correct, Z_to_check)

        self.assertTrue(np.all(comparison))


if __name__ == '__main__':
    unittest.main()

