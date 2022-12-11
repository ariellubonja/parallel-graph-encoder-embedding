import unittest

import numpy as np


class TestStringMethods(unittest.TestCase):

    def test_Twitch(self):
        files_dir = "../../../Experiments/Compare-Results/"
        graph_path = "Twitch/"
        Z_Correct = np.loadtxt(files_dir + graph_path + "Z_CorrectResults.csv")
        Z_to_check = np.loadtxt(files_dir + graph_path + "Z_to_check.csv")

        comparison = np.isclose(Z_Correct, Z_to_check)

        self.assertTrue(np.all(comparison))

    # def test_isupper(self):
    #     self.assertTrue('FOO'.isupper())
    #     self.assertFalse('Foo'.isupper())
    #
    # def test_split(self):
    #     s = 'hello world'
    #     self.assertEqual(s.split(), ['hello', 'world'])
    #     # check that s.split fails when the separator is not a string
    #     with self.assertRaises(TypeError):
    #         s.split(2)

if __name__ == '__main__':
    unittest.main()

