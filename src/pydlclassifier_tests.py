from binaryConvertion.binner import bin_convertion_2d
import dataLoader
import unittest

class TestClassifyTree(unittest.TestCase):
    def test_generate_classify_tree(self):
        dataloader = dataLoader.load_dataloader_by_name("bank")
        np_array_transposed = dataloader.get_values().transpose()
        bin_array = bin_convertion_2d(np_array_transposed, max_bins=2).transpose()
        print(bin_array)










if __name__ == "__main__":
    unittest.main()