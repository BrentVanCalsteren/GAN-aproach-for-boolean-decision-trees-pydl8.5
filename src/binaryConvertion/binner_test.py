import unittest
import numpy as np
import binner
import dataLoader.dataset_loader as loader


class TestBinConvertion(unittest.TestCase):
    """Unit tests for the bin_convertion function (public API only)."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)

    def test_simple_binning(self):
        a = [1, 2, 3, 4, 1, 2, 3, 4]
        r = binner.bin_convertion(a, max_bins=16)
        print(r)


    def test_binning_loading_dataset(self):
        dataset = loader.load_dataloader_by_name('bank', y_seperated=False)
        complete_x = dataset.get_x_complete()
        scaled_x_T = loader.standardize_2d_array(complete_x.T)
        binned, bin_length, clusters = binner.bin_convertion_2d(scaled_x_T)
        print(binned)
        print(bin_length)


if __name__ == "__main__":
    # Run the tests
    unittest.main()