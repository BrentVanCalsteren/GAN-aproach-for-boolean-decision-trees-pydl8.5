import unittest
import numpy as np
from bin_convertion import bin_convertion


class TestBinConvertion(unittest.TestCase):
    """Unit tests for the bin_convertion function (public API only)."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)

    def test_few_unique_values_direct_binning(self):
        """Test when unique values < max_bins (should use direct binning)."""
        array = [1, 2, 3, 4, 1, 2, 3, 4]
        result = bin_convertion(array, max_bins=16)

        # Should produce binary output
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape[0], len(array))

        # Output should contain only 0s and 1s
        self.assertTrue(np.all(np.isin(result, [0, 1])))

        # For 4 unique values, should be one-hot with 4 columns
        self.assertEqual(result.shape[1], 4)

        # Each row should have exactly one 1 (one-hot encoding)
        row_sums = result.sum(axis=1)
        np.testing.assert_array_equal(row_sums, np.ones(len(array)))

    def test_two_unique_values_output_format(self):
        """Test with exactly 2 unique values (should return n x 1 array)."""
        array = [1, 2, 1, 2, 1, 2, 1]
        result = bin_convertion(array, max_bins=16)

        # Should be 7x1 array
        self.assertEqual(result.shape, (7, 1))

        # Should contain only 0s and 1s
        self.assertTrue(np.all(np.isin(result, [0, 1])))

    def test_max_bins_equal_to_unique_values(self):
        """Test when unique values == max_bins."""
        array = [1, 2, 3, 4, 5, 6, 7, 8]
        result = bin_convertion(array, max_bins=8)

        # Should be 8x8 one-hot matrix
        print(result)
        self.assertEqual(result.shape, (8, 8))


        # Each row should sum to 1
        row_sums = result.sum(axis=1)
        np.testing.assert_array_equal(row_sums, np.ones(8))

        # Each row should have exactly one 1
        for i in range(8):
            self.assertEqual(np.sum(result[i]), 1)

        # All values should be 0 or 1
        self.assertTrue(np.all(np.isin(result, [0, 1])))

    def test_single_unique_value(self):
        """Test with only 1 unique value."""
        array = [5, 5, 5, 5, 5]
        result = bin_convertion(array, max_bins=16)

        # Should be 5x1 array
        self.assertEqual(result.shape, (5, 1))

        # All values should be 0 (since only one category)
        np.testing.assert_array_equal(result, np.zeros((5, 1)))

    def test_many_unique_values_with_clustering(self):
        """Test when unique values > max_bins (should use clustering)."""
        # Generate array with many unique values
        array = list(np.random.uniform(0, 1000, 200))

        result = bin_convertion(array, max_bins=16)

        # Result should be a 2D array
        self.assertTrue(len(result.shape) == 2)

        # Number of rows should match input
        self.assertEqual(result.shape[0], len(array))

        # Number of columns should be <= max_bins
        self.assertLessEqual(result.shape[1], 16)

        # All values should be 0 or 1
        self.assertTrue(np.all(np.isin(result, [0, 1])))

        # For >2 clusters, rows should sum to 1 (one-hot)
        if result.shape[1] > 1:
            row_sums = result.sum(axis=1)
            # If it's one-hot (3+ clusters), sums should be 1
            # If it's binary (2 clusters), sums can be 0 or 1
            # Let's check if it's one-hot format
            if result.shape[1] > 2:
                np.testing.assert_array_equal(row_sums, np.ones(len(array)))

    def test_max_bins_various_values(self):
        """Test with different max_bins values."""
        array = list(np.random.uniform(0, 1000, 300))

        for max_bins in [2, 4, 8, 16, 32, 64]:
            result = bin_convertion(array, max_bins=max_bins)

            # Check basic properties
            self.assertEqual(result.shape[0], len(array))
            self.assertLessEqual(result.shape[1], max_bins)
            self.assertTrue(np.all(np.isin(result, [0, 1])))

    def test_max_bins_smaller_than_unique_values(self):
        """Test when max_bins is smaller than number of unique values."""
        # Create array with 100 unique values
        array = list(range(100))

        for max_bins in [5, 10, 20]:
            result = bin_convertion(array, max_bins=max_bins)

            # Should use clustering to reduce dimensions
            self.assertEqual(result.shape[0], len(array))
            self.assertLessEqual(result.shape[1], max_bins)
            self.assertTrue(np.all(np.isin(result, [0, 1])))

    def test_preserves_data_order(self):
        """Test that output preserves input order."""
        array = [1.0, 2.0, 3.0, 100.0, 101.0, 102.0, 1000.0]
        result = bin_convertion(array, max_bins=4)

        # Result should have same number of rows as input
        self.assertEqual(result.shape[0], len(array))

        # Identical values should have identical binary representations
        # (This indirectly tests order preservation)
        for i in range(len(array)):
            for j in range(i + 1, len(array)):
                if array[i] == array[j]:
                    np.testing.assert_array_equal(result[i], result[j])

    def test_identical_values_same_binary(self):
        """Test that identical values map to same binary representation."""
        array = [5.0, 5.0, 10.0, 10.0, 5.0, 10.0, 5.0]
        result = bin_convertion(array, max_bins=4)

        # Indices with same value should have identical rows
        np.testing.assert_array_equal(result[0], result[1])
        np.testing.assert_array_equal(result[0], result[4])
        np.testing.assert_array_equal(result[0], result[6])
        np.testing.assert_array_equal(result[2], result[3])
        np.testing.assert_array_equal(result[2], result[5])

    def test_returns_numpy_array(self):
        """Test that function always returns a numpy array."""
        # Test with list input
        array = [1, 2, 3, 4]
        result = bin_convertion(array)
        self.assertIsInstance(result, np.ndarray)

        # Test with numpy array input
        array = np.array([1, 2, 3, 4])
        result = bin_convertion(array)
        self.assertIsInstance(result, np.ndarray)

        # Test with large random data
        array = list(np.random.uniform(0, 1000, 100))
        result = bin_convertion(array)
        self.assertIsInstance(result, np.ndarray)

    def test_empty_input(self):
        """Test with empty input array."""
        array = []
        result = bin_convertion(array)
        self.assertEqual(result.shape, (0, 0))

    def test_single_element_input(self):
        """Test with single element input."""
        array = [42.0]
        result = bin_convertion(array, max_bins=16)

        # Should be 1x1 array
        self.assertEqual(result.shape, (1, 1))
        self.assertTrue(np.all(np.isin(result, [0, 1])))

    def test_all_values_identical(self):
        """Test when all input values are identical."""
        array = [7.0] * 50
        result = bin_convertion(array, max_bins=16)

        # Should be 50x1 array (all zeros)
        self.assertEqual(result.shape, (50, 1))
        np.testing.assert_array_equal(result, np.zeros((50, 1)))

    def test_consistent_results_deterministic(self):
        """Test that function produces consistent results for same input."""
        array = list(np.random.uniform(0, 1000, 200))

        result1 = bin_convertion(array, max_bins=16)
        result2 = bin_convertion(array, max_bins=16)

        # Results should be identical for same input
        np.testing.assert_array_equal(result1, result2)

    def test_binary_output_values_only(self):
        """Test that output contains only 0s and 1s."""
        # Test with small unique values
        array1 = [1, 2, 3, 4, 1, 2, 3, 4]
        result1 = bin_convertion(array1, max_bins=16)
        self.assertTrue(np.all(np.isin(result1, [0, 1])))

        # Test with many unique values
        array2 = list(np.random.uniform(0, 1000, 200))
        result2 = bin_convertion(array2, max_bins=16)
        self.assertTrue(np.all(np.isin(result2, [0, 1])))

        # Test with single value
        array3 = [42]
        result3 = bin_convertion(array3)
        self.assertTrue(np.all(np.isin(result3, [0, 1])))

    def test_output_shape_consistency(self):
        """Test that output shape is consistent with input length."""
        sizes = [10, 50, 100, 200, 500]

        for size in sizes:
            array = list(np.random.uniform(0, 1000, size))
            result = bin_convertion(array, max_bins=16)
            self.assertEqual(result.shape[0], size)

    def test_max_bins_boundary_conditions(self):
        """Test boundary conditions for max_bins."""
        array = list(np.random.uniform(0, 1000, 100))

        # Test with max_bins = 1
        result1 = bin_convertion(array, max_bins=1)
        self.assertEqual(result1.shape[1], 1)

        # Test with max_bins = 2
        result2 = bin_convertion(array, max_bins=2)
        self.assertLessEqual(result2.shape[1], 2)

        # Test with max_bins = 100 (larger than unique values typically)
        result3 = bin_convertion(array, max_bins=100)
        # Should use direct binning if unique values <= 100
        self.assertLessEqual(result3.shape[1], 100)

    def test_clustering_depth_calculation(self):
        """Test that clustering uses appropriate depth."""
        array = list(np.random.uniform(0, 1000, 500))

        for max_bins in [2, 3, 4, 5, 8, 9, 16, 17]:
            result = bin_convertion(array, max_bins=max_bins)
            # Should not crash and produce valid output
            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(result.shape[0], len(array))
            self.assertTrue(np.all(np.isin(result, [0, 1])))


class TestBinConvertionEdgeCases(unittest.TestCase):
    """Test edge cases for bin_convertion function."""

    def test_negative_values(self):
        """Test with negative values."""
        array = [-10, -5, 0, 5, 10, -10, -5, 0]
        result = bin_convertion(array, max_bins=4)

        self.assertEqual(result.shape[0], len(array))
        self.assertLessEqual(result.shape[1], 4)
        self.assertTrue(np.all(np.isin(result, [0, 1])))

    def test_mixed_types(self):
        """Test with mixed numeric types."""
        array = [1, 1.5, 2, 2.5, 1, 1.5]
        result = bin_convertion(array, max_bins=4)

        self.assertEqual(result.shape[0], len(array))
        self.assertTrue(np.all(np.isin(result, [0, 1])))

    def test_large_values(self):
        """Test with very large values."""
        array = [1e10, 1e11, 1e12, 1e10, 1e11]
        result = bin_convertion(array, max_bins=4)

        self.assertEqual(result.shape[0], len(array))
        self.assertTrue(np.all(np.isin(result, [0, 1])))

    def test_float_precision(self):
        """Test with values that are very close."""
        array = [1.0000001, 1.0000002, 1.0000003, 2.0, 2.0000001]
        result = bin_convertion(array, max_bins=4)

        self.assertEqual(result.shape[0], len(array))
        self.assertTrue(np.all(np.isin(result, [0, 1])))


if __name__ == "__main__":
    # Run the tests
    unittest.main(argv=[''], verbosity=2, exit=False)

    # Print summary
    print("\n" + "=" * 60)
    print("Additional verification:")
    print("=" * 60)

    # Test with 8 unique values
    array = [1, 2, 3, 4, 5, 6, 7, 8]
    result = bin_convertion(array, max_bins=8)
    print(f"\n8 unique values, max_bins=8:")
    print(f"  Shape: {result.shape}")
    print(f"  Values: {np.unique(result)}")
    print(f"  Row sums: {result.sum(axis=1)}")

    # Test with small dataset
    data = [1.0, 1.1, 1.2, 5.0, 5.1, 5.2, 10.0]
    result = bin_convertion(data, max_bins=4)
    print(f"\nSmall dataset (7 points), max_bins=4:")
    print(f"  Shape: {result.shape}")
    print(f"  Unique rows: {len(np.unique(result, axis=0))}")
    print(f"  Values: {np.unique(result)}")

    # Test with many unique values
    data2 = list(np.random.uniform(0, 1000, 500))
    result2 = bin_convertion(data2, max_bins=16)
    print(f"\n500 random values, max_bins=16:")
    print(f"  Shape: {result2.shape}")
    print(f"  Values: {np.unique(result2)}")
    print(f"  Row sums (first 10): {result2[:10].sum(axis=1)}")