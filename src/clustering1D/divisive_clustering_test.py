import unittest
import random
from divisive_clustering_1D import DivisiveCluster
import dataLoader.dataset_loader as loader


class TestDivisiveCluster(unittest.TestCase):
    """Some unit tests to test the cluster algo"""

    def setUp(self):
        self.simple_data = [1.1, 1.2, 5.0, 5.1, 5.2, 10.0, 10.5, 11.0, 20.0, 20.1]
        #result should be [1.1,1.2],[5.0-5.2][10-11][20,20.1]

    def test_with_simple_data(self):
        clusterer = DivisiveCluster(max_depth=4, min_cluster_size=1)
        clusterer.fit(self.simple_data)
        clusters = clusterer.get_clusters()
        print(clusters)
        self.assertEqual(len(clusters), 4)

    def test_with_dataset(self):
        dataset = loader.load_dataloader_by_name('bank',y_seperated = False)
        complete_x = dataset.get_x_complete()
        print(f'Y should be none:{dataset.get_y_complete()}')
        scaled_x_T = loader.standardize_2d_array(complete_x.T)
        print(scaled_x_T)
        clusterer = DivisiveCluster(max_depth=16, min_cluster_size=1)
        clusterer.fit(scaled_x_T[0])
        clusters = clusterer.get_clusters()


if __name__ == "__main__":
    unittest.main()