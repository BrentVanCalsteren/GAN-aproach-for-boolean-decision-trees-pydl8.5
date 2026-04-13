import unittest
import dataset_loader

class TestDatasetLoader(unittest.TestCase):

    def test_open_all_datasets(self):
        dataset_names = ["bank", "adult", "breast_cancer", "heart_disease", "iris", "wine_quality", "mushroom"]
        for data_name in dataset_names:
            dataset = dataset_loader.load_dataloader_by_name(dataset_name=data_name)
            all_x = dataset.get_x_all()
            print(all_x)
            complete_x = dataset.get_x_complete()
            print(complete_x)
            missing_x = dataset.get_x_missing()
            print(missing_x)
            all_y = dataset.get_y_all()
            print(all_y)
            complete_y = dataset.get_y_complete()
            print(complete_y)
            missing_y = dataset.get_y_missing()
            print(missing_y)
        self.assertTrue(True,"All datasets should have been loaded")

if __name__ == "__main__":
    unittest.main()
