import unittest
from main import main

class TestDatasetB(unittest.TestCase):
    def test_dataset_b(self):
        self.assertAlmostEqual(main(dataset='dataset_b', graph_builder='structural', metric_filter=['nonzero'], graph_filter=['pearson'], ranker='latentregressor', cpus=16), 0.615, places=2)

if __name__ == '__main__':
    unittest.main()
