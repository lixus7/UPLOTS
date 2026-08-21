import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from Utils.Data_utils.real_datasets import CustomDataset


class DatasetLoadingTest(unittest.TestCase):
    def test_etth_csv_drops_timestamp_column(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'etth.csv'
            pd.DataFrame({
                'date': ['2026-01-01', '2026-01-02'],
                'a': [1.0, 2.0],
                'b': [3.0, 4.0],
            }).to_csv(path, index=False)
            data, _ = CustomDataset.read_data(path, name='etth')
        self.assertEqual(data.shape, (2, 2))

    def test_pems_csv_remains_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pems.csv'
            expected = np.arange(12, dtype=float).reshape(3, 4)
            pd.DataFrame(expected).to_csv(path, index=False)
            data, _ = CustomDataset.read_data(path, name='pems04')
        np.testing.assert_allclose(data, expected)

    def test_pems_npz_uses_flow_feature(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pems.npz'
            source = np.arange(24, dtype=float).reshape(3, 4, 2)
            np.savez(path, data=source)
            data, _ = CustomDataset.read_data(path, name='pems04')
        np.testing.assert_allclose(data, source[:, :, 0])


if __name__ == '__main__':
    unittest.main()
