import unittest
import numpy as np
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils

class TestDataProcessing(unittest.TestCase):

    def test_group_crash_events(self):
        # Case 1: Simple contiguous block
        y_true = np.array([0, 0, 1, 1, 1, 0, 0])
        expected = [(2, 4)]
        self.assertEqual(utils.group_crash_events(y_true), expected)

        # Case 2: Multiple blocks
        y_true = np.array([1, 1, 0, 0, 1, 0, 1, 1])
        expected = [(0, 1), (4, 4), (6, 7)]
        self.assertEqual(utils.group_crash_events(y_true), expected)

        # Case 3: No events
        y_true = np.array([0, 0, 0])
        expected = []
        self.assertEqual(utils.group_crash_events(y_true), expected)

    def test_create_warning_labels(self):
        # Case 1: Window fits perfectly
        y_true = np.array([0, 0, 0, 0, 1, 1, 0])
        window = 2
        # Expect 1s at indices 2 and 3 (4-2=2 to 4)
        # Crash days (4, 5) should be 0
        expected = np.array([0, 0, 1, 1, 0, 0, 0])
        result = utils.create_warning_labels(y_true, window)
        np.testing.assert_array_equal(result, expected)

        # Case 2: Window overlaps with start of array
        y_true = np.array([0, 1, 0])
        window = 5
        # Should fill from index 0 up to 1
        expected = np.array([1, 0, 0])
        result = utils.create_warning_labels(y_true, window)
        np.testing.assert_array_equal(result, expected)

if __name__ == '__main__':
    unittest.main()