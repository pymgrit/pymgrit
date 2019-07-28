import unittest

import numpy as np
from mgrit import mgrit_fas


class TestMgritFas(unittest.TestCase):
    def test_split_into(self):
        """
        Test that it can sum a list of integers
        """
        result = np.array([4,3,3])
        mgrit = mgrit_fas.MgritFas(problem=[], transfer=[], nested_iteration=False)
        np.testing.assert_equal(result, mgrit.split_into(10,3))

if __name__ == '__main__':
    unittest.main()