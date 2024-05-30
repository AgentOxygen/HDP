import unittest
import numpy as np
import heat_stats
from numpy.testing import assert_allclose


class TestHeatStats(unittest.TestCase):

    def test_index_heatwave(self):
        hot_days_1 = np.array(
            [0, 1, 1, 1, 1, 0, 0, 1, 0, 0]
        )
        assert_allclose(heat_stats.index_heatwaves(hot_days_1, max_break=2, min_duration=3),
                        np.array([0, 1, 1, 1, 1, 0, 0, 1, 0, 0]))
        assert_allclose(heat_stats.index_heatwaves(hot_days_1, max_break=1, min_duration=3),
                        np.array([0, 1, 1, 1, 1, 0, 0, 0, 0, 0]))
        assert_allclose(heat_stats.index_heatwaves(hot_days_1, max_break=1, min_duration=4),
                        np.array([0, 1, 1, 1, 1, 0, 0, 0, 0, 0]))
        assert_allclose(heat_stats.index_heatwaves(hot_days_1, max_break=1, min_duration=5),
                        np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))

        hot_days_2 = np.array(
            [0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1]
        )
        assert_allclose(heat_stats.index_heatwaves(hot_days_2, max_break=1, min_duration=3),
                        np.array([0, 1, 1, 1, 0, 0, 2, 2, 2, 0, 2]))
        assert_allclose(heat_stats.index_heatwaves(hot_days_2, max_break=2, min_duration=3),
                        np.array([0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0]))

        hot_days_3 = np.array(
            [0, 1, 1, 0, 0, 1, 1, 0, 1, 0]
        )
        assert_allclose(heat_stats.index_heatwaves(hot_days_3, max_break=1, min_duration=3),
                        np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        assert_allclose(heat_stats.index_heatwaves(hot_days_3, max_break=1, min_duration=2),
                        np.array([0, 1, 1, 0, 0, 2, 2, 0, 2, 0]))
        assert_allclose(heat_stats.index_heatwaves(hot_days_3, max_break=2, min_duration=2),
                        np.array([0, 1, 1, 0, 0, 1, 1, 0, 0, 0]))

        hot_days_4 = np.array(
            [1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1]
        )
        assert_allclose(heat_stats.index_heatwaves(hot_days_4, max_break=1, min_duration=2),
                        np.array([0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 2, 2, 0, 2, 0, 0, 0, 3, 3]))
        assert_allclose(heat_stats.index_heatwaves(hot_days_4, max_break=0, min_duration=2),
                        np.array([0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 0, 0, 0, 3, 3]))

        hot_days_null_1 = np.array(
            [1, 1, 1, 1, 1]
        )
        assert_allclose(heat_stats.index_heatwaves(hot_days_null_1, max_break=1, min_duration=1),
                        np.array([1, 1, 1, 1, 1]))
        assert_allclose(heat_stats.index_heatwaves(hot_days_null_1, max_break=1, min_duration=10),
                        np.array([0, 0, 0, 0, 0]))
        assert_allclose(heat_stats.index_heatwaves(hot_days_null_1, max_break=0, min_duration=0),
                        np.array([1, 1, 1, 1, 1]))

        hot_days_null_2 = np.array(
            [0, 0, 0, 0, 0]
        )
        assert_allclose(heat_stats.index_heatwaves(hot_days_null_2, max_break=1, min_duration=1),
                        np.array([0, 0, 0, 0, 0]))
        assert_allclose(heat_stats.index_heatwaves(hot_days_null_2, max_break=1, min_duration=10),
                        np.array([0, 0, 0, 0, 0]))
        assert_allclose(heat_stats.index_heatwaves(hot_days_null_2, max_break=0, min_duration=0),
                        np.array([0, 0, 0, 0, 0]))

    def test_heatwave_frequency(self):
        hw_flags_null = np.array(
            [0, 0, 0, 0, 0, 0, 0, 0]
        )
        self.assertEqual(heat_stats.heatwave_frequency(hw_flags_null), 0)

        hw_flags_all = np.array(
            [1, 1, 1, 1, 1, 1, 1, 1]
        )
        self.assertEqual(heat_stats.heatwave_frequency(hw_flags_all), 8)

        hw_flags_mix = np.array(
            [0, 0, 1, 1, 1, 0, 3, 0, 4, 4, 0, 5, 0]
        )
        self.assertEqual(heat_stats.heatwave_frequency(hw_flags_mix), 7)

    # def test_heatwave_event_frequency(self):
    #     hw_flags_null = np.array(
    #         [0, 0, 0, 0, 0, 0, 0, 0]
    #     )
    #     self.assertEqual(heat_stats.HeatStats.heatwave_event_frequency(hw_flags_null), 0)

    #     hw_flags_all = np.array(
    #         [1, 1, 1, 1, 1, 1, 1, 1]
    #     )
    #     self.assertEqual(heat_stats.HeatStats.heatwave_frequency(hw_flags_all), 1)

    #     hw_flags_mix = np.array(
    #         [0, 0, 1, 1, 1, 0, 3, 0, 4, 4, 0, 5, 0]
    #     )
    #     self.assertEqual(heat_stats.HeatStats.heatwave_frequency(hw_flags_mix), 4)

    def test_heatwave_duration(self):
        hw_flags_null = np.array(
            [0, 0, 0, 0, 0, 0, 0, 0]
        )
        self.assertEqual(heat_stats.heatwave_duration(hw_flags_null), 0)

        hw_flags_all = np.array(
            [1, 1, 1, 1, 1, 1, 1, 1]
        )
        self.assertEqual(heat_stats.heatwave_duration(hw_flags_all), 8)

        hw_flags_mix_1 = np.array(
            [0, 0, 1, 1, 1, 0, 3, 0, 4, 4, 0, 5, 0]
        )
        self.assertEqual(heat_stats.heatwave_duration(hw_flags_mix_1), 3)

        hw_flags_mix_2 = np.array(
            [0, 0, 1, 1, 0, 3, 3, 3, 3, 3, 0, 3, 0]
        )
        self.assertEqual(heat_stats.heatwave_duration(hw_flags_mix_2), 6)

        hw_flags_mix_3 = np.array(
            [0, 0, 1, 1, 0, 3, 3, 3, 3, 3, 0, 4, 0]
        )
        self.assertEqual(heat_stats.heatwave_duration(hw_flags_mix_3), 5)

    # def test_heatwave_average_duration(self):
    #     hw_flags_null = np.array(
    #         [0, 0, 0, 0, 0, 0, 0, 0]
    #     )
    #     self.assertAlmostEqual(heat_stats.HeatStats.heatwave_average_duration(hw_flags_null), 0.0)

    #     hw_flags_all = np.array(
    #         [1, 1, 1, 1, 1, 1, 1, 1]
    #     )
    #     self.assertAlmostEqual(heat_stats.HeatStats.heatwave_average_duration(hw_flags_all), 8.0)

    #     hw_flags_mix_1 = np.array(
    #         [0, 0, 1, 1, 1, 0, 3, 0, 4, 4, 0, 5, 0]
    #     )
    #     self.assertAlmostEqual(heat_stats.HeatStats.heatwave_duration(hw_flags_mix_1), 1.75)

    #     hw_flags_mix_2 = np.array(
    #         [0, 1, 2, 3, 4, 5, 6]
    #     )
    #     self.assertAlmostEqual(heat_stats.HeatStats.heatwave_duration(hw_flags_mix_2), 1)

    #     hw_flags_mix_3 = np.array(
    #         [0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6]
    #     )
    #     self.assertAlmostEqual(heat_stats.HeatStats.heatwave_duration(hw_flags_mix_3), 1)

if __name__ == '__main__':
    unittest.main()