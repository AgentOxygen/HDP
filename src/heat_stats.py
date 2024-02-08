#!/usr/bin/env python
"""
heat_core.py

Contains function definitions for various heatwave statistics.
All methods are static and should be called when computing heatwave metrics.

Developer: Cameron Cummins
Contact: cameron.cummins@utexas.edu
2/8/24
"""
from numba import njit
import numpy as np


class HeatStats:
    @staticmethod
    @njit
    def heatwave_frequency(hw_ts: np.array)->int:
        return np.sum(hw_ts > 0)
    
    
    @staticmethod
    @njit
    def heatwave_duration(hw_ts: np.array)->int:
        hwd = 0
        for value in np.unique(hw_ts):
            index_count = 0
            if value != 0:
                for day in hw_ts:
                    if day == value:
                        index_count += 1
            if index_count > hwd:
                hwd = index_count
        return hwd