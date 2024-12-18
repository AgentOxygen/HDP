import numpy as np
import numba as nb

def main():
    pass

@nb.njit
def heat_index(temp: float, rel_humid: float):
    r"""Calculates heat index from temperature and relative humidity.

    This function relies on a regression of the National Weather Service heat index.
    It has an error of +/- 1.3 degrees Fahrenheit.
    Source: https://www.weather.gov/ama/heatindex

    Parameters
    ----------
    temp : float
        Temperature value in degrees Fahrenheit
    rel_humid : float
        Relative humidity on scale from 0 (no moisture) to 100 (fully saturated).

    Returns
    -------
    hi : float
        Heat index value
    """
    hi = -42.379
    hi += 2.04901523*temp
    hi += 10.14333127*rel_humid 
    hi += -0.22475541*temp*rel_humid 
    hi += -0.00683783*(temp**2)
    hi += -0.05481717*(rel_humid**2)
    hi += 0.00122874*(temp**2)*rel_humid
    hi += 0.00085282*temp*(rel_humid**2)
    hi += -0.00000199*((rel_humid*temp)**2)
    
    if rel_humid < 13 and 80 <= temp <= 112:
        hi -= ((13 - rel_humid)/4)*np.sqrt(((17 - np.abs(temp - 95))/17))
    elif rel_humid > 85 and 80 <= temp <= 87:
        hi += ((rel_humid - 85)/10) * ((87 - temp)/5)
    
    if hi < 80:
        hi = 0.5 * (temp + 61.0 + ((temp - 68.0)*1.2) + (rel_humid*0.094))

    return hi


if __name__ == "__main__":
    main()