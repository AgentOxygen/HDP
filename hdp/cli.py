#!/usr/bin/env python
"""
cli.py

Command Line Interface for running HDP in the terminal without a notebook.

Developer: Cameron Cummins
Contact: cameron.cummins@utexas.edu
"""
from hdp.utils import get_version
import argparse
from hdp.threshold import compute_threshold_io
from hdp.metric import compute_metrics_io

def main():
    parser = argparse.ArgumentParser(description = f"Command Line Interface (CLI) for Heatwave Diagnostics Package (HDP) Version {get_version()}")
    subparsers = parser.add_subparsers(help='HDP commands')
    parser.add_argument('-d', '--dask', metavar="ADDRESS", type=str, help="Address of existing dask cluster to connect to instead of creating internally")
    parser.add_argument('-o', '--overwrite', action='store_true')
    parser.add_argument('-v', '--version', action='store_true')
    parser.add_argument('-V', '--verbose', action='store_true')

    args = parser.parse_args()
    if args.version:
        print(get_version())
        return None
    
    threshold_parser = subparsers.add_parser('threshold', help='Generate a range of extreme heat thresholds from baseline heat measure datasets')
    metric_parser = subparsers.add_parser('metric', help='Generate heatwave metrics for given threshold and heat measure datasets')

    threshold_parser.add_argument('output_path', type=str, help="Path to write threshold dataset to as a netCDF or Zarr store")
    threshold_parser.add_argument('input_path', type=str, help="Path to netCDF for Zarr store containing baseline measurements to generate thresholds for")
    threshold_parser.add_argument('-var', '--variable', type=str, help="Variables to calculate thresholds for (defaults to all available)")
    threshold_parser.add_argument('-p','--percentile', nargs='+', help='List of percentiles to calculate thresholds for (ex: 0.9 0.91 0.92)')
    threshold_parser.add_argument('-f', '--fixed', action='store_true', help="Include fixed, non-seasonally-varying percentiles (excluded by default)")
    threshold_parser.add_argument('-ns', '--noseason', action='store_true', help="Exclude seasonally-varying percentiles (included by default)")
    threshold_parser.add_argument('-abs', '--absolute', metavar="FLOAT", type=float, help="Include absolute value as threshold (excluded by default)")
    threshold_parser.add_argument('-w', '--window', metavar="INTEGER", type=int, help="For seasonally varying, number of days for rolling-window percentile calculation")

    metric_parser.add_argument('output_path', metavar="PATH", type=str, help="Path to write heatwave metric dataset to as a netCDF or Zarr store")
    metric_parser.add_argument('measure_path', metavar="PATH", type=str, help="Path to heat measure dataset")
    metric_parser.add_argument('threshold_path', metavar="PATH", type=str, help="Path to threshold dataset")
    metric_parser.add_argument('-var', '--variable', nargs='+', type=str, help="Variable(s) to use from heat measure dataset (defaults to all available)")
    metric_parser.add_argument('-p','--percentile', nargs='+', help='List of percentiles to calculate metrics for, assuming thresholds exist (ex: 0.9 0.91 0.92)')    
    
    threshold_args = threshold_parser.parse_args()
    metric_args = metric_parser.parse_args()
    
    # compute_threshold_io(baseline_path: str,
    #                      baseline_var: str,
    #                      output_path: str,
    #                      percentiles: np.ndarray,
    #                      no_season: bool=False,
    #                      rolling_window_size: int=7,
    #                      fixed_value: float=None,
    #                      overwrite: bool=False)

    # compute_metrics_io(output_path: str,
    #                    measure_path: str,
    #                    measure_var: str,
    #                    threshold_path: str,
    #                    hw_definitions: list,
    #                    include_threshold: bool=False,
    #                    override_threshold_var: str=None)


if __name__ == "__main__":
    main()