#!/usr/bin/env python
"""
cli.py

Command Line Interface for running HDP in the terminal without a notebook.

Developer: Cameron Cummins
Contact: cameron.cummins@utexas.edu
"""
from importlib.metadata import version as getVersion
import argparse


def main():
    parser = argparse.ArgumentParser(description = f"Command Line Interface (CLI) for Heatwave Diagnostics Package (HDP) Version {getVersion('hdp_python')}")
    subparsers = parser.add_subparsers(help='HDP commands')
    parser.add_argument('-d', '--dask', metavar="ADDRESS", type=str, help="Address of existing dask cluster to connect to instead of creating internally")
    parser.add_argument('-o', '--overwrite', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    
    # measure_parser = subparsers.add_parser('measure', help='Compute measures of heat from climate variable datasets')
    threshold_parser = subparsers.add_parser('threshold', help='Generate a range of extreme heat thresholds from baseline heat measure datasets')
    metric_parser = subparsers.add_parser('metric', help='Generate heatwave metrics for given threshold and heat measure datasets')

    # measure_parser.add_argument('output_path', metavar="PATH", type=str, help="Path to write heat measure dataset to as a netCDF or Zarr store")
    # # measure_parser.add_argument('-i', '--input_path', metavar="PATH", type=str, help="Path to netCDF for Zarr store containing general daily data")
    # # measure_parser.add_argument('-var', '--variable', metavar="NAME", type=str, help="Variable to use for general dataset")
    # measure_parser.add_argument('-ta', '--tavg_path', metavar="PATH", type=str, help="Path to netCDF or Zarr store containing daily average temperature data")
    # measure_parser.add_argument('-tx', '--tmax_path', metavar="PATH", type=str, help="Path to netCDF or Zarr store containing daily maximum temperature data")
    # measure_parser.add_argument('-tn', '--tmin_path', metavar="PATH", type=str, help="Path to netCDF or Zarr store containing daily minimum temperature data")
    # measure_parser.add_argument('-rh', '--rh_path', metavar="PATH", type=str, help="Path to netCDF for Zarr store containing daily relative humidity data")
    # measure_parser.add_argument('-tav', '--tavg_var', metavar="NAME", type=str, help="Variable to use for netCDF or Zarr store containing daily average temperature data")
    # measure_parser.add_argument('-txv', '--tmax_var', metavar="NAME", type=str, help="Variable to use for netCDF or Zarr store containing daily maximum temperature data")
    # measure_parser.add_argument('-tnv', '--tmin_var', metavar="NAME", type=str, help="Variable to use for netCDF or Zarr store containing daily minimum temperature data")
    # measure_parser.add_argument('-rhv', '--rh_var', metavar="NAME", type=str, help="Variable to use for netCDF to netCDF or Zarr store containing daily relative humidity data")
    # measure_parser.add_argument('-a', '--append', action='store_true', help="If output dataset exists, append new variables")

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
    
    args = parser.parse_args()
    # measure_args = measure_parser.parse_args()
    threshold_args = threshold_parser.parse_args()
    metric_args = metric_parser.parse_args()

if __name__ == "__main__":
    main()