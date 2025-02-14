import xarray
import datetime
from time import time
from importlib.metadata import version

def get_time_stamp():
    return datetime.datetime.fromtimestamp(time()).strftime('%Y-%m-%d %H:%M')


def add_history(ds, msg):
    if "history" in ds.attrs:
        ds.attrs["history"] += f"({get_time_stamp()}) {msg}\n"
    else:
        ds.attrs["history"] = f"({get_time_stamp()}) History metadata initialized by HDP v{get_version()}.\n"
    return ds


def get_version():
    return version('hdp_python')


def get_func_description(func):
    lines = func.__doc__.split("\n")
    desc = ""
    for line in lines:
        if ":param" in line:
            break
        line = line.strip()
        if line != "":
            desc += line.strip() + " "
    return desc