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
        ds.attrs["history"] = f"({get_time_stamp()}) History metadata initialized by HDP.\n"
    return ds


def getVersion():
    return version('hdp_python')