# Heatwave Analysis Package
```
aerosol-heatwaves
│   .gitignore
|   README.md
│   environment.yml
|   heatwave_threshold.py
└───heatwave_metrics.py
```
## A Useful Package for Quantifying Heatwaves
This package contains methods for quantifying extreme heat events that have been optimized for code-readability while maintaining an acceptable level of performance for use on HPCs. The [https://github.com/tammasloughran/ehfheatwaves](original algorithm) was written by [Dr. Tammas Loughran](https://www.linkedin.com/in/tammas-loughran-839a31150/?originalSubdomain=au) and later modified by [Dr. Jane Baldwin](https://www.janebaldw.in/) for use in her AGU publication: [Temporally Compound Heat Wave Events and Global Warming: An Emerging Hazard](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2018EF000989).

`heatwave_threshold.py` contains the `compute_threshold` function which takes temperature time series data as an input and outputs a 365-day time series of threshold temperatures for each day of the year at the specified percentile. The data output by the `compute_threshold` function is a required input for many of the functions contained in `heatwave_metrics.py`.

All code is annotated and each function definition is followed by a docustring that outlines the parameters and produced output. A brief description of each function is detailed below.
