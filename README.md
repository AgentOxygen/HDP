# Heatwave Analysis Methodology
A brief outline of the methodology used for computing the heatwave thresholds and metrics. This methodology is applicable to any daily temperature dataset, though the interpretation of resulting metrics varies between the choice of daily minimum, maximum, or average temperature.
## Threshold
A baseline daily temperature dataset is selected for defining the threshold of what daily temperature constitutes an "exceedingly hot day." The threshold is a seasonally measured percentile temperature from January 1st to December 31st for each location in the global grid, however we only use the warm seasons for computing the heatwave metrics (see the "Heatwave Days" section below). A series of schematics are provided to understand the process for computing the heatwave threshold data.

![Schematic of gridded timeseries data being analyzed to produce seasonal percentiles thresholds for each grid cell.](imgs/schematic1.png "Schematic 1")

The baseline dataset should be a gridded timeseries of daily temperature data spanning at least 30 continuous years. For this experiment, we use the 1961 to 1990 ETCCDI baseline definition for extreme daily minimum temperature. The data's array should be described by date timestamps (time t), latitudinal coordinates (spatial y), and longitudinal coordinates (spatial x). To compute a seasonal threshold, we need to analyze each day of the year (that is, one of the 365 days of the year) separately. The schematic above shows how we obtain the January 1st daily minimum temperatures for each year from 1961 to 1990 (reducing the time dimension from 10950 lat./lon. grids to 30 lat./lon. grids). The 95th percentile temperature (q) for each grid cell's annual timeseries is then computed with the numpy function call. The percentile computed is arbitrary, but we chose 95th for this experiment. This produces the dataset structure shown below:

![Schematic of structure of threshold data.](imgs/schematic2.png "Schematic 2")

The resulting threshold dataset has the same spatial dimensions as the original data. The time dimension has units "day of year," referring to each day in the 365 calendar year (Jan 1st, Jan 2nd, etc.). The resulting time dimension for the threshold dataset has a length of 365. Note that this value may need to change depending on what calendar the baseline data uses (our experiment uses the 365 noleap calendar).

## Heatwave Days

To determine which days in a timeseries of interest are part of a heatwave, we need to first indicate which days exceed the temperature threshold. These days with exceedingly hot temperatures may or may not constitute a heatwave, so we refer to them simply as "hot days" before we apply a heatwave definition. Computationally, hot days are stored as boolean flags where true (represented by a 1) indicates a hot day and false (represented by a 0) indicates a non-hot day.

![Schematic showing how hot days are computed using timeseries data.](imgs/schematic3.png "Schematic 3")

The schematic above shows how a comparable gridded daily minimum temperature timeseries, such as future SSP projections, is compared against the threshold we generated using the baseline data. In order to apply the seasonally defined threshold, the timeseries of interest must be sliced annually such that each slice contains the same days (Jan 1st, Jan 2nd, etc.) as the threshold. An array comparison is computed for each year and grid cell separately, indicating which days exceed the threshold in the "hot days" boolean array. The output from this process is essentially a dataset of boolean flags where each flag corresponds to a daily minimum temperature in the timeseries of interest, indicating whether that temperature constitutes a hot day or not. Therefore, the resulting boolean dataset will be of the same dimensions as the timeseries of interest.

![Schematic showing how heatwave days are computed using hot days boolean data.](imgs/schematic4.png "Schematic 4")

To determine which hot days constitute a heatwave (and are thus heatwave days), an algorithm is used to iterate over the hot day timeseries and systematically identify heatwaves based on a heatwave definition. The heatwave definition is comprised of two variables that refer to consecutive hot days as "events" of a heatwave: the minimum number of hot days in the first event and the maximum number of non-hot days between the first and second event. This definition allows the conditional second event to be shorter than the first event (as short as one hot day), but only allows for a maximum of two events per heatwave. For this example, we assume a heatwave definition of at least three hot days in the first event followed by no more than a one day break. The case examples are provided with explanations below:

> A - No heatwave. The first event is only two hot days long.
> 
> B - No heatwave. The third hot day is not considered as part of the first event because there is a break and thus the first event is only two hot days long.
> 
> C - No heatwave. The last two hot days are not considered as part of the first event because there is a break and thus the first event is only two hot days long.
> 
> D - Heatwave. The first event is three days long and there is no second event (the second event is conditional).
> 
> E - Heatwave. The first event is three days long and the second event is separated by only one day.
> 
> F - Heatwave. The first event is three days long and the second event is two days long, separated by only one day break.
> 
> G - Heatwave. The first event is four days long and there is no second event.
> 
> H* - Partial Heatwave. The first event is three days long and the second event is one day long, separated by only one day break. The last hot day would constitute a theoretical "third" event due to the break between the second event, but our definition does not allow for more than two events per heatwave.

Heatwave days can then be identified by applying this definition to the hot days boolean dataset, replacing hot days that do not constitute part of a heatwave with zeros. Note that the number of heatwave days can only be equal to or less than the number of hot days.

## Heatwave Frequency and Duration
To calculate the heatwave frequency for each year, we count the number of heatwave days each year. To calculate heatwave duration for each year, we identify the greatest number of heatwave days in a single heatwave for that year. Note that for a given heatwave, there may be two events and a break. This means that while a heatwave spans both events, it also spans the break and thus contains days that are not hot days. Heatwave frequency and duration do not count these non-hot days as heatwave days even though they connect the first and second events. So, if the longest heatwave in a given year has a three day first event followed by a one day break and then a three day second event, then the heatwave duration would be six days for that particular year.
