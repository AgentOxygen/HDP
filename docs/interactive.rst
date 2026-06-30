Interactive Parameter Explorer
====================

This page lets you *feel* how the two key parameter choices in the HDP workflow, namely the **threshold** and the **heatwave definition** ``[min_duration, max_break, max_subs]``, change which days count as a heatwave. It runs entirely in your browser on a small synthetic slice of data.

The top pane shows a 30-day temperature time series that rises and then falls with some daily noise, along with a flat threshold line. The bottom pane illustrates how heatwaves can are identified by a step classification:

* ``0`` - **not hot** (at or below the threshold)
* ``1`` - **hot day** (above the threshold, but not part of a qualifying heatwave)
* ``2`` - **heatwave day** (part of an event that satisfies the heatwave definition)

Move the sliders to update the bottom pane. This mirrors how the HDP computes heatwaves via :py:func:`hdp.metric.index_heatwaves`.

.. note::

   The threshold here is a single value for illustration. In HDP, the threshold
   is typically a **percentile computed for each day of the year** from a baseline measure,
   so the real cutoff varies across the calendar rather than being a flat line.
   See the :doc:`overview` for how thresholds are actually derived.

.. raw:: html

   <div id="hdp-hw-widget">
     <div class="hdp-hw-controls">
       <div class="hdp-hw-control">
         <label for="hdp-hw-threshold">Threshold (°C)<span class="hdp-hw-val" id="hdp-hw-threshold-val">35.0°C</span></label>
         <input type="range" id="hdp-hw-threshold" min="28" max="42" step="0.5" value="35">
       </div>
       <div class="hdp-hw-control">
         <label for="hdp-hw-min-duration">Minimum heatwave length<span class="hdp-hw-val" id="hdp-hw-min-duration-val">3 days</span></label>
         <input type="range" id="hdp-hw-min-duration" min="1" max="7" step="1" value="3">
       </div>
       <div class="hdp-hw-control">
         <label for="hdp-hw-max-break">Max break days after start<span class="hdp-hw-val" id="hdp-hw-max-break-val">1</span></label>
         <input type="range" id="hdp-hw-max-break" min="0" max="4" step="1" value="1">
       </div>
       <div class="hdp-hw-control">
         <label for="hdp-hw-max-subs">Max subsequent events<span class="hdp-hw-val" id="hdp-hw-max-subs-val">1</span></label>
         <input type="range" id="hdp-hw-max-subs" min="0" max="4" step="1" value="1">
       </div>
     </div>

     <div class="hdp-hw-actions">
       <button type="button" id="hdp-hw-regenerate">Regenerate data</button>
     </div>

     <div class="hdp-hw-pane">
       <div class="hdp-hw-pane-title">Pane 1 - daily temperature (°C) and threshold</div>
       <canvas id="hdp-hw-temp"></canvas>
     </div>
     <div class="hdp-hw-pane">
       <div class="hdp-hw-pane-title">Pane 2 - heatwave classification (0 = not hot, 1 = hot, 2 = heatwave)</div>
       <canvas id="hdp-hw-step"></canvas>
     </div>

     <div class="hdp-hw-metrics" id="hdp-hw-metrics"></div>
   </div>

Heatwave Metric Interpretations
-------------------------------

The four boxes below the panes report the HDP metrics computed **over this
slice** (not per-season, as in a real run):

* **HWN** - number of distinct heatwave events
* **HWF** - total heatwave days
* **HWD** - length of the longest event
* **HWA** - mean event length