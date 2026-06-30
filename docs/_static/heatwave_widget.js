/*
 * Interactive Heatwave Definition Explorer
 * ----------------------------------------
 * Self-contained, dependency-free widget for the HDP documentation.
 *
 * `indexHeatwaves` below is a faithful JavaScript port of HDP's
 * `index_heatwaves` kernel (the source of truth is hdp/metric.py).
 * It classifies a 0/1 "hot day" timeseries into indexed heatwave events
 * using the definition triple [min_duration, max_break, max_subs].
 *
 * The widget renders entirely client-side (no kernel, no network, no CDN)
 * so it works as static HTML on ReadTheDocs and as a crash-proof live demo.
 */
(function () {
  "use strict";

  // ---------------------------------------------------------------------------
  // Heatwave classification (port of hdp/metric.py :: index_heatwaves)
  // ---------------------------------------------------------------------------
  function indexHeatwaves(hotDays, minDuration, maxBreak, maxSubs) {
    var n = hotDays.length;
    // Zero-pad to n + 2 so leading/trailing events are bounded by a diff edge.
    var ts = new Int32Array(n + 2);
    for (var i = 0; i < n; i++) {
      if (hotDays[i]) ts[i + 1] = 1;
    }

    // Discrete difference of the padded series.
    var diffTs = new Int32Array(ts.length - 1);
    for (var j = 0; j < diffTs.length; j++) {
      diffTs[j] = ts[j + 1] - ts[j];
    }

    // Indices where the series transitions (rising = 1, falling = -1).
    var diffIndices = [];
    for (var k = 0; k < diffTs.length; k++) {
      if (diffTs[k] !== 0) diffIndices.push(k);
    }

    var inHeatwave = false;
    var currentHwIndex = 0;
    var subEvents = 0;
    var hwIndices = new Int32Array(diffTs.length);

    for (var m = 0; m < diffIndices.length - 1; m++) {
      var index = diffIndices[m];
      var nextIndex = diffIndices[m + 1];

      if (diffTs[index] === 1 && nextIndex - index >= minDuration && !inHeatwave) {
        currentHwIndex += 1;
        inHeatwave = true;
        fill(hwIndices, index, nextIndex, currentHwIndex);
      } else if (diffTs[index] === -1 && nextIndex - index > maxBreak) {
        inHeatwave = false;
      } else if (diffTs[index] === 1 && inHeatwave && subEvents < maxSubs) {
        subEvents += 1;
        fill(hwIndices, index, nextIndex, currentHwIndex);
      } else if (diffTs[index] === 1 && inHeatwave && subEvents >= maxSubs) {
        if (nextIndex - index >= minDuration) {
          currentHwIndex += 1;
          fill(hwIndices, index, nextIndex, currentHwIndex);
        } else {
          inHeatwave = false;
        }
        subEvents = 0;
      }
    }

    // Drop the trailing pad cell to realign with the input length.
    return hwIndices.subarray(0, hwIndices.length - 1);
  }

  function fill(arr, start, end, value) {
    for (var i = start; i < end; i++) arr[i] = value;
  }

  // ---------------------------------------------------------------------------
  // Metrics over the slice (analogous to HWN/HWF/HWD/HWA, whole window)
  // ---------------------------------------------------------------------------
  function computeMetrics(hwIndex) {
    var events = {}; // index -> day count
    for (var i = 0; i < hwIndex.length; i++) {
      var v = hwIndex[i];
      if (v !== 0) events[v] = (events[v] || 0) + 1;
    }
    var lengths = Object.keys(events).map(function (key) {
      return events[key];
    });
    var hwn = lengths.length;
    var hwf = lengths.reduce(function (a, b) {
      return a + b;
    }, 0);
    var hwd = lengths.length ? Math.max.apply(null, lengths) : 0;
    var hwa = lengths.length ? hwf / lengths.length : 0;
    return { HWN: hwn, HWF: hwf, HWD: hwd, HWA: hwa };
  }

  // ---------------------------------------------------------------------------
  // Deterministic synthetic data: rise-then-fall hump + seeded noise
  // ---------------------------------------------------------------------------
  var N_DAYS = 30;

  function mulberry32(seed) {
    return function () {
      seed |= 0;
      seed = (seed + 0x6d2b79f5) | 0;
      var t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function generateTemps(seed) {
    var rand = mulberry32(seed);
    var temps = new Float64Array(N_DAYS);
    var base = 34;
    // var amplitude = 12; // peak ~40 degC at the middle of the window
    for (var i = 0; i < N_DAYS; i++) {
      // Smooth hump peaking near the center of the two-week window.
      // var hump = Math.sin((i / (N_DAYS - 1)) * Math.PI);
      var noise = (rand() - 0.5) * 10.0; // +/- ~1.5 degC
      temps[i] = base + noise;
    }
    return temps;
  }

  // ---------------------------------------------------------------------------
  // Canvas rendering helpers
  // ---------------------------------------------------------------------------
  var COLORS = {
    notHot: "#9aa5b1",
    hot: "#f6a623",
    heatwave: "#d0021b",
    line: "#2a6592",
    threshold: "#444",
    axis: "#888",
    grid: "#e3e8ee"
  };

  var PAD = { left: 48, right: 16, top: 16, bottom: 28 };

  function setupCanvas(canvas) {
    var ratio = window.devicePixelRatio || 1;
    var cssWidth = canvas.clientWidth;
    var cssHeight = canvas.clientHeight;
    canvas.width = Math.round(cssWidth * ratio);
    canvas.height = Math.round(cssHeight * ratio);
    var ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, cssWidth, cssHeight);
    return { ctx: ctx, w: cssWidth, h: cssHeight };
  }

  function xForDay(i, w) {
    var inner = w - PAD.left - PAD.right;
    return PAD.left + (inner * i) / (N_DAYS - 1);
  }

  function drawTempPane(canvas, temps, threshold) {
    var s = setupCanvas(canvas);
    var ctx = s.ctx;
    var yMin = 24;
    var yMax = 44;
    var inner = s.h - PAD.top - PAD.bottom;
    function yFor(t) {
      return PAD.top + inner * (1 - (t - yMin) / (yMax - yMin));
    }

    // Gridlines + y-axis labels (degC).
    ctx.strokeStyle = COLORS.grid;
    ctx.fillStyle = COLORS.axis;
    ctx.font = "11px sans-serif";
    ctx.lineWidth = 1;
    for (var t = yMin; t <= yMax; t += 5) {
      var y = yFor(t);
      ctx.beginPath();
      ctx.moveTo(PAD.left, y);
      ctx.lineTo(s.w - PAD.right, y);
      ctx.stroke();
      ctx.fillText(t + "°C", 8, y + 3);
    }

    // Threshold line.
    ctx.strokeStyle = COLORS.threshold;
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.moveTo(PAD.left, yFor(threshold));
    ctx.lineTo(s.w - PAD.right, yFor(threshold));
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = COLORS.threshold;
    ctx.fillText("threshold " + threshold.toFixed(1) + "°C", PAD.left + 4, yFor(threshold) - 5);

    // Temperature line.
    ctx.strokeStyle = COLORS.line;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (var i = 0; i < N_DAYS; i++) {
      var px = xForDay(i, s.w);
      var py = yFor(temps[i]);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();

    // Point markers colored by hot / not-hot.
    for (var j = 0; j < N_DAYS; j++) {
      ctx.fillStyle = temps[j] > threshold ? COLORS.hot : COLORS.notHot;
      ctx.beginPath();
      ctx.arc(xForDay(j, s.w), yFor(temps[j]), 4, 0, 2 * Math.PI);
      ctx.fill();
    }

    drawDayAxis(ctx, s);
  }

  function drawStepPane(canvas, steps) {
    var s = setupCanvas(canvas);
    var ctx = s.ctx;
    var inner = s.h - PAD.top - PAD.bottom;
    var levels = [0, 1, 2];
    function yFor(level) {
      return PAD.top + inner * (1 - level / 2);
    }

    // Gridlines + level labels.
    ctx.strokeStyle = COLORS.grid;
    ctx.fillStyle = COLORS.axis;
    ctx.font = "11px sans-serif";
    var labels = ["0 NH", "1 H", "2 HW"];
    for (var l = 0; l < levels.length; l++) {
      var y = yFor(levels[l]);
      ctx.beginPath();
      ctx.moveTo(PAD.left, y);
      ctx.lineTo(s.w - PAD.right, y);
      ctx.stroke();
      ctx.fillText(labels[l], 4, y + 3);
    }

    // Colored bars per day.
    var innerW = s.w - PAD.left - PAD.right;
    var barW = (innerW / N_DAYS) * 0.7;
    for (var i = 0; i < N_DAYS; i++) {
      var level = steps[i];
      var color = level === 2 ? COLORS.heatwave : level === 1 ? COLORS.hot : COLORS.notHot;
      var cx = xForDay(i, s.w);
      var top = yFor(level);
      var bottom = yFor(0);
      ctx.fillStyle = color;
      if (level === 0) {
        // Draw a thin marker at the baseline so "not hot" days stay visible.
        ctx.fillRect(cx - barW / 2, bottom - 2, barW, 2);
      } else {
        ctx.fillRect(cx - barW / 2, top, barW, bottom - top);
      }
    }

    drawDayAxis(ctx, s);
  }

  function drawDayAxis(ctx, s) {
    ctx.strokeStyle = COLORS.axis;
    ctx.fillStyle = COLORS.axis;
    ctx.lineWidth = 1;
    ctx.font = "11px sans-serif";
    var y = s.h - PAD.bottom;
    ctx.beginPath();
    ctx.moveTo(PAD.left, y);
    ctx.lineTo(s.w - PAD.right, y);
    ctx.stroke();
    for (var i = 0; i < N_DAYS; i++) {
      var px = xForDay(i, s.w);
      ctx.fillText(String(i + 1), px - 3, y + 16);
    }
  }

  // ---------------------------------------------------------------------------
  // Widget wiring
  // ---------------------------------------------------------------------------
  function initWidget(root) {
    var els = {
      tempCanvas: root.querySelector("#hdp-hw-temp"),
      stepCanvas: root.querySelector("#hdp-hw-step"),
      threshold: root.querySelector("#hdp-hw-threshold"),
      minDuration: root.querySelector("#hdp-hw-min-duration"),
      maxBreak: root.querySelector("#hdp-hw-max-break"),
      maxSubs: root.querySelector("#hdp-hw-max-subs"),
      thresholdVal: root.querySelector("#hdp-hw-threshold-val"),
      minDurationVal: root.querySelector("#hdp-hw-min-duration-val"),
      maxBreakVal: root.querySelector("#hdp-hw-max-break-val"),
      maxSubsVal: root.querySelector("#hdp-hw-max-subs-val"),
      regenerate: root.querySelector("#hdp-hw-regenerate"),
      metrics: root.querySelector("#hdp-hw-metrics")
    };

    var seed = 1234;
    var temps = generateTemps(seed);

    function render() {
      var threshold = parseFloat(els.threshold.value);
      var minDuration = parseInt(els.minDuration.value, 10);
      var maxBreak = parseInt(els.maxBreak.value, 10);
      var maxSubs = parseInt(els.maxSubs.value, 10);

      els.thresholdVal.textContent = threshold.toFixed(1) + "°C";
      els.minDurationVal.textContent = minDuration + " day" + (minDuration === 1 ? "" : "s");
      els.maxBreakVal.textContent = String(maxBreak);
      els.maxSubsVal.textContent = String(maxSubs);

      var hotDays = new Int32Array(N_DAYS);
      for (var i = 0; i < N_DAYS; i++) hotDays[i] = temps[i] > threshold ? 1 : 0;

      var hwIndex = indexHeatwaves(hotDays, minDuration, maxBreak, maxSubs);
      var steps = new Int32Array(N_DAYS);
      for (var j = 0; j < N_DAYS; j++) {
        steps[j] = hotDays[j] + (hwIndex[j] > 0 ? 1 : 0);
      }

      drawTempPane(els.tempCanvas, temps, threshold);
      drawStepPane(els.stepCanvas, steps);

      var m = computeMetrics(hwIndex);
      els.metrics.innerHTML =
        metricCell("HWN", m.HWN, "events") +
        metricCell("HWF", m.HWF, "heatwave days") +
        metricCell("HWD", m.HWD, "longest event, days") +
        metricCell("HWA", m.HWA.toFixed(1), "mean event, days");
    }

    function metricCell(name, value, desc) {
      return (
        '<div class="hdp-hw-metric"><span class="hdp-hw-metric-name">' +
        name +
        '</span><span class="hdp-hw-metric-value">' +
        value +
        '</span><span class="hdp-hw-metric-desc">' +
        desc +
        "</span></div>"
      );
    }

    [els.threshold, els.minDuration, els.maxBreak, els.maxSubs].forEach(function (slider) {
      slider.addEventListener("input", render);
    });

    els.regenerate.addEventListener("click", function () {
      seed = (Math.random() * 1e9) | 0;
      temps = generateTemps(seed);
      render();
    });

    window.addEventListener("resize", render);
    render();
  }

  function boot() {
    var root = document.getElementById("hdp-hw-widget");
    if (!root) return; // No-op on pages without the widget.
    initWidget(root);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
