"""Historical unassessed metrics must not become zero-valued observations."""
import unittest
import numpy as np
import matplotlib.pyplot as plt
from scripts.plot_v2_research import draw_control_curve, format_numeric_ticks


class ControlPlotTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_all_unavailable_is_labeled_without_zero_imputation(self):
        _, ax = plt.subplots()
        draw_control_curve(ax, [{"x": 1, "metric": None}, {"x": 2, "metric": None}],
                           "x", "metric", "Unassessed", "blue")
        self.assertTrue(ax.texts)
        self.assertTrue(all(len(line.get_xdata()) == 0 for line in ax.lines))

    def test_partial_unavailable_leaves_a_gap_and_counts_missing(self):
        _, ax = plt.subplots()
        rows = [{"x": 1, "metric": 2.}, {"x": 1, "metric": None},
                {"x": 2, "metric": None}, {"x": 3, "metric": 4.}]
        draw_control_curve(ax, rows, "x", "metric", "Known", "blue")
        values = ax.lines[0].get_ydata()
        self.assertEqual(values[0], 2.)
        self.assertTrue(np.isnan(values[1]))
        self.assertEqual(values[2], 4.)
        self.assertIn("2/4 unavailable", ax.lines[0].get_label())

    def test_boolean_fraction_and_nonfinite_recorded_values(self):
        _, ax = plt.subplots()
        draw_control_curve(ax, [{"x": 1, "interpolating": True}, {"x": 1, "interpolating": False}],
                           "x", "interpolating", "Measured", "blue")
        self.assertEqual(ax.lines[0].get_ydata()[0], .5)
        with self.assertRaises(ValueError):
            draw_control_curve(ax, [{"x": 1, "metric": float("inf")}], "x", "metric", "Bad", "blue")

    def test_small_gain_tick_labels_do_not_overlap_after_rendering(self):
        # Same panel geometry and long row labels as the late-gate figure;
        # deterministic small values isolate layout from scientific artifacts.
        with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                             "axes.formatter.limits": (-5, 6),
                             "axes.formatter.use_mathtext": False}):
            fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=155, constrained_layout=True)
            labels = [f"Synthetic (new data per seed)\nrandom_strength={i}; positive gains 4/12" for i in range(6)]
            for ax in axes:
                ax.errorbar(np.linspace(-.00008, .00004, 6), range(6), xerr=.00001, fmt="o")
                ax.set_xticks(np.linspace(-.0001, .00006, 9))
                ax.set_yticks(range(6), labels)
                ax.set_xlabel("Fixed-window train descent-speed gain")

            def overlapping_ticks():
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                overlap_count = 0
                for ax in axes:
                    lo, hi = sorted(ax.get_xlim())
                    boxes = [label.get_window_extent(renderer) for value, label in zip(ax.get_xticks(), ax.get_xticklabels())
                             if lo <= value <= hi and label.get_visible() and label.get_text()]
                    self.assertGreaterEqual(len(boxes), 3)
                    boxes.sort(key=lambda box: box.x0)
                    overlap_count += sum(left.overlaps(right) for left, right in zip(boxes, boxes[1:]))
                return overlap_count

            self.assertGreater(overlapping_ticks(), 0, "Regression fixture must reproduce the long-decimal collision")
            format_numeric_ticks(fig)
            self.assertEqual(overlapping_ticks(), 0)
            self.assertTrue(all(ax.xaxis.get_offset_text().get_text() for ax in axes))

    def test_shared_numeric_formatter_preserves_categorical_and_log_axes(self):
        fig, axes = plt.subplots(1, 2)
        axes[0].set_xticks([0, 1], ["first", "second"])
        axes[1].set_yscale("log")
        categorical = axes[0].xaxis.get_major_formatter()
        logarithmic = axes[1].yaxis.get_major_formatter()
        format_numeric_ticks(fig)
        self.assertIs(axes[0].xaxis.get_major_formatter(), categorical)
        self.assertIs(axes[1].yaxis.get_major_formatter(), logarithmic)
        self.assertEqual([label.get_text() for label in axes[0].get_xticklabels()], ["first", "second"])
