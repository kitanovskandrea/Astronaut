"""Create self-contained SVG experiment figures using the standard library.

Run ``python -m astronaut_rl.plots --results results --out results/figures``.
The manifest and raw CSVs are the data sources. Curves, error bars and bands
show means +/- sample SD across independent training policies. Episodes within
a policy are averaged first; SD is not a confidence interval.
"""

import argparse
import csv
import html
import json
import math
from pathlib import Path
import statistics

from .report import aggregate, groups_for, training_curves


ORDER = ("fly", "asteroids", "oxygen")
LABELS = {"fly": "Flight", "asteroids": "+ Asteroids", "oxygen": "+ Oxygen"}
COLORS = {"fly": "#2867c7", "asteroids": "#008d82", "oxygen": "#d55b36"}
WIDTH, HEIGHT = 1200, 740
INK, MUTED, GRID = "#172d45", "#526778", "#e2e9ef"


def _number(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("%s must be finite" % name)
    return value


def _mean(values):
    return statistics.fmean(values)


def _nice_ceiling(value):
    if value <= 0:
        return 1.0
    scale = 10 ** math.floor(math.log10(value))
    normalized = value / scale
    return next(step * scale for step in (1, 2, 2.5, 5, 10) if step >= normalized)


def _label(value):
    if abs(value) >= 1000000:
        return "%gM" % (value / 1000000)
    if abs(value) >= 1000:
        return "%gk" % (value / 1000)
    return "%g" % value


class SVG:
    def __init__(self, title, subtitle):
        self.parts = [
            '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="740" viewBox="0 0 1200 740" role="img">',
            '<title>%s</title>' % html.escape(title),
            '<desc>%s</desc>' % html.escape(subtitle),
            '<rect width="1200" height="740" fill="#fff"/>',
            '<style>text{font-family:Arial,Helvetica,sans-serif;fill:%s} '
            '.title{font-size:28px;font-weight:700} .subtitle{font-size:16px;fill:%s} '
            '.axis{font-size:15px;fill:%s} .small{font-size:14px;fill:%s} '
            '.value{font-size:17px;font-weight:700} .panel{font-size:19px;font-weight:700}</style>'
            % (INK, MUTED, MUTED, MUTED),
        ]
        self.text(52, 55, title, cls="title")
        self.text(52, 87, subtitle, cls="subtitle")

    def add(self, content):
        self.parts.append(content)

    def text(self, x, y, text, cls="axis", anchor="start", fill=None, rotate=None):
        attrs = 'x="%g" y="%g" class="%s" text-anchor="%s"' % (x, y, cls, anchor)
        if fill:
            attrs += ' style="fill:%s"' % fill
        if rotate is not None:
            attrs += ' transform="rotate(%g %g %g)"' % (rotate, x, y)
        self.add('<text %s>%s</text>' % (attrs, html.escape(str(text))))

    def line(self, x1, y1, x2, y2, stroke=GRID, width=1, dash=None):
        dashed = ' stroke-dasharray="%s"' % dash if dash else ""
        self.add('<line x1="%g" y1="%g" x2="%g" y2="%g" stroke="%s" stroke-width="%g"%s/>'
                 % (x1, y1, x2, y2, stroke, width, dashed))

    def rect(self, x, y, width, height, fill, opacity=1):
        self.add('<rect x="%g" y="%g" width="%g" height="%g" fill="%s" opacity="%g"/>'
                 % (x, y, width, height, fill, opacity))

    def save(self, path):
        Path(path).write_text("\n".join(self.parts + ["</svg>"]) + "\n", encoding="utf-8")


def _band(svg, points, xmap, ymap, color, lower=0.0, upper=None):
    def bound(value):
        return max(lower, min(upper, value)) if upper is not None else max(lower, value)
    high = [(xmap(point["step"]), ymap(bound(point["mean"] + point["sd"]))) for point in points]
    low = [(xmap(point["step"]), ymap(bound(point["mean"] - point["sd"]))) for point in reversed(points)]
    svg.add('<polygon points="%s" fill="%s" opacity="0.14"/>' %
            (" ".join("%.2f,%.2f" % p for p in high + low), color))
    svg.add('<polyline points="%s" fill="none" stroke="%s" stroke-width="2.7" stroke-linejoin="round"/>' %
            (" ".join("%.2f,%.2f" % (xmap(point["step"]), ymap(point["mean"])) for point in points), color))
    if len(points) <= 12:
        for point in points:
            svg.add('<circle cx="%.2f" cy="%.2f" r="3.3" fill="%s"/>' %
                    (xmap(point["step"]), ymap(point["mean"]), color))


def _axes(svg, maximum_x, maximum_y, ylabel, percent=False):
    left, top, width, height = 112, 152, 1030, 430
    xmap = lambda value: left + width * value / maximum_x
    ymap = lambda value: top + height * (1 - value / maximum_y)
    for index in range(6):
        value = maximum_y * index / 5
        y = ymap(value)
        svg.line(left, y, left + width, y)
        svg.text(left - 12, y + 5, ("%g%%" % value) if percent else _label(value), anchor="end")
    svg.line(left, top, left, top + height, stroke="#8ea1b2")
    svg.text(left - 68, top + height / 2, ylabel, anchor="middle", rotate=-90)
    for index in range(6):
        value = maximum_x * index / 5
        x = xmap(value)
        svg.line(x, top + height, x, top + height + 6, stroke="#8ea1b2")
        svg.text(x, top + height + 28, _label(value), anchor="middle")
    svg.text(left + width / 2, top + height + 65, "Q decisions (up to six actual physics frames per decision)", anchor="middle")
    return xmap, ymap


def _legend(svg, variants):
    x = 112
    for variant in variants:
        svg.line(x, 119, x + 28, 119, stroke=COLORS[variant], width=3)
        svg.text(x + 38, 124, LABELS[variant])
        x += 218
    svg.text(1142, 124, "Band: mean +/- sample SD", cls="small", anchor="end")


def curve_plot(series, title, ylabel, path, maximum=None, percent=False, footer=None):
    if not series or any(not points for points in series.values()):
        raise ValueError("A curve needs observations for every plotted variant")
    maximum_x = max(point["step"] for points in series.values() for point in points)
    maximum_y = maximum or _nice_ceiling(max(point["mean"] + point["sd"] for points in series.values() for point in points) * 1.05)
    counts = sorted({point["n"] for points in series.values() for point in points})
    svg = SVG(title, "Independent curriculum-chain means; n = " + ", ".join(map(str, counts)) + "; official rewards unchanged")
    xmap, ymap = _axes(svg, maximum_x, maximum_y, ylabel, percent)
    _legend(svg, series)
    for variant, points in series.items():
        _band(svg, points, xmap, ymap, COLORS[variant], upper=maximum)
    svg.text(52, 687, footer or "Frozen greedy checkpoint evaluations on validation seeds; final testing uses separate unseen seeds.", cls="small")
    svg.text(52, 712, "Sample SD describes variation across independent chains, not confidence; bands are clipped at physical bounds.", cls="small")
    svg.save(path)


def performance_plot(summary, path):
    svg = SVG("Validation-selected policies on unseen test scenarios", "Episodes averaged within each policy, then mean +/- sample SD across independent chains")
    svg.rect(803, 110, 19, 14, "#2867c7")
    svg.text(831, 123, "Selected Q policy", cls="small")
    svg.rect(1000, 110, 19, 14, "#b7c5d2")
    svg.text(1028, 123, "Random", cls="small")
    variants = list(summary["variants"])
    for panel_index, (title, metric, scale, maximum) in enumerate((
            ("Full-episode completion", "survival_rate", 100, 100),
            ("Oxygen collection", "oxygen_mean", 1, None))):
        left, top, width, height = 110 + panel_index * 600, 195, 420, 365
        values = [entry[policy][metric] for entry in summary["variants"].values() for policy in ("greedy", "random")]
        maximum = maximum or _nice_ceiling(max((v["mean"] + v["sd"]) * scale for v in values) * 1.25)
        svg.text(left + width / 2, 164, title, cls="panel", anchor="middle")
        for index in range(6):
            v = maximum * index / 5
            yy = top + height * (1 - index / 5)
            svg.line(left, yy, left + width, yy)
            svg.text(left - 12, yy + 5, ("%g%%" % v) if scale == 100 else _label(v), anchor="end")
        svg.text(left - 62, top + height / 2, "Completion (%)" if scale == 100 else "Tanks / episode", anchor="middle", rotate=-90)
        ymap = lambda v: top + height * (1 - v / maximum)
        group_width = width / len(variants)
        bar_width = min(43, group_width * .29)
        for index, variant in enumerate(variants):
            center = left + group_width * (index + .5)
            for offset, policy, color in ((-bar_width - 3, "greedy", "#2867c7"), (3, "random", "#b7c5d2")):
                stat = summary["variants"][variant][policy][metric]
                value = stat["mean"] * scale
                low, high = max(0, (stat["mean"] - stat["sd"]) * scale), min(maximum, (stat["mean"] + stat["sd"]) * scale)
                xx = center + offset
                svg.rect(xx, ymap(value), bar_width, height * value / maximum, color)
                svg.line(xx + bar_width / 2, ymap(low), xx + bar_width / 2, ymap(high), stroke=INK, width=1.7)
                for bound in (low, high):
                    svg.line(xx + bar_width / 2 - 5, ymap(bound), xx + bar_width / 2 + 5, ymap(bound), stroke=INK, width=1.7)
                svg.text(xx + bar_width / 2, max(top - 6, ymap(high) - 12), "%.1f" % value if scale == 100 else "%.2f" % value, cls="small", anchor="middle")
            svg.text(center, top + height + 32, LABELS[variant], anchor="middle")
    svg.text(52, 652, "Main model = checkpoint selected with validation only; final test outcomes do not select the model.", cls="small")
    svg.text(52, 678, "Flight: one deterministic test episode. Asteroids/oxygen: 100 paired test seeds per policy.", cls="small")
    svg.text(52, 704, "Independent training chains are the replicates. Repeated random baselines are the same paired seed set.", cls="small")
    svg.save(path)


def time_plot(summary, path):
    svg = SVG("Training cost of the learned curriculum", "Mean +/- sample SD across independent chains; stage allocations differ")
    for panel_index, (key, title) in enumerate((("stage_cost", "Incremental stage training"), ("cumulative_cost", "Including prerequisite training"))):
        left, top, width, height = 110 + panel_index * 600, 195, 420, 365
        maximum = _nice_ceiling(max(entry[key]["train_seconds"]["mean"] + entry[key]["train_seconds"]["sd"] for entry in summary["variants"].values()) * 1.22)
        ymap = lambda value: top + height * (1 - value / maximum)
        svg.text(left + width / 2, 156, title, cls="panel", anchor="middle")
        for index in range(6):
            value = maximum * index / 5
            yy = ymap(value)
            svg.line(left, yy, left + width, yy)
            svg.text(left - 12, yy + 5, _label(value), anchor="end")
        svg.text(left - 62, top + height / 2, "Training seconds", anchor="middle", rotate=-90)
        group_width = width / len(summary["variants"])
        for index, (variant, entry) in enumerate(summary["variants"].items()):
            stat = entry[key]["train_seconds"]
            center = left + group_width * (index + .5)
            bar_width = min(67, group_width * .52)
            low, high = max(0, stat["mean"] - stat["sd"]), stat["mean"] + stat["sd"]
            svg.rect(center - bar_width / 2, ymap(stat["mean"]), bar_width, height * stat["mean"] / maximum, COLORS[variant])
            svg.line(center, ymap(low), center, ymap(high), stroke=INK, width=1.7)
            for value in (low, high):
                svg.line(center - 6, ymap(value), center + 6, ymap(value), stroke=INK, width=1.7)
            svg.text(center, ymap(high) - 15, "%.1f +/- %.1f" % (stat["mean"], stat["sd"]), cls="small", anchor="middle")
            svg.text(center, top + height + 31, LABELS[variant], anchor="middle")
            svg.text(center, top + height + 59, _label(entry["budget"]["mean"]) + " decisions", cls="small", anchor="middle")
    svg.text(52, 652, "Flight starts from zero; asteroid and oxygen stages inherit learned values from the same chain's selected parent.", cls="small")
    svg.text(52, 678, "Different budgets and transferred learning: this is not an equal-budget from-zero comparison or time to convergence.", cls="small")
    svg.text(52, 704, "Timers exclude validation/testing and export. Cumulative cost includes full allocated prerequisite training.", cls="small")
    svg.save(path)


def build_all(results, out):
    """Read primary official-reward records and create seven measured figures."""
    results, out = Path(results), Path(out)
    summary = aggregate(results)
    manifest = json.loads((results / "run_manifest.json").read_text(encoding="utf-8"))
    groups = groups_for(manifest["records"])
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / name for name in ("loss.svg", "survival.svg", "completion.svg", "oxygen.svg", "coverage.svg", "performance.svg", "training_time.svg")]
    curve_plot(training_curves(results, groups), "TD residual during learning", "Mean squared TD residual", paths[0],
               footer="Pre-update squared Bellman residuals averaged in fixed-decision blocks; low zero-reward loss can coexist with poor play.")
    for path, title, metric, ylabel, scale, maximum, variants in (
            (paths[1], "Validation survival learning curve", "frames_mean", "Survived physics steps", 1, 1200, tuple(groups)),
            (paths[2], "Validation full-episode success curve", "survival_rate", "Completion (%)", 100, 100, tuple(groups)),
            (paths[3], "Validation oxygen collection curve", "oxygen_mean", "Tanks / episode", 1, None, ("oxygen",))):
        selected = {variant: [{"step": point["step"], **{key: value * scale if key in ("mean", "sd", "min", "max") else value for key, value in point[metric].items()}}
                              for point in summary["validation_curves"][variant]] for variant in variants if variant in groups}
        if selected:
            curve_plot(selected, title, ylabel, path, maximum=maximum, percent=scale == 100)
    curve_plot(training_curves(results, groups, "visited_states"), "Q-table visitation during training", "Distinct updated states", paths[4],
               footer="Updated states count positive child visits; inherited Q rows can be stored with zero child updates.")
    performance_plot(summary, paths[5])
    time_plot(summary, paths[6])
    return [path for path in paths if path.exists()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("results/figures"))
    args = parser.parse_args()
    try:
        for path in build_all(args.results, args.out):
            print(path)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, "Cannot generate experiment figures: %s\n" % error)


if __name__ == "__main__":
    main()
