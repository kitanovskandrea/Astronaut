"""Build the A1 landscape conference poster from measured results, using stdlib.

This small vector-PDF writer keeps the submitted program free of extra plotting
or document libraries. It uses the standard PDF base fonts and no raster plots.
"""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from .report import aggregate, groups_for, training_curves

W, H = 2383.94, 1683.78
NAVY, INK, MUTED = "142C42", "183247", "526777"
BG, GRID, WHITE = "F2F5F8", "DCE4EA", "FFFFFF"
COLORS = {"fly": "2867C7", "asteroids": "008D82", "oxygen": "D55B36"}
LABELS = {"fly": "Flight", "asteroids": "+ Asteroids", "oxygen": "+ Oxygen"}


def rgb(h):
    return " ".join(f"{int(h[i:i+2],16)/255:.5f}" for i in (0, 2, 4))


class PDF:
    def __init__(self):
        self.commands = []

    def rect(self, x, y, w, h, color):
        self.commands.append(f"{rgb(color)} rg {x:.2f} {H-y-h:.2f} {w:.2f} {h:.2f} re f")

    def line(self, x1, y1, x2, y2, color=GRID, width=1):
        self.commands.append(f"{rgb(color)} RG {width:.2f} w {x1:.2f} {H-y1:.2f} m {x2:.2f} {H-y2:.2f} l S")

    def path(self, points, color, width=2, fill=False):
        if not points:
            return
        text = f"{rgb(color)} {'rg' if fill else 'RG'} {width:.2f} w "
        text += f"{points[0][0]:.2f} {H-points[0][1]:.2f} m "
        text += " ".join(f"{x:.2f} {H-y:.2f} l" for x, y in points[1:])
        self.commands.append(text + (" h f" if fill else " S"))

    def text(self, x, y, value, size=24, font="F1", color=INK):
        value = str(value).encode("cp1252", errors="strict").decode("latin1")
        value = value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        self.commands.append(f"BT /{font} {size:.2f} Tf {rgb(color)} rg 1 0 0 1 {x:.2f} {H-y:.2f} Tm ({value}) Tj ET")

    def para(self, x, y, text, width, size=24, leading=33, color=INK, font="F1"):
        # Conservative proportional-width estimates: tight text is checked in QA.
        def weight(s):
            return sum(0.25 if c in " il.,:;'!|" else 0.85 if c in "MW@%" else 0.54 for c in s) * size
        for paragraph in text.split("\n"):
            words, line = paragraph.split(), ""
            for word in words:
                candidate = line + (" " if line else "") + word
                if line and weight(candidate) > width:
                    self.text(x, y, line, size, font, color)
                    y += leading
                    line = word
                else:
                    line = candidate
            if line:
                self.text(x, y, line, size, font, color)
                y += leading
        return y

    def save(self, path):
        stream = ("\n".join(self.commands) + "\n").encode("latin1")
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {W} {H}] "
             "/Resources << /Font << /F1 4 0 R /F2 5 0 R /F3 6 0 R /F4 7 0 R >> >> /Contents 8 0 R >>").encode(),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Oblique /Encoding /WinAnsiEncoding >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier /Encoding /WinAnsiEncoding >>",
            f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"endstream",
            b"<< /Title (Reinforcement learning with the Astronaut) /Author (Andrea Kitanovska; Hristijan Kochovski) /Subject (Group 7 - Intelligent Systems Seminar 1) >>",
        ]
        data = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = [0]
        for i, obj in enumerate(objects, 1):
            offsets.append(len(data))
            data.extend(f"{i} 0 obj\n".encode() + obj + b"\nendobj\n")
        xref = len(data)
        data.extend(f"xref\n0 {len(objects)+1}\n0000000000 65535 f \n".encode())
        for offset in offsets[1:]:
            data.extend(f"{offset:010d} 00000 n \n".encode())
        data.extend((f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R /Info 9 0 R >>\n"
                     f"startxref\n{xref}\n%%EOF\n").encode())
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(data)


def avg(values):
    return statistics.fmean(values)


def heading(pdf, x, y, number, title):
    pdf.rect(x, y - 27, 43, 36, NAVY)
    pdf.text(x + 7, y, number, 23, "F2", WHITE)
    pdf.text(x + 60, y, title, 30, "F2")


def pale(color, strength=.16):
    return "".join("%02X" % round(int(color[i:i+2], 16) * strength + 255 * (1-strength)) for i in (0, 2, 4))


def number(value):
    if abs(value) >= 1000000:
        return "%gM" % (value / 1000000)
    return "%gk" % (value / 1000) if abs(value) >= 1000 else "%g" % value


def stat_text(stat, digits=1, scale=1):
    return ("{:.%df} +/- {:.%df}" % (digits, digits)).format(stat["mean"] * scale, stat["sd"] * scale)


def chart(pdf, series, x, y, width, height, title, maximum=None):
    """Small vector plot with physically clipped mean +/- sample-SD bands."""
    max_x = max(point["step"] for points in series.values() for point in points)
    max_y = maximum or max(.00001, max(point["mean"] + point["sd"] for points in series.values() for point in points) * 1.08)
    left, top, pw, ph = x + 70, y + 36, width - 94, height - 94
    xm = lambda value: left + pw * value / max_x
    ym = lambda value: top + ph * (1 - value / max_y)
    pdf.text(x + 12, y + 4, title, 23, "F2")
    for index in range(4):
        value = max_y * index / 3
        yy = ym(value)
        pdf.line(left, yy, left + pw, yy)
        label = "%.3f" % value if max_y < .1 else number(round(value, 1))
        pdf.text(left - 64, yy + 6, label, 18, color=MUTED)
    for index in range(6):
        value = max_x * index / 5
        xx = xm(value)
        pdf.line(xx, top + ph, xx, top + ph + 5, MUTED)
        pdf.text(xx - 13, top + ph + 27, number(value), 18, color=MUTED)
    pdf.text(left + pw / 2 - 120, y + height - 2, "Q decisions (up to 6 physics frames)", 19, color=MUTED)
    for variant, points in series.items():
        hi = [(xm(point["step"]), ym(min(max_y, point["mean"] + point["sd"]))) for point in points]
        lo = [(xm(point["step"]), ym(max(0, point["mean"] - point["sd"]))) for point in reversed(points)]
        pdf.path(hi + lo, pale(COLORS[variant]), fill=True)
    for variant, points in series.items():
        pdf.path([(xm(point["step"]), ym(point["mean"])) for point in points], COLORS[variant], 3)


def timing_chart(pdf, variants, x, y, width):
    maximum = max(entry["cumulative_cost"]["train_seconds"]["mean"] + entry["cumulative_cost"]["train_seconds"]["sd"] for entry in variants.values()) * 1.05
    for index, (variant, entry) in enumerate(variants.items()):
        yy = y + index * 90
        pdf.text(x, yy, LABELS[variant], 22, "F2", COLORS[variant])
        incremental, cumulative = entry["stage_cost"]["train_seconds"], entry["cumulative_cost"]["train_seconds"]
        pdf.text(x + 205, yy, "%.1fs / %.1fs" % (incremental["mean"], cumulative["mean"]), 22, "F2")
        for offset, stat, color in ((12, cumulative, pale(COLORS[variant], .35)), (34, incremental, COLORS[variant])):
            pdf.rect(x, yy + offset, width, 14, BG)
            pdf.rect(x, yy + offset, width * stat["mean"] / maximum, 14, color)
            lo, hi = max(0, stat["mean"] - stat["sd"]), stat["mean"] + stat["sd"]
            pdf.line(x + width * lo / maximum, yy + offset + 7, x + width * hi / maximum, yy + offset + 7, INK, 2)
            for value in (lo, hi):
                xx = x + width * value / maximum
                pdf.line(xx, yy + offset + 2, xx, yy + offset + 12, INK, 1.5)


def _validation_series(summary, metric, variants=None):
    return {variant: [{"step": point["step"], **point[metric]} for point in points]
            for variant, points in summary["validation_curves"].items()
            if variants is None or variant in variants}


def build(results, out):
    results = Path(results)
    manifest = json.loads((results / "run_manifest.json").read_text(encoding="utf-8"))
    summary = aggregate(results)
    groups = groups_for(manifest["records"])
    variants = summary["variants"]
    if set(variants) != {"fly", "asteroids", "oxygen"} or any(entry["n"] != 5 for entry in variants.values()):
        raise ValueError("Primary poster requires five official-reward runs for each variant")
    if any(int(row["repeat"]) != 6 for row in manifest["records"]):
        raise ValueError("Curriculum poster requires the frozen six-frame action interval")
    if any(not row.get("qualified", False) for row in manifest["records"]):
        raise ValueError("Final poster requires all five chains to qualify at every stage; inspect the failed-gate report")
    budgets = {variant: {int(row["budget"]) for row in rows} for variant, rows in groups.items()}
    if any(len(values) != 1 for values in budgets.values()):
        raise ValueError("Allocated budgets must match across chains within a stage")
    budget_labels = {variant: number(next(iter(values))) for variant, values in budgets.items()}
    pdf = PDF()
    pdf.rect(0, 0, W, H, BG)
    pdf.rect(0, 0, W, 265, NAVY)
    pdf.text(65, 53, "INTELLIGENT SYSTEMS  /  SEMINAR 1", 24, "F2", "8FD9E1")
    pdf.text(2153, 53, "GROUP 7", 24, "F2", WHITE)
    pdf.text(65, 139, "Reinforcement learning with the Astronaut", 76, "F2", WHITE)
    pdf.text(68, 191, "Tabular Q-learning with official rewards, learned skill transfer and survival-gated checkpoints", 31, color="CEDDE8")
    pdf.text(68, 238, "Andrea Kitanovska  (89262058)   |   Hristijan Kochovski  (89262125)", 28, "F2", WHITE)
    pdf.text(1960, 238, "UP FAMNIT  |  2026", 24, color="CEDDE8")

    card_w = (W - 130 - 44) / 3
    for index, (variant, metric, scale, label) in enumerate((
            ("fly", "survival_rate", 100, "full-episode completion (%)"),
            ("asteroids", "survival_rate", 100, "full-episode completion (%)"),
            ("oxygen", "oxygen_mean", 1, "oxygen tanks / episode"))):
        xx = 65 + index * (card_w + 22)
        pdf.rect(xx, 291, card_w, 139, WHITE)
        pdf.rect(xx, 291, 8, 139, COLORS[variant])
        pdf.text(xx + 24, 326, LABELS[variant].upper(), 23, "F2", COLORS[variant])
        pdf.text(xx + 24, 379, stat_text(variants[variant]["greedy"][metric], 1 if scale == 100 else 2, scale), 45, "F2")
        pdf.text(xx + 24, 412, label + "  |  mean +/- sample SD", 21, color=MUTED)
    pdf.text(68, 467, "Q decisions: flight %s / asteroids %s / oxygen %s  |  5 independent chains  |  Validation gates  |  Unseen final tests" % (budget_labels["fly"], budget_labels["asteroids"], budget_labels["oxygen"]), 24, color=MUTED)

    lx, mx, rx = 65, 750, 1685
    lw, mw, rw = 650, 900, 634
    for xx, width in ((lx, lw), (mx, mw), (rx, rw)):
        pdf.rect(xx, 490, width, 1048, WHITE)
    heading(pdf, lx + 22, 542, "01", "Method and experiment")
    pdf.para(lx + 23, 590,
             "Learn flight, extend its Q values to asteroid avoidance, then extend learned avoidance to oxygen. The course simulator is unchanged.", lw - 46, 24, 31)
    pdf.text(lx + 23, 713, "PUBLIC OBSERVATIONS -> DISCRETE STATE", 22, "F2", MUTED)
    for index, variant in enumerate(variants):
        yy = 758 + index * 64
        pdf.rect(lx + 23, yy - 26, lw - 46, 54, BG)
        pdf.text(lx + 35, yy + 8, LABELS[variant], 24, "F2", COLORS[variant])
        pdf.text(lx + 355, yy + 8, "%s states" % format(variants[variant]["state_capacity"], ","), 24, "F2")
    pdf.para(lx + 23, 944,
             "Public height, velocity, arrival time and motion-projected offsets. Oxygen keeps the full hazard prefix. Copied rows may have zero child visits.", lw - 46, 22, 29)
    pdf.text(lx + 23, 1064, "CLASSICAL Q-LEARNING", 22, "F2", MUTED)
    pdf.text(lx + 23, 1105, "target = R + g^k max Q(s',a')", 23, "F4")
    pdf.text(lx + 23, 1143, "Q(s,a) <- Q(s,a) + alpha*(target-Q(s,a))", 20, "F4")
    pdf.para(lx + 23, 1184,
             "R sums discounted official rewards over k <= 6 frames. Use g^k for actual executed frames. Collision: no bootstrap; time limit: bootstrap.", lw - 46, 22, 29)
    settings = groups["fly"][0]
    alpha_values = {row["alpha"] for row in manifest["records"]}
    gamma_values = {row["gamma"] for row in manifest["records"]}
    config = settings.get("config", {})
    alpha_label = "%g" % next(iter(alpha_values)) if len(alpha_values) == 1 else "see stage configs"
    if len(alpha_values) > 1 and groups["asteroids"][0]["alpha"] == groups["oxygen"][0]["alpha"]:
        alpha_label = "%g flight, %g wrapped" % (settings["alpha"], groups["oxygen"][0]["alpha"])
    gamma_label = "%g" % next(iter(gamma_values)) if len(gamma_values) == 1 else "see stage configs"
    pdf.para(lx + 23, 1285,
             "alpha: %s. g/frame = %g. Epsilon: flight 1 to 0.05; asteroids 0.25 to 0.01; oxygen 0.25 to 0.05. Seeds: 11, 23, 37, 51, 79." %
             (alpha_label, settings.get("gamma_per_frame", config.get("gamma_per_frame", .9995))), lw - 46, 22, 29)
    pdf.text(lx + 23, 1400, "OFFICIAL REWARDS / FROZEN EVALUATION", 22, "F2", MUTED)
    pdf.para(lx + 23, 1439,
             "-1 collision; +1 pickup; 0 otherwise. No shaping. 50 validation / 100 test seeds for wrapped tasks; flight uses one deterministic episode.", lw - 46, 22, 29)

    heading(pdf, mx + 22, 542, "02", "Learning, beyond the loss")
    legend_x = mx + 23
    for variant in variants:
        pdf.line(legend_x, 580, legend_x + 28, 580, COLORS[variant], 4)
        pdf.text(legend_x + 38, 587, LABELS[variant], 20)
        legend_x += 228
    chart(pdf, _validation_series(summary, "frames_mean"), mx + 23, 621, mw - 46, 255,
          "Validation: mean survived physics steps", 1200)
    chart(pdf, _validation_series(summary, "oxygen_mean", ("oxygen",)), mx + 23, 948, mw - 46, 233,
          "Validation: oxygen tanks / episode")
    chart(pdf, training_curves(results, groups), mx + 23, 1247, mw - 46, 220,
          "Training: mean squared TD residual")
    pdf.para(mx + 26, 899, "Five-chain means +/- sample SD; phases have different budgets.", mw - 52, 21, 27)
    pdf.para(mx + 26, 1202, "Promotion needs two passes; survival constrains tank ranking.", mw - 52, 21, 27)
    pdf.para(mx + 26, 1493, "Oxygen TD residuals persist; better behavior does not establish loss convergence.", mw - 52, 21, 27)

    heading(pdf, rx + 22, 542, "03", "Test results and cost")
    pdf.text(rx + 23, 592, "SELECTED POLICY: MEAN +/- SAMPLE SD", 21, "F2", MUTED)
    for index, variant in enumerate(variants):
        yy = 639 + index * 106
        entry = variants[variant]["greedy"]
        pdf.text(rx + 23, yy, LABELS[variant], 24, "F2", COLORS[variant])
        pdf.text(rx + 23, yy + 32, "Frames: " + stat_text(entry["frames_mean"], 1), 23)
        pdf.text(rx + 23, yy + 62, "Complete: " + stat_text(entry["survival_rate"], 1, 100) + "%", 23)
    oxy = variants["oxygen"]["greedy"]
    pdf.para(rx + 23, 965,
             "Oxygen: %s tanks per 1,000 survived steps. Complete with >=3 tanks: %s%%." %
             (stat_text(oxy["oxygen_per_1000_frames"], 2), stat_text(oxy["combined_success_rate"], 1, 100)), rw - 46, 22, 29)
    pdf.text(rx + 23, 1072, "TRAINING COST: INCREMENTAL / CUMULATIVE", 21, "F2", MUTED)
    timing_chart(pdf, variants, rx + 23, 1114, rw - 46)
    pdf.text(rx + 23, 1380, "Dark: stage / pale: plus parents; whiskers: sample SD.", 19, color=MUTED)
    random_frames = variants["oxygen"]["random"]["frames_mean"]["mean"]
    random_tanks = variants["oxygen"]["random"]["oxygen_mean"]["mean"]
    pdf.para(rx + 23, 1425,
             "Qualified oxygen chains: %d/%d. Random: %.1f frames, %.2f tanks. Seed 11 is the fixed demo. Transfer and unequal stage budgets prevent a from-zero cost comparison." %
             (variants["oxygen"]["qualified_runs"], variants["oxygen"]["n"], random_frames, random_tanks), rw - 46, 21, 27)


    pdf.rect(65, 1557, W - 130, 91, NAVY)
    pdf.text(83, 1588, "REPRODUCIBLE AND UNMODIFIED", 20, "F2", "8FD9E1")
    sha = str(summary["environment_commit"])[:12]
    pdf.text(83, 1620, "Python %s / %s  |  Official famnit-gym %s  |  Default physics and rewards" % (manifest.get("python", "recorded"), manifest.get("machine", "recorded"), sha), 20, color=WHITE)
    pdf.text(1290, 1588, "COST AND UNCERTAINTY", 20, "F2", "8FD9E1")
    pdf.text(1290, 1620, "Five parallel chains; active wall time excludes validation/export. Cumulative includes parents.", 19, color=WHITE)
    pdf.text(65, 1670, "Sources: Watkins & Dayan, Q-learning (1992); Domen Soberl, official famnit-gym. Exact bins, configs, checkpoint logs, tests and model hashes accompany the source ZIP.", 18, color=MUTED)
    pdf.save(out)
    return Path(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("poster.pdf"))
    args = parser.parse_args()
    print(build(args.results, args.out))


if __name__ == "__main__":
    main()
