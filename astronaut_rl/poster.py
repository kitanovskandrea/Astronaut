"""Build the A1 landscape conference poster from measured results, using stdlib.

This small vector-PDF writer keeps the submitted program free of extra plotting
or document libraries. It uses the standard PDF base fonts and no raster plots.
"""
import argparse
import json
import struct
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
        self.images = []

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

    def image(self, path, x, y, width):
        """Embed a prepared RGB JPEG using the standard PDF image format."""
        data = Path(path).read_bytes()
        if data[:2] != b"\xff\xd8":
            raise ValueError("The poster logo must be a JPEG")
        position = 2
        while position < len(data):
            marker = data[position + 1]
            length = struct.unpack_from(">H", data, position + 2)[0]
            if marker in (0xC0, 0xC1, 0xC2):
                bits, pixels_h, pixels_w, channels = struct.unpack_from(">BHHB", data, position + 4)
                if bits != 8 or channels != 3:
                    raise ValueError("The poster logo must use 8-bit RGB")
                break
            position += 2 + length
        else:
            raise ValueError("Cannot read JPEG dimensions")
        name = "Im" + str(len(self.images))
        self.images.append((data, pixels_w, pixels_h))
        height = width * pixels_h / pixels_w
        self.commands.append(f"q {width:.2f} 0 0 {height:.2f} {x:.2f} {H-y-height:.2f} cm /{name} Do Q")

    def save(self, path):
        stream = ("\n".join(self.commands) + "\n").encode("latin1")
        image_refs = " ".join(f"/Im{i} {10+i} 0 R" for i in range(len(self.images)))
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {W} {H}] "
             f"/Resources << /Font << /F1 4 0 R /F2 5 0 R /F3 6 0 R /F4 7 0 R >> /XObject << {image_refs} >> >> /Contents 8 0 R >>").encode(),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Oblique /Encoding /WinAnsiEncoding >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier /Encoding /WinAnsiEncoding >>",
            f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"endstream",
            b"<< /Title (Reinforcement learning with the Astronaut) /Author (Andrea Kitanovska; Hristijan Kochovski) /Subject (Intelligent Systems Seminar 1) >>",
        ]
        for content, width, height in self.images:
            header = (f"<< /Type /XObject /Subtype /Image /Width {width} /Height {height} "
                      f"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length {len(content)} >>\nstream\n").encode()
            objects.append(header + content + b"\nendstream")
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


def chart(pdf, series, x, y, width, height, title, ylabel, maximum=None, percent=False):
    max_x = max(point['step'] for points in series.values() for point in points)
    max_y = maximum or max(.00001, max(p['mean'] + p['sd'] for points in series.values() for p in points) * 1.08)
    left, top, pw, ph = x + 74, y + 55, width - 100, height - 110
    xm = lambda value: left + pw * value / max_x
    ym = lambda value: top + ph * (1 - value / max_y)
    pdf.text(x + 12, y + 4, title, 23, 'F2')
    pdf.text(left, y + 31, ylabel, 18, color=MUTED)
    for index in range(5):
        value = max_y * index / 4
        yy = ym(value)
        pdf.line(left, yy, left + pw, yy)
        label = ('%g%%' % value) if percent else ('%.3f' % value if max_y < .1 else number(round(value, 1)))
        pdf.text(left - 65, yy + 6, label, 18, color=MUTED)
    for index in range(6):
        value = max_x * index / 5
        xx = xm(value)
        pdf.line(xx, top + ph, xx, top + ph + 5, MUTED)
        pdf.text(xx - 15, top + ph + 27, number(value), 18, color=MUTED)
    pdf.text(left + pw / 2 - 90, y + height - 2, 'Training decisions', 20, color=MUTED)
    for variant, points in series.items():
        hi = [(xm(p['step']), ym(min(max_y, p['mean'] + p['sd']))) for p in points]
        lo = [(xm(p['step']), ym(max(0, p['mean'] - p['sd']))) for p in reversed(points)]
        pdf.path(hi + lo, pale(COLORS[variant]), fill=True)
        pdf.path([(xm(p['step']), ym(p['mean'])) for p in points], COLORS[variant], 3)


def timing_chart(pdf, variants, x, y, width):
    """The two training-time comparisons, drawn from the same data as training_time.svg."""
    panel_width = (width - 140) / 2
    for panel, (key, title) in enumerate((('stage_cost', 'Time spent learning this stage'),
                                         ('cumulative_cost', 'Total time, including earlier stages'))):
        xx = x + panel * (panel_width + 140)
        left, top, pw, ph = xx + 72, y + 60, panel_width - 100, 120
        maximum = max(v[key]['train_seconds']['mean'] + v[key]['train_seconds']['sd'] for v in variants.values()) * 1.22
        maximum = max(100, ((int(maximum) + 99) // 100) * 100)
        pdf.text(xx + 40, y + 8, title, 24, 'F2')
        pdf.text(xx + 12, y + 37, 'Seconds', 18, color=MUTED)
        ym = lambda value: top + ph * (1 - value / maximum)
        for i in range(4):
            value = maximum * i / 3
            pdf.line(left, ym(value), left + pw, ym(value))
            pdf.text(left - 59, ym(value) + 6, '%.0f' % value, 18, color=MUTED)
        for i, (variant, entry) in enumerate(variants.items()):
            stat = entry[key]['train_seconds']
            center = left + pw * (i + .5) / 3
            low, high = max(0, stat['mean'] - stat['sd']), stat['mean'] + stat['sd']
            pdf.rect(center - 48, ym(stat['mean']), 96, ph * stat['mean'] / maximum, COLORS[variant])
            pdf.line(center, ym(low), center, ym(high), INK, 2)
            for bound in (low, high):
                pdf.line(center - 7, ym(bound), center + 7, ym(bound), INK, 2)
            pdf.text(center - 33, ym(high) - 12, '%.1f s' % stat['mean'], 23, 'F2')
            pdf.text(center - 52, top + ph + 30, LABELS[variant], 21)


def _validation_series(summary, metric, scale=1):
    return {variant: [{'step': p['step'], 'mean': p[metric]['mean'] * scale,
                      'sd': p[metric]['sd'] * scale} for p in points]
            for variant, points in summary['validation_curves'].items()}


def build(results, out):
    results = Path(results)
    manifest = json.loads((results / 'run_manifest.json').read_text(encoding='utf-8'))
    summary = aggregate(results)
    groups = groups_for(manifest['records'])
    variants = summary['variants']
    if set(variants) != {'fly', 'asteroids', 'oxygen'} or any(v['n'] != 5 for v in variants.values()):
        raise ValueError('Poster requires five official-reward runs for each variant')
    if any(r['repeat'] != 6 or not r.get('qualified', False) for r in manifest['records']):
        raise ValueError('Poster requires the qualified six-frame curriculum models')
    oxy = variants['oxygen']['greedy']
    oxygen_games = sum(r['greedy']['episodes'] for r in groups['oxygen'])
    oxygen_total = round(sum(r['greedy']['oxygen_mean'] * r['greedy']['episodes'] for r in groups['oxygen']))
    pdf = PDF()
    pdf.rect(0, 0, W, H, BG)
    pdf.rect(0, 0, W, 250, NAVY)
    pdf.text(65, 52, 'INTELLIGENT SYSTEMS / SEMINAR 1', 23, 'F2', '8FD9E1')
    pdf.text(65, 127, 'Reinforcement learning with the Astronaut', 65, 'F2', WHITE)
    pdf.text(68, 182, 'Training an astronaut to fly, dodge asteroids and collect oxygen using Q-learning', 28, color='CEDDE8')
    pdf.text(68, 225, 'Andrea Kitanovska (89262058)  |  Hristijan Kochovski (89262125)', 27, 'F2', WHITE)
    # Official UP FAMNIT logo: https://www.famnit.upr.si/wp-content/uploads/2025/12/FAMNIT-logo-bel-1.png
    pdf.image(Path(__file__).with_name('assets') / 'famnit-logo.jpg', 1948, 22, 340)

    card_w = (W - 174) / 3
    for i, variant in enumerate(variants):
        xx = 65 + i * (card_w + 22)
        pdf.rect(xx, 275, card_w, 145, WHITE)
        pdf.rect(xx, 275, 7, 145, COLORS[variant])
        pdf.text(xx + 23, 310, LABELS[variant].upper(), 23, 'F2', COLORS[variant])
        completion = variants[variant]['greedy']['survival_rate']['mean'] * 100
        if variant == 'oxygen':
            pdf.text(xx + 23, 361, '%.1f%%  /  %.2f tanks' % (completion, oxy['oxygen_mean']['mean']), 40, 'F2')
            pdf.text(xx + 23, 399, 'Games completed / average tanks per game', 21, color=MUTED)
        else:
            pdf.text(xx + 23, 361, '%.1f%%' % completion, 46, 'F2')
            pdf.text(xx + 23, 399, 'Games completed without a collision', 21, color=MUTED)
    pdf.text(68, 458, 'Five independent training runs; for asteroids and oxygen, each trained agent was tested in 100 unseen games.', 23, color=MUTED)

    lx, mx, rx = 65, 750, 1685
    lw, mw, rw = 650, 900, 634
    for xx, width in ((lx, lw), (mx, mw), (rx, rw)):
        pdf.rect(xx, 482, width, 800, WHITE)
    heading(pdf, lx + 22, 535, '01', 'How the agent learns')
    pdf.para(lx + 23, 582, 'We begin with flight control, then add asteroids and oxygen tanks. Each stage builds on the Q-values learned in the previous one.', lw - 46, 23, 30)
    pdf.text(lx + 23, 720, 'WHAT GOES INTO THE Q-TABLE?', 21, 'F2', MUTED)
    details = {'fly': 'Height and vertical speed', 'asteroids': 'Flight + asteroid arrival time and height',
               'oxygen': 'Avoidance + oxygen position and availability'}
    for i, variant in enumerate(variants):
        yy = 760 + i * 73
        pdf.rect(lx + 23, yy - 25, lw - 46, 64, BG)
        pdf.text(lx + 35, yy, LABELS[variant], 23, 'F2', COLORS[variant])
        pdf.text(lx + 330, yy, '%s states' % format(variants[variant]['state_capacity'], ','), 23, 'F2')
        pdf.text(lx + 35, yy + 25, details[variant], 19, color=MUTED)
    pdf.para(lx + 23, 997, 'For each state, the Q-table stores two values: one for coasting and one for thrusting. The astronaut tries both and updates the values based on what happens next.', lw - 46, 22, 29)
    pdf.text(lx + 23, 1112, 'Q(s,a) <- Q(s,a) + alpha * error', 23, 'F4')
    pdf.para(lx + 23, 1155, 'The update uses the rewards received and the estimated value of the best next action. Each choice lasts up to six physics frames. A tank gives +1, a collision gives -1, and other steps give 0.', lw - 46, 22, 29)

    heading(pdf, mx + 22, 535, '02', 'What changes during training')
    for i, variant in enumerate(variants):
        xx = mx + 23 + i * 243
        pdf.line(xx, 572, xx + 27, 572, COLORS[variant], 4)
        pdf.text(xx + 37, 579, LABELS[variant], 20)
    chart(pdf, _validation_series(summary, 'survival_rate', 100), mx + 23, 614, mw - 46, 245,
          'How survival improves during training', '20-second games finished without collision (%)', 100, True)
    pdf.para(mx + 26, 889, 'Each color shows a game stage. The x-axis counts action choices during training; the y-axis shows the percentage of 20-second games finished without a crash. At checkpoints, the agent plays without random exploration. Flight improves first; later stages build on earlier skills. Shading shows differences between agents.', mw - 52, 21, 27)
    chart(pdf, training_curves(results, groups), mx + 23, 995, mw - 46, 200,
          'How the Q-value prediction error changes', 'Mean squared TD error')
    pdf.para(mx + 26, 1219, 'Flight and asteroid prediction errors fall. Oxygen error remains higher and rises later, even though the agent collects tanks. We cannot claim loss convergence for oxygen.', mw - 52, 21, 27)

    heading(pdf, rx + 22, 535, '03', 'How it plays after training')
    pdf.para(rx + 23, 583, "After training, the agents play using only their learned Q-values. 'Completed' means finishing all 20 seconds without a collision; 'Average game' is the mean survival time. The oxygen stage also reports tanks collected per game. Values are averages across five trained agents (+/- SD).", rw - 46, 21, 27)
    for i, variant in enumerate(variants):
        yy = 720 + i * 116
        entry = variants[variant]['greedy']
        pdf.text(rx + 23, yy, LABELS[variant], 24, 'F2', COLORS[variant])
        pdf.text(rx + 23, yy + 34, 'Completed: ' + stat_text(entry['survival_rate'], 1, 100) + '%', 23)
        if variant == 'oxygen':
            pdf.text(rx + 23, yy + 68, 'Tanks/game: ' + stat_text(entry['oxygen_mean'], 2), 23)
        else:
            pdf.text(rx + 23, yy + 68, 'Average game: %.2f seconds' % (entry['frames_mean']['mean'] / 60), 22)

    heading(pdf, rx + 22, 1080, '04', 'Conclusion')
    pdf.para(rx + 23, 1127, 'We trained the astronaut with tabular Q-learning, first to fly, then to avoid asteroids and collect oxygen. The hardest part was describing moving objects in a small enough state table without losing important detail. We are proud that the final agents finished 96.8% of test games and collected 5.32 tanks per game using the original rewards.', rw - 46, 20.5, 25.5)
    pdf.rect(65, 1307, W - 130, 342, WHITE)
    pdf.text(87, 1346, 'TRAINING TIME: HOW LONG DID LEARNING TAKE?', 25, 'F2')
    timing_chart(pdf, variants, 90, 1382, W - 180)
    pdf.save(out)
    return Path(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results'))
    parser.add_argument('--out', type=Path, default=Path('poster.pdf'))
    args = parser.parse_args()
    print(build(args.results, args.out))


if __name__ == '__main__':
    main()
