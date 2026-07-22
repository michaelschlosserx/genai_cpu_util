#!/usr/bin/env python3
"""Plots timeline.png and cores.png from the samples a run wrote."""

import matplotlib
matplotlib.use("Agg")  # headless dev host
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

NCPU = 16
INTERVAL = 0.1  # replaced in render() by the rate actually present in the data
WIN = 10        # samples in a 1 s window

BLUE = "#2a78d6"
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
        "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SURFACE, INK, INK2, MUTED, GRID, AXIS = ("#fcfcfb", "#0b0b0b", "#52514e",
                                         "#898781", "#e1e0d9", "#c3c2b7")
CRIT = "#d03b3b"
BLUES = LinearSegmentedColormap.from_list("blues", RAMP)


def caption():
    return (f"sampled every {INTERVAL:g} s; percentages are % of all {NCPU} cores "
            f"(100% = every core busy), not % of one core")


def load_cpu(d):
    a = np.loadtxt(d / "cpu.csv", delimiter=",")
    total = a[:, 1:9].sum(1)
    busy = total - a[:, 4] - a[:, 5]  # idle and iowait are not busy
    return a[:, 0], busy, total, a[:, 9:41:2], a[:, 10:42:2]


def load_marks(d):
    rows = [l.split(",") for l in (d / "marks.csv").read_text().splitlines()]
    return [(float(r[0]), r[1]) for r in rows]


# the mark is called "inference" everywhere in the data; only the charts rename it
RENAME = {"inference": "concurrent inference"}


def display(tag):
    return RENAME.get(tag.strip(), tag)


def rolling(t, busy, total, w):
    """% over a w-sample window, taken from the cumulative endpoints so it is
    exact, and timestamped at the window midpoint."""
    return (t[w:] + t[:-w]) / 2, 100.0 * (busy[w:] - busy[:-w]) / (total[w:] - total[:-w])


def figure(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, lw=0.6)
    for side, spine in ax.spines.items():
        spine.set_visible(side in ("left", "bottom"))
        spine.set_color(AXIS)
        spine.set_linewidth(0.8)
    ax.tick_params(colors=AXIS, labelcolor=INK2, labelsize=8, length=3, width=0.8)
    return fig, ax


def legend(ax, **kw):
    leg = ax.legend(frameon=False, fontsize=9, **kw)
    for txt in leg.get_texts():
        txt.set_color(INK2)


def finish(fig, path, text=None):
    fig.tight_layout(rect=(0, 0.05, 1, 1))  # reserve a strip for the caption
    fig.text(0.008, 0.012, text or caption(), color=MUTED, fontsize=7.5,
             ha="left", va="bottom")
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def phase_series(runs):
    """Average the traces across runs. Runs drift by a second or two, so a plain
    time-aligned mean would smear the phase boundaries -- which are the sharpest
    and most interesting transitions. Instead each phase is resampled onto a
    common grid, averaged there, and the phases are laid end to end at their
    median durations, so boundaries stay exactly aligned."""
    per_run = []
    for run in runs:
        t, busy, total, _, _ = load_cpu(run)
        marks = load_marks(run)
        x, y = rolling(t, busy, total, 1)
        per_run.append((x, y, [(tag, a, b) for (a, tag), (b, _) in zip(marks, marks[1:])]))

    tags = [tag for tag, _, _ in per_run[0][2]]
    xs, mean, lo, hi, phases, clock = [], [], [], [], [], 0.0
    for i, tag in enumerate(tags):
        span = sorted(p[2][i][2] - p[2][i][1] for p in per_run)[len(per_run) // 2]
        grid = np.linspace(0, 1, max(2, int(span / INTERVAL)))
        stack = []
        for x, y, ph in per_run:
            a, b = ph[i][1], ph[i][2]
            sel = (x >= a) & (x < b)  # half-open, as in the peak marking
            if sel.sum() > 1:
                stack.append(np.interp(grid, (x[sel] - a) / (b - a), y[sel]))
        if not stack:
            continue
        stack = np.array(stack)
        xs.append(clock + grid * span)
        mean.append(stack.mean(0))
        lo.append(stack.min(0))
        hi.append(stack.max(0))
        phases.append((tag, clock, clock + span))
        clock += span
    return (np.concatenate(xs), np.concatenate(mean), np.concatenate(lo),
            np.concatenate(hi), phases)


def timeline(runs, label, path):
    if len(runs) == 1:
        t, busy, total, _, _ = load_cpu(runs[0])
        marks = load_marks(runs[0])
        t0 = marks[0][0]
        xr, yr = rolling(t, busy, total, 1)
        xs, ys = rolling(t, busy, total, WIN)
        xr, xs = xr - t0, xs - t0
        phases = [(tag, a - t0, b - t0) for (a, tag), (b, _) in zip(marks, marks[1:])]
    else:
        xr, mean, lo, hi, phases = phase_series(runs)
        yr = hi  # peaks are the max across runs, never an average

    fig, ax = figure(13, 5)
    for i, (tag, a, b) in enumerate(phases):
        ax.axvspan(a, b, color=GRID, alpha=0.45 if i % 2 else 0.15, lw=0)
        ax.axvline(a, color=AXIS, lw=0.5, alpha=0.7)
    if len(runs) == 1:
        # the rolling mean is the shape to read; the raw samples sit behind it as
        # context, the same weighting the multi-run chart gives its min-max band
        ax.plot(xr, yr, color=BLUE, lw=0.8, alpha=0.3,
                label=f"raw {INTERVAL:g} s samples")
        ax.plot(xs, ys, color=BLUE, lw=2, label="1 s rolling mean")
    else:
        ax.fill_between(xr, lo, hi, color=BLUE, alpha=0.18, lw=0,
                        label=f"min-max across {len(runs)} runs")
        ax.plot(xr, mean, color=BLUE, lw=2, label=f"mean of {len(runs)} runs")

    ax.set_xlim(0, phases[-1][2])
    ax.set_ylim(0, max(yr) * 1.35)
    ax.set_xlabel("elapsed seconds", color=INK2, fontsize=9)
    ax.set_ylabel(f"CPU busy, % of all {NCPU} cores", color=INK2, fontsize=9)
    ax.set_title(f"System CPU over the run  ({label})", color=INK, fontsize=13,
                 loc="left", pad=14)
    # above the axes: the top of the plot is full of phase labels
    legend(ax, loc="lower right", bbox_to_anchor=(1, 1.005), ncol=2)

    # absolute peak of the raw samples, marked per phase; the global one also
    # gets its phase named
    # half-open [a, b): an inclusive upper bound puts a sample landing exactly on
    # a boundary in both phases, which duplicates the spike at a phase edge
    window = lambda a, b: (xr >= a) & (xr < b)
    top = max(range(len(phases)),
              key=lambda i: yr[window(phases[i][1], phases[i][2])].max(initial=0))
    right = phases[-1][2] * 0.88
    prev_x, high = -1e9, False
    for i, (tag, a, b) in enumerate(phases):
        sel = window(a, b)
        if not sel.any():
            continue
        j = int(np.flatnonzero(sel)[yr[sel].argmax()])
        ax.plot(xr[j], yr[j], "o", ms=6 if i == top else 4.5, color=CRIT,
                mec=SURFACE, mew=1.2, zorder=5)
        # lift every second label when two peaks are close, and flip the last
        # ones inward so they are not clipped by the right spine
        high = not high if xr[j] - prev_x < phases[-1][2] * 0.06 else False
        prev_x = xr[j]
        text = f"{yr[j]:.1f}%" + (f"  ({display(tag)})" if i == top else "")
        ax.annotate(text, (xr[j], yr[j]), textcoords="offset points",
                    xytext=(-6 if xr[j] > right else 6, 14 if high else 5),
                    ha="right" if xr[j] > right else "left",
                    color=CRIT, fontsize=8.5 if i == top else 7.5)

    # rotated labels only where the band is wide enough for the glyph height,
    # otherwise short phases stack their names on top of each other
    fig.canvas.draw()
    px = ax.transData.transform
    for tag, a, b in phases:
        if px((b, 0))[0] - px((a, 0))[0] < 14:
            continue
        name = display(tag)
        name = name if len(name) <= 21 else name[:20] + "…"
        ax.text((a + b) / 2, ax.get_ylim()[1] * 0.985, name, rotation=90,
                ha="center", va="top", color=MUTED, fontsize=7.5)
    finish(fig, path)


def cores(run, label, path):
    t, _, _, cbusy, ctotal = load_cpu(run)
    pct = 100.0 * np.diff(cbusy, axis=0) / np.diff(ctotal, axis=0)
    k = max(1, -(-len(pct) // 600))  # average into at most ~600 columns
    n = len(pct) // k * k
    grid = pct[:n].reshape(-1, k, NCPU).mean(1).T
    span = t[n] - t[0]

    fig, ax = figure(13, 4.4)
    ax.grid(False)
    im = ax.imshow(grid, aspect="auto", cmap=BLUES, vmin=0, vmax=100,
                   interpolation="nearest", extent=(0, span, NCPU - 0.5, -0.5))
    marks = load_marks(run)
    phases = [(tag, a - t[0], b - t[0]) for (a, tag), (b, _) in zip(marks, marks[1:])]
    for _, a, _ in phases[1:]:
        ax.axvline(a, color=SURFACE, lw=0.6, alpha=0.55)
    ax.set_yticks(range(NCPU), [f"cpu{i}" for i in range(NCPU)])
    ax.tick_params(labelsize=7.5)
    ax.set_xlabel("elapsed seconds", color=INK2, fontsize=9)
    ax.set_title(f"Per-core utilisation  ({label})", color=INK, fontsize=13,
                 loc="left", pad=26)  # room for the phase names above the image

    # phase names above the heatmap, with a tick bracketing each band; skipped
    # where the band is too narrow for the text, as in the timeline
    fig.canvas.draw()
    px, renderer = ax.transData.transform, fig.canvas.get_renderer()
    last_right = -1e9
    for tag, a, b in phases:
        ax.plot([a, b], [-0.95, -0.95], color=AXIS, lw=0.8, clip_on=False)
        name = display(tag)
        name = name if len(name) <= 20 else name[:19] + "…"
        txt = ax.text((a + b) / 2, -1.15, name, ha="center", va="bottom",
                      color=MUTED, fontsize=7, clip_on=False)
        # measure rather than guess. A label may overhang its band -- the bracket
        # underneath shows the real extent -- so only collisions are fatal, and
        # left-to-right order means the earlier phase wins the space
        box = txt.get_window_extent(renderer=renderer)
        if box.x0 < last_right + 6:
            txt.remove()
        else:
            last_right = box.x1
    cb = fig.colorbar(im, ax=ax, pad=0.012, fraction=0.03)
    cb.set_label("core utilisation %", color=INK2, fontsize=9)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=AXIS, labelcolor=INK2, labelsize=8, length=3, width=0.8)
    # not the shared caption: this is the one chart whose scale is per-core, and
    # inheriting the "% of all 16 cores" wording would contradict the colorbar
    finish(fig, path, f"sampled every {INTERVAL:g} s; each row is one core and the "
                      f"scale is 0-100% of that core alone, so a single dark row "
                      f"means one saturated core while the board is mostly idle; "
                      f"thin white lines are phase boundaries")


def render(out):
    """Write timeline.png and cores.png for a report directory."""
    runs = ([out] if (out / "cpu.csv").is_file()
            else sorted(out.glob("run[0-9]*"), key=lambda p: int(p.name[3:])))
    label = "single run" if runs[0] == out else runs[0].name

    # the sampling rate is a benchmark option, so read it back off the data
    # rather than assuming; the caption and the 1 s window depend on it
    global INTERVAL, WIN
    INTERVAL = round(float(np.median(np.diff(load_cpu(runs[0])[0]))), 4)
    WIN = max(1, int(round(1.0 / INTERVAL)))
    # the timeline can aggregate (phase-normalised); a per-core heatmap cannot be
    # averaged without destroying the which-core identity, so it stays on one run
    agg = f"{len(runs)} runs, phase-normalised" if len(runs) > 1 else label

    written = [out / n for n in ("timeline.png", "cores.png")]
    timeline(runs, agg, written[0])
    cores(runs[0], label, written[1])
    print("written: " + ", ".join(str(p) for p in written))
    return written
