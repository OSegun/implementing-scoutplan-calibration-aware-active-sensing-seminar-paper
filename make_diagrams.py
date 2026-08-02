"""
Generate the four structural diagrams as PNG (for embedding in the .docx)
and as matching editable .drawio XML (for diagrams.net / Visio import).

Figure 1  System architecture
Figure 2  Data flow diagram
Figure 3  Algorithm flowchart (one scouting episode)
Figure 4  Sequence diagram (one decision step)
"""
from __future__ import annotations
import os
import textwrap
import xml.sax.saxutils as sx

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
DRW = os.path.join(HERE, "drawio")
os.makedirs(FIG, exist_ok=True)
os.makedirs(DRW, exist_ok=True)

GREEN = "#2C5F2D"
LEAF = "#5A8F4A"
LIGHT = "#EAF1E6"
TINT2 = "#DCEBD3"
WARM = "#F2EEE6"
INK = "#1F2A1F"
GREY = "#6B7A6B"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})


# --------------------------------------------------------------- helpers

def box(ax, x, y, w, h, label, fc=LIGHT, ec=GREEN, fs=9, bold=True, tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=1.3, edgecolor=ec, facecolor=fc))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal", color=tc,
            linespacing=1.35)


def arrow(ax, p, q, style="-|>", color=LEAF, lw=1.5, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=13,
                                 linewidth=lw, color=color, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=2, shrinkB=2))


def label(ax, x, y, t, fs=7.5, color=GREY, style="italic", ha="center"):
    ax.text(x, y, t, ha=ha, va="center", fontsize=fs, color=color, style=style)


def canvas(w=11, h=6.2):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    return fig, ax


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", p)


# ------------------------------------------------------- drawio emitter

def drawio(filename, nodes, edges, page=("1100", "850")):
    """
    nodes: list of dicts {id,label,x,y,w,h,fill,stroke,shape}
    edges: list of dicts {src,dst,label,dashed}
    """
    cells = []
    for n in nodes:
        shape = n.get("shape", "rounded=1")
        style = (f"{shape};whiteSpace=wrap;html=1;"
                 f"fillColor={n.get('fill', LIGHT)};"
                 f"strokeColor={n.get('stroke', GREEN)};"
                 f"fontColor={n.get('font', INK)};fontSize=12;"
                 f"align=center;verticalAlign=middle;arcSize=8;")
        cells.append(
            f'        <mxCell id="{n["id"]}" value="{sx.escape(n["label"])}" '
            f'style="{style}" vertex="1" parent="1">\n'
            f'          <mxGeometry x="{n["x"]}" y="{n["y"]}" '
            f'width="{n["w"]}" height="{n["h"]}" as="geometry"/>\n'
            f'        </mxCell>')
    for i, e in enumerate(edges):
        style = ("edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;"
                 f"strokeColor={e.get('stroke', LEAF)};strokeWidth=2;"
                 "endArrow=blockThin;endFill=1;fontSize=11;"
                 + ("dashed=1;" if e.get("dashed") else ""))
        cells.append(
            f'        <mxCell id="e{i}" value="{sx.escape(e.get("label",""))}" '
            f'style="{style}" edge="1" parent="1" '
            f'source="{e["src"]}" target="{e["dst"]}">\n'
            f'          <mxGeometry relative="1" as="geometry"/>\n'
            f'        </mxCell>')

    xml = (
        '<mxfile host="app.diagrams.net" type="device">\n'
        f'  <diagram name="{os.path.splitext(filename)[0]}">\n'
        f'    <mxGraphModel dx="1100" dy="800" grid="1" gridSize="10" '
        f'guides="1" tooltips="1" connect="1" arrows="1" fold="1" '
        f'page="1" pageScale="1" pageWidth="{page[0]}" pageHeight="{page[1]}" '
        f'math="0" shadow="0">\n'
        '      <root>\n'
        '        <mxCell id="0"/>\n'
        '        <mxCell id="1" parent="0"/>\n'
        + "\n".join(cells) + "\n"
        '      </root>\n'
        '    </mxGraphModel>\n'
        '  </diagram>\n'
        '</mxfile>\n')
    p = os.path.join(DRW, filename)
    with open(p, "w", encoding="utf-8") as f:
        f.write(xml)
    print("wrote", p)


# =========================================================== Figure 1

def figure1():
    fig, ax = canvas(11, 6.0)
    ax.text(0.5, 0.965, "ScoutPlan system architecture",
            ha="center", fontsize=12.5, fontweight="bold", color=GREEN)

    box(ax, 0.03, 0.60, 0.20, 0.24,
        "Field Simulator\n(Neyman–Scott\ncluster process)", fc=LIGHT)
    box(ax, 0.03, 0.24, 0.20, 0.24,
        "Image Store\n(PlantVillage /\nPlantDoc pools)", fc=LIGHT)

    box(ax, 0.29, 0.42, 0.20, 0.26,
        "Perception Module\nEfficientNet-B0\n+ temperature layer T", fc=TINT2)

    box(ax, 0.545, 0.42, 0.17, 0.26,
        "Bayesian\nBelief Map\nb(r,c), H(b)", fc=LIGHT)

    box(ax, 0.775, 0.60, 0.20, 0.24,
        "RL Planner\nPPO / D3QN\n(REINFORCE pilot)", fc=TINT2)
    box(ax, 0.775, 0.24, 0.20, 0.24,
        "Baseline Planners\nLawnmower, Greedy,\nRandom, Oracle", fc=WARM)

    box(ax, 0.29, 0.05, 0.42, 0.15,
        "Energy & Kinematics Model\n(hover + translation + altitude)", fc=WARM, fs=9)

    box(ax, 0.29, 0.78, 0.42, 0.15,
        "Evaluation Harness\n(recall@B, detections/joule, IQM, bootstrap CI)",
        fc=LIGHT, fs=9)

    arrow(ax, (0.23, 0.72), (0.29, 0.62)); label(ax, 0.247, 0.695, "labels", ha="left")
    arrow(ax, (0.23, 0.36), (0.29, 0.48)); label(ax, 0.247, 0.395, "images", ha="left")
    arrow(ax, (0.49, 0.55), (0.545, 0.55)); label(ax, 0.517, 0.585, "p(y|x)")
    arrow(ax, (0.715, 0.58), (0.775, 0.68)); label(ax, 0.735, 0.655, "obs", ha="left")
    arrow(ax, (0.715, 0.50), (0.775, 0.40)); label(ax, 0.735, 0.425, "obs", ha="left")
    arrow(ax, (0.875, 0.60), (0.875, 0.48))
    label(ax, 0.895, 0.545, "shared\ninterface", fs=7, ha="left")
    arrow(ax, (0.775, 0.30), (0.71, 0.16))
    label(ax, 0.735, 0.205, "action a", ha="left")
    arrow(ax, (0.36, 0.20), (0.36, 0.42))
    label(ax, 0.378, 0.31, "energy cost", ha="left")
    arrow(ax, (0.63, 0.68), (0.58, 0.78)); label(ax, 0.567, 0.735, "metrics", ha="right")

    save(fig, "Figure1_System_Architecture.png")

    nodes = [
        dict(id="f", label="Field Simulator\n(Neyman-Scott cluster process)", x=40, y=90, w=200, h=90),
        dict(id="i", label="Image Store\n(PlantVillage / PlantDoc pools)", x=40, y=330, w=200, h=90),
        dict(id="p", label="Perception Module\nEfficientNet-B0 + temperature layer T", x=320, y=205, w=220, h=100, fill=TINT2),
        dict(id="b", label="Bayesian Belief Map\nb(r,c), H(b)", x=610, y=210, w=180, h=90),
        dict(id="r", label="RL Planner\nPPO / D3QN (REINFORCE pilot)", x=860, y=90, w=200, h=90, fill=TINT2),
        dict(id="s", label="Baseline Planners\nLawnmower, Greedy, Random, Oracle", x=860, y=330, w=200, h=90, fill=WARM),
        dict(id="e", label="Energy & Kinematics Model\n(hover + translation + altitude)", x=320, y=490, w=470, h=70, fill=WARM),
        dict(id="v", label="Evaluation Harness\n(recall@B, detections/joule, IQM, bootstrap CI)", x=320, y=20, w=470, h=60),
    ]
    edges = [
        dict(src="f", dst="p", label="labels"),
        dict(src="i", dst="p", label="images"),
        dict(src="p", dst="b", label="p(y|x)"),
        dict(src="b", dst="r", label="observation"),
        dict(src="b", dst="s", label="observation"),
        dict(src="r", dst="e", label="action a"),
        dict(src="s", dst="e", label="action a"),
        dict(src="e", dst="p", label="energy cost", dashed=True),
        dict(src="b", dst="v", label="metrics", dashed=True),
    ]
    drawio("Figure1_System_Architecture.drawio", nodes, edges)


# =========================================================== Figure 2

def figure2():
    fig, ax = canvas(11, 5.2)
    ax.text(0.5, 0.95, "Data flow: the perception–planning control loop",
            ha="center", fontsize=12.5, fontweight="bold", color=GREEN)

    ys = 0.46
    box(ax, 0.02, ys, 0.155, 0.24, "True label\ny(r,c)", fc=WARM)
    box(ax, 0.215, ys, 0.165, 0.24, "Classifier\nlogit z", fc=LIGHT)
    box(ax, 0.42, ys, 0.175, 0.24, "Temperature\nscaling\np = σ(2μz/T)", fc=TINT2)
    box(ax, 0.635, ys, 0.165, 0.24, "Bayes update\nb ← b·LR", fc=LIGHT)
    box(ax, 0.84, ys, 0.14, 0.24, "Policy\nπ(a|o)", fc=TINT2)

    for a, b in [(0.175, 0.215), (0.38, 0.42), (0.595, 0.635), (0.80, 0.84)]:
        arrow(ax, (a, ys + 0.12), (b, ys + 0.12))

    arrow(ax, (0.91, ys), (0.10, ys - 0.20), rad=0.14, ls="--", color=GREY)
    label(ax, 0.50, 0.20, "action moves the agent → a new cell is observed → the loop closes",
          fs=8.5, color=GREY)

    label(ax, 0.505, 0.40, "T is the experimental instrument:\n"
                           "accuracy invariant, calibration varied", fs=8, color=LEAF)
    ax.annotate("", xy=(0.505, 0.44), xytext=(0.505, 0.36),
                arrowprops=dict(arrowstyle="-", color=LEAF, lw=1))

    save(fig, "Figure2_Data_Flow.png")

    nodes = [
        dict(id="y", label="True label y(r,c)", x=40, y=200, w=170, h=80, fill=WARM),
        dict(id="z", label="Classifier logit z", x=260, y=200, w=170, h=80),
        dict(id="t", label="Temperature scaling\np = sigma(2*mu*z / T)", x=480, y=200, w=200, h=80, fill=TINT2),
        dict(id="b", label="Bayes update\nb <- b * LR", x=730, y=200, w=170, h=80),
        dict(id="pi", label="Policy pi(a|o)", x=950, y=200, w=150, h=80, fill=TINT2),
        dict(id="n", label="Agent moves; next cell observed", x=480, y=380, w=300, h=60, fill=WARM),
    ]
    edges = [
        dict(src="y", dst="z"), dict(src="z", dst="t"),
        dict(src="t", dst="b", label="p(y|x)"), dict(src="b", dst="pi", label="o"),
        dict(src="pi", dst="n", label="a"), dict(src="n", dst="y", dashed=True, label="new cell"),
    ]
    drawio("Figure2_Data_Flow.drawio", nodes, edges)


# =========================================================== Figure 3

def figure3():
    fig, ax = canvas(7.6, 10.6)
    ax.text(0.5, 0.985, "Algorithm flowchart: one scouting episode",
            ha="center", fontsize=12.5, fontweight="bold", color=GREEN)

    def term(x, y, w, h, t):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.004,rounding_size=0.022",
                                    linewidth=1.3, edgecolor=GREEN, facecolor=WARM,
                                    mutation_aspect=0.35))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center",
                fontsize=9, fontweight="bold", color=INK)

    def dec(x, y, w, h, t):
        ax.add_patch(Polygon([[x + w / 2, y + h], [x + w, y + h / 2],
                              [x + w / 2, y], [x, y + h / 2]],
                             closed=True, linewidth=1.3,
                             edgecolor=GREEN, facecolor=TINT2))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center",
                fontsize=8.5, fontweight="bold", color=INK)

    cx, w = 0.28, 0.44
    steps = [
        ("term", 0.905, 0.055, "Start episode"),
        ("proc", 0.805, 0.070, "Initialise belief b ← prior π₀,\nenergy E ← B, position ← centre"),
        ("proc", 0.700, 0.060, "Observe current cell:\np ← σ(2μz / T)"),
        ("proc", 0.600, 0.060, "Bayes update b(r,c);\nrecord (confidence, correct)"),
        ("dec", 0.480, 0.085, "b(r,c) ≥ τ ?"),
        ("proc", 0.370, 0.060, "Record detection\nor false alarm"),
        ("proc", 0.270, 0.060, "Policy selects action a ~ π(·|o)"),
        ("proc", 0.170, 0.070, "Move; E ← E − (e_hover + e_trans·d);\naccumulate reward r"),
        ("dec", 0.048, 0.085, "E ≤ E_min ?"),
    ]
    ys = {}
    for i, (kind, y, h, t) in enumerate(steps):
        if kind == "term":
            term(cx, y, w, h, t)
        elif kind == "dec":
            dec(cx + 0.06, y, w - 0.12, h, t)
        else:
            box(ax, cx, y, w, h, t, fc=LIGHT, fs=8.5)
        ys[i] = (y, h)

    for i in range(len(steps) - 1):
        y0 = ys[i][0]
        y1 = ys[i + 1][0] + ys[i + 1][1]
        arrow(ax, (cx + w / 2, y0), (cx + w / 2, y1))

    # "no" branch around the detection test
    ax.annotate("", xy=(cx + w / 2, 0.330), xytext=(0.85, 0.330),
                arrowprops=dict(arrowstyle="-|>", color=LEAF, lw=1.5))
    ax.annotate("", xy=(0.85, 0.330), xytext=(0.85, 0.5225),
                arrowprops=dict(arrowstyle="-", color=LEAF, lw=1.5))
    ax.annotate("", xy=(0.85, 0.5225), xytext=(cx + w - 0.06, 0.5225),
                arrowprops=dict(arrowstyle="-", color=LEAF, lw=1.5))
    label(ax, 0.885, 0.44, "no", fs=8, color=LEAF, style="normal")
    label(ax, 0.395, 0.455, "yes", fs=8, color=LEAF, style="normal", ha="left")

    # loop back
    ax.annotate("", xy=(0.09, 0.0905), xytext=(cx + 0.06, 0.0905),
                arrowprops=dict(arrowstyle="-", color=LEAF, lw=1.5))
    ax.annotate("", xy=(0.09, 0.0905), xytext=(0.09, 0.730),
                arrowprops=dict(arrowstyle="-", color=LEAF, lw=1.5))
    ax.annotate("", xy=(cx, 0.730), xytext=(0.09, 0.730),
                arrowprops=dict(arrowstyle="-|>", color=LEAF, lw=1.5))
    label(ax, 0.055, 0.42, "no", fs=8, color=LEAF, style="normal")

    term(cx, -0.045, w, 0.048, "End episode → report metrics")
    arrow(ax, (cx + w / 2, 0.048), (cx + w / 2, 0.005))
    label(ax, cx + w / 2 + 0.02, 0.028, "yes", fs=8, color=LEAF,
          style="normal", ha="left")

    save(fig, "Figure3_Algorithm_Flowchart.png")

    nodes = [
        dict(id="s", label="Start episode", x=380, y=20, w=240, h=50, fill=WARM, shape="rounded=1;arcSize=40"),
        dict(id="a", label="Initialise belief b <- prior, energy E <- B, position <- centre", x=360, y=100, w=280, h=70),
        dict(id="b", label="Observe current cell: p = sigma(2*mu*z / T)", x=360, y=200, w=280, h=60),
        dict(id="c", label="Bayes update b(r,c); record (confidence, correct)", x=360, y=290, w=280, h=60),
        dict(id="d", label="b(r,c) >= tau ?", x=420, y=380, w=160, h=80, fill=TINT2, shape="rhombus"),
        dict(id="e", label="Record detection or false alarm", x=360, y=490, w=280, h=60),
        dict(id="f", label="Policy selects action a ~ pi(.|o)", x=360, y=580, w=280, h=60),
        dict(id="g", label="Move; E <- E - (e_hover + e_trans*d); accumulate reward r", x=360, y=670, w=280, h=70),
        dict(id="h", label="E <= E_min ?", x=420, y=770, w=160, h=80, fill=TINT2, shape="rhombus"),
        dict(id="i", label="End episode -> report metrics", x=380, y=880, w=240, h=50, fill=WARM, shape="rounded=1;arcSize=40"),
    ]
    edges = [
        dict(src="s", dst="a"), dict(src="a", dst="b"), dict(src="b", dst="c"),
        dict(src="c", dst="d"), dict(src="d", dst="e", label="yes"),
        dict(src="d", dst="f", label="no"), dict(src="e", dst="f"),
        dict(src="f", dst="g"), dict(src="g", dst="h"),
        dict(src="h", dst="i", label="yes"), dict(src="h", dst="b", label="no", dashed=True),
    ]
    drawio("Figure3_Algorithm_Flowchart.drawio", nodes, edges, page=("1100", "980"))


# =========================================================== Figure 4

def figure4():
    fig, ax = canvas(11, 5.6)
    ax.text(0.5, 0.96, "Sequence diagram: one decision step",
            ha="center", fontsize=12.5, fontweight="bold", color=GREEN)

    actors = ["Planner", "Environment", "Field", "Classifier", "Belief Map"]
    xs = [0.08, 0.29, 0.50, 0.69, 0.90]
    for x, a in zip(xs, actors):
        box(ax, x - 0.075, 0.80, 0.15, 0.10, a, fc=TINT2, fs=9)
        ax.plot([x, x], [0.10, 0.80], color=GREY, lw=0.9, ls=(0, (4, 4)))

    msgs = [
        (0, 1, 0.72, "step(a)"),
        (1, 2, 0.635, "label(r,c)"),
        (2, 1, 0.555, "y", True),
        (1, 3, 0.475, "observe(y)"),
        (3, 1, 0.395, "p = σ(2μz/T)", True),
        (1, 4, 0.315, "bayes_update(r,c,p)"),
        (4, 1, 0.235, "b, H(b)", True),
        (1, 0, 0.155, "obs, reward, done", True),
    ]
    for i, j, y, t, *dash in msgs:
        d = bool(dash and dash[0])
        arrow(ax, (xs[i], y), (xs[j], y), color=GREY if d else LEAF,
              ls="--" if d else "-", lw=1.2)
        label(ax, (xs[i] + xs[j]) / 2, y + 0.032, t, fs=8,
              color=GREY if d else INK, style="normal")

    label(ax, 0.5, 0.055,
          "Energy is decremented inside Environment.step() before the observation is taken,\n"
          "so an action that cannot be afforded terminates the episode without yielding information.",
          fs=8, color=GREY)

    save(fig, "Figure4_Sequence_Diagram.png")

    nodes = [
        dict(id="pl", label="Planner", x=40, y=20, w=140, h=50, fill=TINT2),
        dict(id="en", label="Environment", x=280, y=20, w=140, h=50, fill=TINT2),
        dict(id="fi", label="Field", x=520, y=20, w=140, h=50, fill=TINT2),
        dict(id="cl", label="Classifier", x=760, y=20, w=140, h=50, fill=TINT2),
        dict(id="bm", label="Belief Map", x=1000, y=20, w=140, h=50, fill=TINT2),
        dict(id="m1", label="1. step(a)", x=40, y=130, w=380, h=40, fill=LIGHT),
        dict(id="m2", label="2. label(r,c) -> y", x=280, y=200, w=380, h=40, fill=LIGHT),
        dict(id="m3", label="3. observe(y) -> p = sigma(2*mu*z / T)", x=280, y=270, w=620, h=40, fill=TINT2),
        dict(id="m4", label="4. bayes_update(r,c,p) -> b, H(b)", x=280, y=340, w=860, h=40, fill=LIGHT),
        dict(id="m5", label="5. return obs, reward, done", x=40, y=410, w=380, h=40, fill=WARM),
    ]
    edges = [dict(src="m1", dst="m2"), dict(src="m2", dst="m3"),
             dict(src="m3", dst="m4"), dict(src="m4", dst="m5")]
    drawio("Figure4_Sequence_Diagram.drawio", nodes, edges)


if __name__ == "__main__":
    figure1(); figure2(); figure3(); figure4()
