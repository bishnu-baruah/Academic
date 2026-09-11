import sys
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QComboBox, QGroupBox, QFrame
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPalette, QColor


# ─────────────────────────────  Physics  ──────────────────────────────────────
import core_phy_2
from core_phy_2 import (
    M_QWP_theta, M_HWP_theta, jones_in, apply_plate,
    jones_to_stokes, stokes_to_sphere, polarization_label, compute_coordinate,
    fast_axis_coordinate
)


# ─────────────────────────────  Matplotlib canvas  ────────────────────────────

class SphereCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = plt.figure(figsize=(6, 5), facecolor="#0b1623")
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_facecolor("#0b1623")
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self._n_static = 0
        self._build_static()

    def _build_static(self):
        ax = self.ax

        # Great circles
        t = np.linspace(0, 2*np.pi, 200)
        kw = dict(color="#2a5a8c", alpha=0.4, linewidth=0.8)
        ax.plot(np.cos(t), np.sin(t), np.zeros_like(t), **kw)
        ax.plot(np.cos(t), np.zeros_like(t), np.sin(t), **kw)
        ax.plot(np.zeros_like(t), np.cos(t), np.sin(t), **kw)

        # Sphere as latitude + longitude lines
        sphere_kw = dict(color="#00cfff", alpha=0.18, linewidth=0.6)
        for lat in np.linspace(-np.pi/2, np.pi/2, 10):       # latitudes
            r = np.cos(lat)
            ax.plot(r*np.cos(t), r*np.sin(t),
                    np.full_like(t, np.sin(lat)), **sphere_kw)
        for lon in np.linspace(0, np.pi, 9):                  # longitudes
            ax.plot(np.sin(t)*np.cos(lon), np.sin(t)*np.sin(lon),
                    np.cos(t), **sphere_kw)

        # Cartesian axes + labels
        for pts, col, lbl, pos in [
            ([[-1.4,1.4],[0,0],[0,0]],  "#4a9ff5", "$S_1$ (H)",    (1.55, 0,    0   )),
            ([[0,0],[-1.4,1.4],[0,0]],  "#3ad5a0", "$S_2$ (+45°)", (0,    1.55, 0   )),
            ([[0,0],[0,0],[-1.4,1.4]],  "#d53a7f", "$S_3$ (RCP)",  (0,    0,    1.6 )),
        ]:
            ax.plot(*pts, color=col, alpha=0.7, linewidth=1.2)
            ax.text(*pos, lbl, color=col, fontsize=8, ha="center")

        ax.set_xlim([-1.3, 1.3])
        ax.set_ylim([-1.3, 1.3])
        ax.set_zlim([-1.3, 1.3])
        ax.set_box_aspect([1, 1, 1])
        ax.axis("off")

        self._n_static = len(ax.lines)

    def update_points(self, sph_in, sph_out, fa):
        ax = self.ax

        # Remove dynamic lines and scatter collections
        while len(ax.lines) > self._n_static:
            ax.lines[-1].remove()
        while ax.collections:
            ax.collections[-1].remove()

        xi, yi, zi = sph_in[0],  sph_in[1],  sph_in[2]
        xo, yo, zo = sph_out[0], sph_out[1], sph_out[2]
        fx, fy, fz = fa

        # Geodesic arc (input → output)
        steps = 60
        arc = np.array([(1-t)*np.array([xi,yi,zi]) + t*np.array([xo,yo,zo])
                        for t in np.linspace(0, 1, steps)])
        norms = np.linalg.norm(arc, axis=1, keepdims=True)
        arc = arc / np.where(norms == 0, 1, norms)
        
        # Fast axis diameter (dashed pink line through origin)
        ax.plot([-fx, fx], [-fy, fy], [0, 0],
                color="#dd1e8e", linewidth=1.4, alpha=0.9, linestyle="--")

        # Scatter: input (red), output (cyan), fast axis (pink diamond)
        ax.scatter([xi], [yi], [zi],
                   color="#ff4466", s=80, zorder=5, depthshade=False)
        ax.scatter([xo], [yo], [zo],
                   color="#00e5ff", s=80, zorder=5, depthshade=False)
        ax.scatter([fx], [fy], [fz],
                   color="#dd1e8e", s=60, zorder=5, depthshade=False, marker="D")

        self.draw()


# ─────────────────────────────  Styling helpers  ──────────────────────────────

DARK   = "#0b1623"
PANEL  = "#0f1e30"
ACCENT = "#3a7fd5"
TEXT   = "#c8dff8"
DIM    = "#4a6a8c"

def styled_label(text, size=10, color=TEXT, bold=False):
    lbl = QLabel(text)
    f = QFont("Courier New", size)
    f.setBold(bold)
    lbl.setFont(f)
    lbl.setStyleSheet(f"color: {color}; background: transparent;")
    return lbl


class SliderRow(QWidget):
    def __init__(self, label, min_val, max_val, init_val,
                 unit="°", color=ACCENT, callback=None):
        super().__init__()
        self.unit     = unit
        self.callback = callback
        self.setStyleSheet("background: transparent;")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 4)
        lay.setSpacing(3)

        top = QHBoxLayout()
        self.name_lbl = styled_label(label, size=9, color=DIM)
        self.val_lbl  = styled_label(f"{init_val}{unit}", size=10,
                                     color=color, bold=True)
        top.addWidget(self.name_lbl)
        top.addStretch()
        top.addWidget(self.val_lbl)
        lay.addLayout(top)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(min_val)
        self.slider.setMaximum(max_val)
        self.slider.setValue(init_val)
        self.slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                height: 3px; background: #1a3a5c; border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                width: 13px; height: 13px; margin: -5px 0;
                background: {color}; border-radius: 7px;
            }}
            QSlider::sub-page:horizontal {{
                background: {color}; border-radius: 2px;
            }}
        """)
        self.slider.valueChanged.connect(self._on_change)
        lay.addWidget(self.slider)

    def _on_change(self, v):
        self.val_lbl.setText(f"{v}{self.unit}")
        if self.callback:
            self.callback(v)

    def value(self):
        return self.slider.value()


# ─────────────────────────────  Main window  ──────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Poincaré Sphere — Polarization Visualizer")
        self.setMinimumSize(920, 640)

        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(DARK))
        self.setPalette(palette)
        self.setStyleSheet(f"background-color: {DARK};")

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Left panel ──────────────────────────────────────────────────────
        panel = QWidget()
        panel.setFixedWidth(295)
        panel.setStyleSheet(
            f"background-color: {PANEL}; border-right: 1px solid #1a3a5c;")
        pv = QVBoxLayout(panel)
        pv.setContentsMargins(20, 24, 20, 20)
        pv.setSpacing(18)

        pv.addWidget(styled_label("POINCARÉ SPHERE", size=14,
                                  color="#e0f0ff", bold=True))
        pv.addWidget(styled_label("polarization visualizer", size=8, color=DIM))

        # ── Input state ──────────────────────────────────────────────────
        grp_in = self._group("INPUT JONES VECTOR")
        gv = QVBoxLayout(grp_in)
        gv.setSpacing(8)
        self.sl_angle = SliderRow("Polarization angle  θ", 0, 180, 0,
                                  color="#ff4466", callback=self._update)
        self.sl_phase = SliderRow("Phase difference  δ",   0, 360, 0,
                                  color="#ff7a94", callback=self._update)
        gv.addWidget(self.sl_angle)
        gv.addWidget(self.sl_phase)
        gv.addWidget(self._sep())
        gv.addWidget(styled_label("Stokes (in):", size=8, color=DIM))
        self.stokes_in_lbl = styled_label("", size=9, color="#ff6680")
        self.pol_in_lbl    = styled_label("", size=9, color="#ff8899")
        gv.addWidget(self.stokes_in_lbl)
        gv.addWidget(self.pol_in_lbl)
        pv.addWidget(grp_in)

        # ── Wave plate ───────────────────────────────────────────────────
        grp_wp = self._group("WAVE PLATE")
        wv = QVBoxLayout(grp_wp)
        wv.setSpacing(8)

        plate_row = QHBoxLayout()
        plate_row.addWidget(styled_label("Plate type:", size=9, color=DIM))
        self.combo = QComboBox()
        self.combo.addItems(["QWP", "HWP"])
        self.combo.setStyleSheet(f"""
            QComboBox {{
                background: #0b1623; color: {TEXT}; border: 1px solid #1a3a5c;
                border-radius: 4px; padding: 4px 10px;
                font-family: 'Courier New'; font-size: 10pt;
            }}
            QComboBox::drop-down {{ border: none; }}
            QComboBox QAbstractItemView {{
                background: #0b1623; color: {TEXT};
                selection-background-color: {ACCENT};
            }}
        """)
        self.combo.currentTextChanged.connect(self._update)
        plate_row.addWidget(self.combo)
        wv.addLayout(plate_row)

        self.sl_plate = SliderRow("Fast axis angle  α", 0, 180, 10,
                                  color="#00e5ff", callback=self._update)
        wv.addWidget(self.sl_plate)
        wv.addWidget(self._sep())
        wv.addWidget(styled_label("Stokes (out):", size=8, color=DIM))
        self.stokes_out_lbl = styled_label("", size=9, color="#00ccdd")
        self.pol_out_lbl    = styled_label("", size=9, color="#00e5ff")
        wv.addWidget(self.stokes_out_lbl)
        wv.addWidget(self.pol_out_lbl)
        pv.addWidget(grp_wp)

        # ── Legend ───────────────────────────────────────────────────────
        legend = styled_label(
            "●  Red    = input state\n"
            "●  Cyan   = output state\n"
            "◆  Pink   = fast axis point\n"
            "╌  Pink   = fast axis diameter\n"
            "─  Yellow = geodesic arc",
            size=8, color=DIM)
        legend.setAlignment(Qt.AlignLeft)
        pv.addStretch()
        pv.addWidget(legend)

        root.addWidget(panel)

        # ── Right: 3D canvas ─────────────────────────────────────────────
        self.canvas = SphereCanvas()
        root.addWidget(self.canvas, stretch=1)

        self._update()

    def _group(self, title):
        box = QGroupBox(title)
        box.setStyleSheet(f"""
            QGroupBox {{
                color: {ACCENT}; font-family: 'Courier New'; font-size: 8pt;
                letter-spacing: 2px; border: 1px solid #1a3a5c;
                border-radius: 6px; margin-top: 10px; padding-top: 8px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin; left: 10px; padding: 0 4px;
            }}
        """)
        return box

    def _sep(self):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #1a3a5c;")
        return line

    def _update(self, _=None):
        theta = self.sl_angle.value()
        delta = self.sl_phase.value()
        plate = self.combo.currentText()
        alpha = self.sl_plate.value()

        j_in    = jones_in(theta, delta)
        s_in    = jones_to_stokes(j_in)
        sph_in  = stokes_to_sphere(*s_in)

        sph_out = compute_coordinate(j_in, plate, alpha)
        s_out   = jones_to_stokes(apply_plate(j_in, plate, alpha))

        fa = fast_axis_coordinate(alpha)

        self.stokes_in_lbl.setText(
            f"S = ({s_in[0]:+.3f}, {s_in[1]:+.3f}, {s_in[2]:+.3f})")
        self.pol_in_lbl.setText(polarization_label(*s_in))

        self.stokes_out_lbl.setText(
            f"S = ({s_out[0]:+.3f}, {s_out[1]:+.3f}, {s_out[2]:+.3f})")
        self.pol_out_lbl.setText(polarization_label(*s_out))

        self.canvas.update_points(sph_in, sph_out, fa)


# ─────────────────────────────  Entry point  ──────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
