import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

class PlotWithButtons(QWidget):
    def __init__(self):
        super().__init__()
        # Layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # PyQtGraph plot
        self.plot_widget = pg.PlotWidget(title="Plot with Buttons")
        self.layout.addWidget(self.plot_widget)
        self.curve = self.plot_widget.plot(pen='y')

        # Buttons
        self.button1 = QPushButton("Start")
        self.button2 = QPushButton("Stop")
        self.button3 = QPushButton("ToggleTrueSignal")
        self.layout.addWidget(self.button1)
        self.layout.addWidget(self.button2)
        self.layout.addWidget(self.button3)

        # Connect buttons
        self.button1.clicked.connect(self.start_plotting)
        self.button2.clicked.connect(self.stop_plotting)

        # Timer for real-time updates
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.data = np.zeros(100)
        self.ptr = 0

    def start_plotting(self):
        self.timer.start(50)  # 20 Hz

    def stop_plotting(self):
        self.timer.stop()

    def update_plot(self):
        # Simulate new data
        new_val = np.random.normal()
        self.data[:-1] = self.data[1:]
        self.data[-1] = new_val
        self.curve.setData(self.data)

# Run the app
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PlotWithButtons()
    window.resize(800, 500)
    window.show()
    sys.exit(app.exec_())