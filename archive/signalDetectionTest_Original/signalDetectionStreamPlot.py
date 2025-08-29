

from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QSlider
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
import pyqtgraph as pg
import time
import random

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from signaldetection_model import NeuralNetwork
from trainingDataSetLoader import CustomImageDataset



def getReferenceVector(offset):
    ##Create "Perfect" vector
    # Length of vector
    size = 784
    x = np.arange(size)
    
    # Gaussian parameters
    mu1, sigma1 = (100+offset), 10   # center=100, std=10
    mu2, sigma2 = (200+offset), 20   # center=200, std=20
    maxLeftShift = 70
    maxRightShift = size-150

    # Create two Gaussian curves
    gauss1 = np.exp(-0.5 * ((x - mu1) / sigma1) ** 2)
    gauss2 = np.exp(-0.5 * ((x - mu2) / sigma2) ** 2)

    # Combine them into one vector
    vector = gauss1 + gauss2

    return vector

def getSystemNoise():
    size = 784
    x = np.arange(size)
    peakSystemNoise = 0.1
    minSystemNoise = 0.01
    std_dev = random.uniform(minSystemNoise,peakSystemNoise)
    gaussian_noise = np.random.normal(0.0, std_dev, size=x.shape)
    return gaussian_noise
   



##################################
###UTIL CLASSES####
 #######################
import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

# -----------------------
# Background worker thread
# -----------------------
class DataWorker(QThread):
    new_data = pyqtSignal(object)  # Signal to send new data to the GUI
    signal_found = pyqtSignal(bool)  # Signal to send new data to the GUI
    command = pyqtSignal(str)  # receive commands from GUI
    signalOffset = pyqtSignal(int)  # receive commands from GUI
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        print(f"Using {self.device} device")
        self.model = NeuralNetwork().to(self.device)
        self.model.load_state_dict(torch.load("model_signal.pth", weights_only=True))
        test_data = CustomImageDataset('..','TestData')
        print(self.model)
        self.model.eval() # set to eval mode.

        self.running = True
        self.shouldAddSignal = False
        self.command.connect(self.handle_command)
        self.signalOffset.connect(self.setOffset)
        self.currentOffsetValue_int = 0


    def run(self):
        while self.running:
            # Simulate data acquisition
            val = getSystemNoise()
            if self.shouldAddSignal:
                val = val + getReferenceVector(self.currentOffsetValue_int)
            self.new_data.emit(val)  # Send data to GUI
            #Run the ML model to check if the signal is present
            with torch.no_grad():
                val = torch.from_numpy(val.astype(np.float32))
                val = val.to(self.device)
                pred = self.model(val)
                isSignalThere = pred[0] < pred[1]
                print(f'NEW AI REPORT:{ torch.nn.functional.softmax(pred, dim=0)}')
                self.signal_found.emit(isSignalThere)
            time.sleep(0.05)  # 20 Hz

    def stop(self):
        self.running = False
        self.wait()
    
    @pyqtSlot(int)
    def setOffset(self,val):
        self.currentOffsetValue_int = val
        print(f"VALUE = {val}")
    @pyqtSlot(str)
    def handle_command(self, cmd):
        print(cmd)
        if cmd == "START":
            print("PlotStarted")
        if cmd == "STOP":
            print("PlotStopped")
        if cmd == "ADD":
            print("Noise Added.")
            self.shouldAddSignal = not self.shouldAddSignal
            # handle reset logic here

# -----------------------
# Main GUI window
# -----------------------
class PlotWithButtons(QWidget):
    def __init__(self):
        super().__init__()

        # -------------------
        # Layout & widgets
        # -------------------
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Plot
        self.plot_widget = pg.PlotWidget(title="Real-time Plot")
        self.plot_widget.setBackground('k')  # black background
        self.curve = self.plot_widget.plot(pen='y')  # yellow line
        self.layout.addWidget(self.plot_widget)

        # Buttons
        self.start_btn = QPushButton("Start")
        self.toggle_btn = QPushButton("ToggleSignal")
        self.stop_btn = QPushButton("Stop")
        self.layout.addWidget(self.start_btn)
        self.layout.addWidget(self.toggle_btn)
        self.layout.addWidget(self.stop_btn)
        self.slider = QSlider(QtCore.Qt.Orientation.Horizontal)  # Horizontal slider
        self.slider.setMinimum(-100)            # Minimum value
        self.slider.setMaximum(500)          # Maximum value
        self.slider.setValue(0)             # Initial value
        self.slider.setTickPosition(QSlider.TicksBelow)  # Show ticks
        self.slider.setTickInterval(10)                  # Tick interval
        self.layout.addWidget(self.slider)

        # Connect buttons
        self.start_btn.clicked.connect(self.start_worker)
        self.stop_btn.clicked.connect(self.stop_worker)
        self.toggle_btn.clicked.connect(self.addcmd_worker)
        self.slider.valueChanged.connect(self.slider_val_changed)
        # Data buffer
        self.data = np.zeros(100)

        # Worker thread
        self.worker = DataWorker()
        self.worker.new_data.connect(self.update_plot)
        self.worker.signal_found.connect(self.setPlotGreen)


    # -------------------
    # Slot to update plot
    # -------------------
    def update_plot(self, val):
        #self.data[:-1] = self.data[1:]  # shift left
        #self.data[-1] = val             # append new value
        self.data = val
        self.curve.setData(self.data.astype(float))

    def setPlotGreen(self,val):
       # print(f'NEW AI REPORT:{val}')
        if val==True:
            self.plot_widget.setBackground('g')  # green background
            self.curve.setPen('k')
        else:
             self.plot_widget.setBackground('r')  # green background
             self.curve.setPen('k')

    # -------------------
    # Start/stop worker
    # -------------------
    def start_worker(self):
        if not self.worker.isRunning():
            self.worker.running = True
            self.worker.start()
            self.worker.command.emit("START")

    def stop_worker(self):
        self.worker.command.emit("STOP")
        if self.worker.isRunning():
            self.worker.stop()

    def addcmd_worker(self):
        self.worker.command.emit("ADD")

    def slider_val_changed(self,value):
        self.worker.signalOffset.emit(value)
        
        



# -----------------------
# Run the app
# -----------------------
if __name__ == "__main__":
    pg.setConfigOption('background', 'k')  # black background
    pg.setConfigOption('foreground', 'w')  # white axes/text

    app = QApplication(sys.argv)
    window = PlotWithButtons()
    window.resize(800, 500)
    window.show()
    sys.exit(app.exec_())