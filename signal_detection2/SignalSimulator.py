

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

from signaldetection_model import signal_detection_nn
from signal_dataset_loader import SignalDataSet


##################################
###UTIL CLASSES####
 #######################
import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
import common_functions as cf
# -----------------------
# Background worker thread
# -----------------------
class DataWorker(QThread):
    new_data = pyqtSignal(object)  # Signal to send new data to the GUI
    signal_found = pyqtSignal(bool)  # Signal to send new data to the GUI
    command = pyqtSignal(str)  # receive commands from GUI
    signalOffset = pyqtSignal(int)  # receive commands from GUI
    signalSnr = pyqtSignal(int)  # receive commands from GUI
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        print(f"Using {self.device} device")
        self.model = signal_detection_nn().to(self.device)
        self.model.load_state_dict(torch.load(cf.get_model_path(), weights_only=True))
        print(self.model)
        self.model.eval() # set to eval mode.

        self.running = True
        self.command.connect(self.handle_command)
        self.signalOffset.connect(self.setOffset)
        self.currentOffsetValue_int = 0

        self.currentSNRValue_db = 0
        self.signalSnr.connect(self.setSnr)


    def run(self):
        while self.running:
            # Simulate data acquisition
            val = self.getGetDataFromParameters(self.currentSNRValue_db,self.currentOffsetValue_int)
            self.new_data.emit(val)  # Send data to GUI
            #Run the ML model to check if the signal is present
            with torch.no_grad():
                val = torch.from_numpy(val.astype(np.float32))
                val = val.to(self.device)
                pred = self.model(val)
                temp = torch.nn.functional.softmax(pred, dim=0)
                isSignalThere = temp[0] < temp[1] and (temp[0] > 0.6 or temp[1] > 0.6)
                print(f'AI REPORT:{temp}, SNR {self.currentSNRValue_db} , Offset {self.currentOffsetValue_int}')
                self.signal_found.emit(isSignalThere)
            time.sleep(0.05)  # 20 Hz

    def stop(self):
        self.running = False
        self.wait()
    

    def getGetDataFromParameters(self,snr_db,sample_offset):
        temp = cf.ideal_signal()
        temp = cf.shift_vector(temp,sample_offset)
        temp = cf.apply_noise_to_vector(temp,snr_db,-15)
        return temp

    @pyqtSlot(int)
    def setSnr(self,val):
        self.currentSNRValue_db = val

    @pyqtSlot(int)
    def setOffset(self,val):
        self.currentOffsetValue_int = val
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
        
        self.plot_widget.setStyleSheet("color: black;")
        self.curve = self.plot_widget.plot(pen='y')  # yellow line
        self.layout.addWidget(self.plot_widget)

        # Buttons
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.layout.addWidget(self.start_btn)
        self.layout.addWidget(self.stop_btn)

        self.slider = QSlider(QtCore.Qt.Orientation.Horizontal)  # Horizontal slider
        self.slider.setMinimum(-100)            # Minimum value
        self.slider.setMaximum(500)          # Maximum value
        self.slider.setValue(0)             # Initial value
        self.slider.setTickPosition(QSlider.TicksBelow)  # Show ticks
        self.slider.setTickInterval(10)                  # Tick interval
        self.layout.addWidget(self.slider)

        self.snr_slider = QSlider(QtCore.Qt.Orientation.Horizontal)  # Horizontal slider
        self.snr_slider.setMinimum(-50)            # Minimum value
        self.snr_slider.setMaximum(40)          # Maximum value
        self.snr_slider.setValue(5)             # Initial value
        self.snr_slider.setTickPosition(QSlider.TicksBelow)  # Show ticks
        self.snr_slider.setTickInterval(10)                  # Tick interval
        self.layout.addWidget(self.snr_slider)

        # Connect buttons
        self.start_btn.clicked.connect(self.start_worker)
        self.stop_btn.clicked.connect(self.stop_worker)
        self.slider.valueChanged.connect(self.slider_val_changed)
        self.snr_slider.valueChanged.connect(self.snr_slider_val_changed)
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
            self.plot_widget.setBackground('k')  # green background
            self.curve.setPen('g')
        else:
             self.plot_widget.setBackground('k')  # green background
             self.curve.setPen('r')

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

    def snr_slider_val_changed(self,value):
        self.worker.signalSnr.emit(value)
        
        



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