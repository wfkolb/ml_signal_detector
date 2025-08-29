import sys
import time
import torch
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QSlider
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
from signaldetection_model import SignalDetectionNN
import common_functions as cf
# -----------------------
# Background worker thread
# -----------------------
class DataWorker(QThread):
    """A background data worker class that generates test data and executes a neural net in a asynchronous thread.

    This class exetends QThread and essentially holds a set of pyqtSignals for communication with the 
    primary GUI.

    Attributes:
        device (str): The device type used to run the neural net
        model (signal_detection_nn): The neural net used to detect the primary signal
        running_nn (bool): If true the background thread is operating and running the neural net
        current_offset_value_int (int): Offset from the ideal signal in samples
        current_snr_value_db (int): Signal to noise ratio of the test signal generated (in dBFS)
        new_data_signal (pyqtSignal(object)): Signal to alert new data was generated
        signal_found_signal (pyqtSignal(bool)): Signal that the signal is found (true=found)
        signal_offset_signal (pyqtSignal(int)): Signal from gui to set signal offset
        signal_snr_signal (pyqtSignal(int)): Signal from gui to pass the current user defined SNR

    Example:
       >>> self.worker = DataWorker()
       >>> self.worker.new_data_signal.connect(self.update_plot)
    """
    new_data_signal = pyqtSignal(object)  # Signal to send new data to the GUI
    signal_found_signal = pyqtSignal(bool)  # Signal to send new data to the GUI
    signal_offset_signal = pyqtSignal(int)  # receive commands from GUI
    signal_snr_signal = pyqtSignal(int)  # receive commands from GUI
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        print(f"Using {self.device} device")
        self.model = SignalDetectionNN().to(self.device)
        self.model.load_state_dict(torch.load(cf.get_model_path(), weights_only=True))
        print(self.model)
        self.model.eval() # set to eval mode.
        self.running_nn = False
        self.signal_offset_signal.connect(self.set_offset)
        self.current_offset_value_int = 0
        self.current_snr_value_db = 0
        self.signal_snr_signal.connect(self.set_snr)

    def run(self):
        """The primary exeuction for the worker thread.
        """
        while self.running_nn:
            # Simulate data acquisition
            val = self.get_new_test_signal(self.current_snr_value_db,self.current_offset_value_int)
            self.new_data_signal.emit(val)  # Send data to GUI
            #Run the ML model to check if the signal is present
            with torch.no_grad():
                val = torch.from_numpy(val.astype(np.float32))
                val = val.to(self.device)
                pred = self.model(val)
                temp = torch.nn.functional.softmax(pred, dim=0)
                is_signal_found = temp[0] < temp[1] and (temp[0] > 0.6 or temp[1] > 0.6)
                print(f'AI REPORT:{temp}, SNR {self.current_snr_value_db} , Offset {self.current_offset_value_int}')
                self.signal_found_signal.emit(is_signal_found)
            time.sleep(0.05)  # Limit to 20 Hz, if it can go faster thats cool but I dont want that.

    def start_nn(self):
        """Starts the neural net processing of the worker thread.
        """
        self.running_nn = True
        self.start()

    def stop_nn(self):
        """Stops the the neural net processing of the worker thread.
        """
        self.running_nn = False
        self.wait()
    

    def get_new_test_signal(self,snr_db,sample_offset):
        """Generates and returns a new test signal for analysis
        Args:
            snr_db (int): The desired Signal to Noise ratio of the test signal in dBFS
            sample_offset (int): The number of samples to shift the test signal

        Returns:
            numpy array: The new test signal as a numpy array
        """
        new_test_signal = cf.generate_ideal_signal()
        new_test_signal = cf.shift_vector(new_test_signal,sample_offset)
        new_test_signal = cf.apply_noise_to_vector(new_test_signal,snr_db,-15)
        return new_test_signal

    @pyqtSlot(int)
    def set_snr(self,snr_int):
        """Sets the SNR used by the worker to generate a test signal
        Args:
            snr_int (_type_): The desired Signal to Noise ratio of the test signal in dBFS
        """
        self.current_snr_value_db = snr_int

    @pyqtSlot(int)
    def set_offset(self,offset_int):
        """Sets the sample offset used by the worker to generate a test signal

        Args:
            offset_int (int): The desired sample offset of the test signal in dBFS
        """
        self.current_offset_value_int = offset_int

class AnalysisPlot(QWidget):
    """Plot for running neural net tests

    Attributes:
        layout (QVBoxLayout): Box Layout for the gui
        plot_widget (pyqtgraph.PlotWidget): Primary Plot area
        curve (self.plot_widget.plot): Actual plot line
        start_btn (QPushButton): Button to start the neural net test injection
        stop_btn (QPushButton): Button to stop the neural net test injection
        sample_offset_slider (QSlider): Slider to control sample offset
        snr_slider (QSlider): Slider to the test signal SNR
        plot_data (numpy.array) : data that is currently being plotted

    Example:
       >>> self.worker = DataWorker()
       >>> self.worker.new_data_signal.connect(self.update_plot)


    """
    def __init__(self):
        super().__init__()
        # Layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Plot
        self.plot_widget = pg.PlotWidget(title="Evaluated Test Signal")
        self.plot_widget.setBackground('k')  # black background
        self.plot_widget.setStyleSheet("color: black;")
        self.curve = self.plot_widget.plot(pen='y')  # yellow line
        self.layout.addWidget(self.plot_widget)

        # Buttons
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.layout.addWidget(self.start_btn)
        self.layout.addWidget(self.stop_btn)

        # Sample Slider
        self.sample_offset_slider = QSlider(QtCore.Qt.Orientation.Horizontal)  # Horizontal slider
        self.sample_offset_slider.setMinimum(-500)            # Minimum value
        self.sample_offset_slider.setMaximum(500)          # Maximum value
        self.sample_offset_slider.setValue(0)             # Initial value
        self.sample_offset_slider.setTickPosition(QSlider.TicksBelow)  # Show ticks
        self.sample_offset_slider.setTickInterval(10)                  # Tick interval
        self.layout.addWidget(self.sample_offset_slider)

        # SNR Slider
        self.snr_slider = QSlider(QtCore.Qt.Orientation.Horizontal)  # Horizontal slider
        self.snr_slider.setMinimum(-50)            # Minimum value
        self.snr_slider.setMaximum(40)          # Maximum value
        self.snr_slider.setValue(5)             # Initial value
        self.snr_slider.setTickPosition(QSlider.TicksBelow)  # Show ticks
        self.snr_slider.setTickInterval(10)                  # Tick interval
        self.layout.addWidget(self.snr_slider)

        # Connect UI Elements
        self.start_btn.clicked.connect(self.start_worker)
        self.stop_btn.clicked.connect(self.stop_worker)
        self.sample_offset_slider.valueChanged.connect(self.offset_slider_val_changed)
        self.snr_slider.valueChanged.connect(self.snr_slider_val_changed)
        
        # Data buffer
        self.plot_data = np.zeros(100)

        # Worker thread
        self.worker = DataWorker()
        self.worker.new_data_signal.connect(self.update_plot)
        self.worker.signal_found_signal.connect(self.set_plot_green)

    # -------------------
    # Slot to update plot
    # -------------------
    def update_plot(self, plot_data_in):
        """updates the plot with new data

        Args:
            plot_data_in (pytorch.tensor): data to be plotted
        """
        self.plot_data = plot_data_in
        self.curve.setData(self.plot_data.astype(float))

    def set_plot_green(self,should_be_green):
        """Sets the plot green or red

        Args:
            should_be_green (bool): if true, will set the plot to be green
        """
        if should_be_green:
            self.plot_widget.setBackground('k')  # green background
            self.curve.setPen('g')
        else:
            self.plot_widget.setBackground('k')  # green background
            self.curve.setPen('r')

    def start_worker(self):
        """Start the neural net worker
        """
        self.worker.start_nn()

    def stop_worker(self):
        """Stop the neural net worker
        """
        self.worker.stop_nn()

    def offset_slider_val_changed(self,value):
        """Emits the new value of the offset slider
        """
        self.worker.signal_offset_signal.emit(value)

    def snr_slider_val_changed(self,value):
        """Emits the new value of the snr slider
        """
        self.worker.signal_snr_signal.emit(value)
# -----------------------
# Run the app
# -----------------------
if __name__ == "__main__":
    pg.setConfigOption('background', 'k')  # black background
    pg.setConfigOption('foreground', 'w')  # white axes/text
    app = QApplication(sys.argv)
    window = AnalysisPlot()
    window.resize(800, 500)
    window.show()
    sys.exit(app.exec_())