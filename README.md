### ML Signal Detector
Written by Will Kolb.
Train a neural net to detect a specific signal.
see The blog post talking about this at [my website](https://willkolb.com/?p=1066)

## Basic setup

# Dependancies
- pytorch
- numpy
- pyqtgraph
- pyqt5
- pickle
- matplotlib
- colorednoise

# procedure to run
1. run "create_data_sets.py"
   This will create training and test data (in ./test_data and ./training_data folders as pkl files)
3. run "train_detection_model.py"
   This will create a model .pth file inside of the "model" folder.
5. run "signal_simulator.py"

# Extending/modifying
- The neuralnet is setup in "signaldetection_model.py".
- The signal for detection is defined in "common_functions.py" inside the generate_ideal_signal() function.

  I dont think there's too much to change outside of that function and the neural net. The big thing is that the #
  of input nodes matches the length of the reutrn from the generate_ideal_signal() function. The definition for
  output is defined in the "create_data_sets.py" function.

## Credits/Contact
Developed by Will Kolb (wfkolb@gmail.com), [willkolb.com](willkolb.com)
