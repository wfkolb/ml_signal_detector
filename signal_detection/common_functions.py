import math
import os
from pathlib import Path
import colorednoise as cn
import numpy as np

def shift_vector(vector_in, shift):
    """Shifts the input numpy vector by the set amoutn.

    Args:
        vector_in (numpy.array): Vector to be shifted
        shift (int): number of samples to shift

    Returns:
        numpy.array: The shifted vector
    """
    result = np.zeros_like(vector_in)
    if shift > 0:  # shift right
        result[shift:] = vector_in[:-shift]
    elif shift < 0:  # shift left
        result[:shift] = vector_in[-shift:]
    else:  # no shift
        result[:] = vector_in
    return result

def generate_ideal_signal():
    """Generates the ideal signal used for training and test
    Ideally this would be an input file but for convience
    I made this function to save myself another output file.
    -Will K.
    
    Returns:
        numpy.array: The generated ideal signal.
    """
    # Length of vector
    return_size = 784
    x = np.arange(return_size)

    # Gaussian parameters
    mu1, sigma1 = 196, 10 
    mu2, sigma2 = 588, 20 

    # Create two Gaussian curves
    gauss1 = np.exp(-0.5 * ((x - mu1) / sigma1) ** 2)
    gauss2 = np.exp(-0.5 * ((x - mu2) / sigma2) ** 2)

    # Combine them into one vector
    vector = gauss1 + gauss2
    return vector/vector.max()

def rms_vector(vector_in):
    """Calculate the root mean square of a vector

    Args:
        vector_in (numpy.array): vector to determine rms of

    Returns:
        float: The square root of the means squared
    """
    squared_elements = vector_in**2
    mean_of_squares = np.mean(squared_elements)
    rms_value = np.sqrt(mean_of_squares)
    return rms_value

def apply_noise_to_vector(vector_in,snr_db,system_noise_floor):
    """Applies guassian pink noise to a vector

    Args:
        vector_in (_type_): _description_
        snr_db (_type_): _description_
        system_noise_floor (_type_): _description_

    Returns:
        numpy.array: The input array with noise applied
    """
    vector_rms_amp_lin = rms_vector(vector_in)
    noise_vector = generate_noise_vector(len(vector_in),system_noise_floor) ## Technically we could use the 0db but I figured we can just rederive it incase I'm wrong
    noise_vector_rms_amp_lin = rms_vector(noise_vector)
    delta_snr = snr_db - 20*math.log10(vector_rms_amp_lin/noise_vector_rms_amp_lin) ##Assuming input is voltage
    delta_snr_lin = 10**(delta_snr/20)
    vector_out_lin = vector_in * delta_snr_lin
    vector_out_lin = vector_out_lin + noise_vector
    return vector_out_lin

def generate_noise_vector(vector_size,rms_db):
    """Generates a pink noise vector

    Args:
        vector_size (int): size of the output vector
        rms_db (int): Amplitude in db of the noise

    Returns:
        numpy.array: A vector containing pink noise.
    """
    noise_vector_lin = cn.powerlaw_psd_gaussian(0, vector_size)
    noise_vector_out_lin = noise_vector_lin/noise_vector_lin.max()
    noise_vector_rms_amp_lin = rms_vector(noise_vector_lin)
    delta_snr = rms_db - 20*math.log10(noise_vector_rms_amp_lin)
    delta_snr_lin = 10**(delta_snr/20)
    noise_vector_out_lin = noise_vector_lin * delta_snr_lin
    return noise_vector_out_lin

def get_training_data_file():
    """Get the default path to the training data file

    Returns:
        str: the path to the training data file.
    """
    script_dir = Path(__file__).parent.absolute()
    training_dir = os.path.join(script_dir,'training_data')
    training_file = os.path.join(training_dir,'training_data.pkl')
    return training_file

def get_test_data_file():
    """Get the default path to the test data file

    Returns:
        str: the path to the test data file.
    """
    script_dir = Path(__file__).parent.absolute()
    test_dir = os.path.join(script_dir,'test_data')
    test_file = os.path.join(test_dir,'test_data.pkl')
    return test_file

def get_model_path():
    """Get the default path to model file.

    Returns:
        str: the path to the model file.
    """
    script_dir = Path(__file__).parent.absolute()
    model_file = os.path.join(script_dir,'model/model.pth')
    return model_file
