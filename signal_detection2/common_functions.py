#Numpy definitley has a function that does this I'm just too impatient to find it.
import colorednoise as cn
import numpy as np
import math
import os
from pathlib import Path

def shift_vector(vec, shift):
    result = np.zeros_like(vec)
    if shift > 0:  # shift right
        result[shift:] = vec[:-shift]
    elif shift < 0:  # shift left
        result[:shift] = vec[-shift:]
    else:  # no shift
        result[:] = vec
    return result

def ideal_signal():
    # Length of vector
    return_size = 784
    x = np.arange(return_size)

    # Gaussian parameters
    mu1, sigma1 = 196, 10   # center=100, std=10
    mu2, sigma2 = 588, 20   # center=200, std=20

    # Create two Gaussian curves
    gauss1 = np.exp(-0.5 * ((x - mu1) / sigma1) ** 2)
    gauss2 = np.exp(-0.5 * ((x - mu2) / sigma2) ** 2)

    # Combine them into one vector
    vector = gauss1 + gauss2
    return vector/vector.max()

def rms_vector(vector_in):
    #Why the fuck do I have to write this. Numpy is garbage
    squared_elements = vector_in**2
    mean_of_squares = np.mean(squared_elements)
    rms_value = np.sqrt(mean_of_squares)
    return rms_value

def apply_noise_to_vector(vector_in,snr_db,system_noise_floor):
    vector_rms_amp_lin = rms_vector(vector_in)
    noise_vector = generate_noise_vector(len(vector_in),system_noise_floor) ## Technically we could use the 0db but I figured we can just rederive it incase I'm wrong
    noise_vector_rms_amp_lin = rms_vector(noise_vector)
    delta_snr = snr_db - 20*math.log10(vector_rms_amp_lin/noise_vector_rms_amp_lin) ##Assuming input is voltage
    delta_snr_lin = 10**(delta_snr/20)
    vector_out_lin = vector_in * delta_snr_lin
    vector_out_lin = vector_out_lin + noise_vector
    return vector_out_lin

def generate_noise_vector(vector_size,rms_db):
    noise_vector_lin = cn.powerlaw_psd_gaussian(0, vector_size)
    noise_vector_out_lin = noise_vector_lin/noise_vector_lin.max()
    noise_vector_rms_amp_lin = rms_vector(noise_vector_lin)
    delta_snr = rms_db - 20*math.log10(noise_vector_rms_amp_lin)
    delta_snr_lin = 10**(delta_snr/20)
    noise_vector_out_lin = noise_vector_lin * delta_snr_lin
    return noise_vector_out_lin

def get_training_data_file():
    script_dir = Path(__file__).parent.absolute()
    training_dir = os.path.join(script_dir,'training_data')
    training_file = os.path.join(training_dir,'training_data.pkl')
    return training_file

def get_test_data_file():
    script_dir = Path(__file__).parent.absolute()
    test_dir = os.path.join(script_dir,'test_data')
    test_file = os.path.join(test_dir,'test_data.pkl')
    return test_file

def get_model_path():
    script_dir = Path(__file__).parent.absolute()
    model_file = os.path.join(script_dir,'model/model.pth')
    return model_file
