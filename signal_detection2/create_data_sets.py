import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import random
import common_functions as cf
import math
from pathlib import Path

def generate_data_set(ideal_vector,
                      number_positive_training_vectors=50000,
                      number_negative_training_vectors=10000,
                      max_random_shift=-1,
                      min_random_snr_db=-10,
                      max_random_snr_db=10,
                      system_noise_floor_db=-30):
    data_size = len(ideal_vector)
    outputTrainingData = []
    if max_random_shift == -1:
        max_random_shift = math.trunc(data_size/2)
    print('Generating positive data...')
    for x in range(1,number_positive_training_vectors):
        random_shift = random.randint(-1*max_random_shift,max_random_shift)
        #random_shift = 0
        shifted_vector = cf.shift_vector(ideal_vector,random_shift)
        random_snr = random.uniform(min_random_snr_db,max_random_snr_db)
        final_vector = cf.apply_noise_to_vector(shifted_vector,random_snr,system_noise_floor_db)
        if(random_snr > -1):
            positiveVal = True
            label = np.array([0 ,1])
        else:
            positiveVal = False
            label = np.array([1 ,0])
        assert final_vector.shape[0] ==data_size , "Vector does not match intentded size"
        outputTrainingData.append((final_vector, label))
        if x % 100 == 0:
            if(positiveVal):
                plt.plot(final_vector,alpha=0.5,color='green')
            else:
               plt.plot(final_vector,alpha=0.5,color='red')
        
    # print('Generating negative data...')
    # for x in range(1,number_negative_training_vectors):
    #     label = np.array([1, 0])
    #     final_vector = cf.generate_noise_vector(len(ideal_vector),system_noise_floor_db)
    #     assert final_vector.shape[0] ==data_size , "Vector does not match intentded size"
    #     outputTrainingData.append((final_vector, label))
    #     if x % 100 == 0:
    #         plt.plot(final_vector,alpha=0.5,color='red')
    plt.plot(ideal_vector,label="DesiredSignal",color='blue')
    plt.show(block=True)
    return outputTrainingData


if __name__ == "__main__":
    training_file = cf.get_training_data_file()
    test_file = cf.get_test_data_file()
    training_data = generate_data_set(cf.ideal_signal(),max_random_shift=128)
    test_data = generate_data_set(cf.ideal_signal(),max_random_shift=128)
    with open(training_file, "wb") as f:
        pickle.dump(training_data, f)
    with open(test_file, "wb") as f:
        pickle.dump(test_data, f)

    print('####################################################')
    print('FINISHED GENERATING DATA')
    print('####################################################')