
import pickle
import random
import math
import common_functions as cf
import numpy as np
import matplotlib.pyplot as plt

def generate_data_set(ideal_vector,
                      number_training_vectors=50000,
                      max_random_shift=-1,
                      min_random_snr_db=-10,
                      max_random_snr_db=10,
                      system_noise_floor_db=-30):
    """Generates a test data set
       
       The theory here is that you: 
       1.) Take an idealized signal
       2.) Generate a Random snr value and sample shift value
       3.) Shift and add nosie to the idealized signal
       4.) If the snr is above a threshold claim the signal is there.
       5.) Do 1-4 a LOT of times
       6.) Save off all the data.
       
    Args:
        ideal_vector (_type_): _description_
        number_training_vectors (int, optional): # of training vectors to save. Defaults to 50000.
        max_random_shift (int, optional): Max random shift value. Defaults to -1.
        min_random_snr_db (int, optional): Minimum SNR for random pulls. Defaults to -10.
        max_random_snr_db (int, optional): Max SNR for random pulls. Defaults to 10.
        system_noise_floor_db (int, optional): The db value to define nosie power. Defaults to -30.

    Returns:
          (numpy.array, numpy.array): An array of tuples. 
                    First element is the data, second element is two booleans [no_seen_signal, signal_seen]
    """
    data_size = len(ideal_vector)
    output_training_data = []
    if max_random_shift == -1:
        max_random_shift = math.trunc(data_size/2)
    print(f'Generating {number_training_vectors} signals...')
    for x in range(1,number_training_vectors):
        random_shift = random.randint(-1*max_random_shift,max_random_shift)
        shifted_vector = cf.shift_vector(ideal_vector,random_shift)
        random_snr = random.uniform(min_random_snr_db,max_random_snr_db)
        final_vector = cf.apply_noise_to_vector(shifted_vector,random_snr,system_noise_floor_db)
        if(random_snr > -1):
            should_detect_data = True
            label = np.array([0 ,1])
        else:
            should_detect_data = False
            label = np.array([1 ,0])
        assert final_vector.shape[0] ==data_size , "Vector does not match intentded size"
        output_training_data.append((final_vector, label))
        if x % 100 == 0:
            if(should_detect_data):
                plt.plot(final_vector,alpha=0.5,color='green')
            else:
               plt.plot(final_vector,alpha=0.5,color='red')
    plt.plot(ideal_vector,label="DesiredSignal",color='blue')
    print("Close Plot to continue.")
    plt.show(block=True)
    return output_training_data

if __name__ == "__main__":
    training_file = cf.get_training_data_file()
    test_file = cf.get_test_data_file()
    training_data = generate_data_set(cf.generate_ideal_signal(),max_random_shift=128)
    test_data = generate_data_set(cf.generate_ideal_signal(),max_random_shift=128)
    with open(training_file, "wb") as f:
        pickle.dump(training_data, f)
    with open(test_file, "wb") as f:
        pickle.dump(test_data, f)

    print('####################################################')
    print('FINISHED GENERATING DATA')
    print('####################################################')