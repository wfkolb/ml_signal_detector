import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import random

#Numpy definitley has a function that does this I'm just too impatient to find it.
def shift_vector(vec, shift):
    result = np.zeros_like(vec)
    if shift > 0:  # shift right
        result[shift:] = vec[:-shift]
    elif shift < 0:  # shift left
        result[:shift] = vec[-shift:]
    else:  # no shift
        result[:] = vec
    return result


# Length of vector
size = 784
x = np.arange(size)

# Gaussian parameters
mu1, sigma1 = 100, 10   # center=100, std=10
mu2, sigma2 = 200, 20   # center=200, std=20

maxLeftShift = 70
maxRightShift = size-150


# Create two Gaussian curves
gauss1 = np.exp(-0.5 * ((x - mu1) / sigma1) ** 2)
gauss2 = np.exp(-0.5 * ((x - mu2) / sigma2) ** 2)


# Combine them into one vector
vector = gauss1 + gauss2

trueVector = vector
plt.figure(figsize=(10, 6))


numPositiveTrainingData = 200
numNegativeTrainingData = 100

## Base Noise generation values.
mean = 0
std_dev = 0.2  # Adjust this value to control noise intensity

outputTrainingData = []
#outputTrainingData.append((trueVector,"Yes"))
peakSystemNoise = 0.1
minSystemNoise = 0.01

print('Generating positive data...')
for x in range(1,numPositiveTrainingData):
    std_dev = random.uniform(minSystemNoise,peakSystemNoise)
    random_shift = random.randint(-1*maxLeftShift,maxRightShift)
    shifted_vector = shift_vector(trueVector,random_shift)
    gaussian_noise = np.random.normal(mean, std_dev, size=trueVector.shape)
    final_vector = gaussian_noise + trueVector

    label = np.array([0 ,1])
    assert final_vector.shape[0] ==size , "Vector does not match intentded size"
    outputTrainingData.append((final_vector, label))
    if x % 100 == 0:
        plt.plot(final_vector,alpha=0.5,color='green')

print('Generating negative data...')
for x in range(1,numNegativeTrainingData):
    std_dev = random.uniform(minSystemNoise,peakSystemNoise)
    gaussian_noise = np.random.normal(mean, std_dev, size=trueVector.shape)
    label = np.array([1, 0])
    final_vector = (gaussian_noise)
    assert final_vector.shape[0] ==size , "Vector does not match intentded size"
    outputTrainingData.append((final_vector, label))
    if x % 100 == 0:
        plt.plot(final_vector,alpha=0.5,color='red')



plt.plot(trueVector,label="DesiredSignal",color='blue')
plt.show()
isTrain = 1

if(isTrain):
    #os.mkdir('./TrainingData')
    with open("./TrainingData/training_data.pkl", "wb") as f:
        pickle.dump(outputTrainingData, f)
else:
    with open("./TestData/Test_data.pkl", "wb") as f:
        pickle.dump(outputTrainingData, f)

print("FINISHED")