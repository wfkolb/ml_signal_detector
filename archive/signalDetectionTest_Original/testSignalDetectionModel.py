import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from signaldetection_model import NeuralNetwork
from trainingDataSetLoader import CustomImageDataset
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model_signal.pth", weights_only=True))
test_data = CustomImageDataset('..','TestData')
print(model)


# Download test data from open datasets.
model.eval()
x, y = test_data[200]
for batch_idx,(x,y) in enumerate(test_data):
    with torch.no_grad():
        x = x.to(device)
        isSignalThere = y[0] < y[1]
        pred = model(x)
        doesModelThinkItsThere = pred[0] < pred[1]
        passFail = isSignalThere == doesModelThinkItsThere
        #print(f'Actual : {y.tolist()} - Output : {pred.tolist()} -  pass/fail {passFail}' )
        print(
            f"Actual : {[f'{val:.2f}' for val in y.tolist()]} "
            f"- Output : {[f'{val:.2f}' for val in pred.tolist()]} "
            f"- pass/fail {passFail}"
)





