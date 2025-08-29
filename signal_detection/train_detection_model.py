import torch
import common_functions as cf
from torch import nn
from torch.utils.data import DataLoader
from signaldetection_model import SignalDetectionNN
from signal_dataset_loader import SignalDataSet

def train(dataloader, model_in, loss_fn_in, optimizer_in,device_in):
    """Trains the input neural net model
    This was essentially ripped from pytorch tutorials...

    Args:
        dataloader (pytorch.DataLoader): The dataloader for the training data
        model_in (pytorch.Module): The pytorch neural net
        loss_fn_in (pytorch._Loss): The pytorch loss function
        optimizer_in (pytorch.Optimizer): The pytorch optimizer
        device_in (str): String for pytorch device
    """
    size = len(dataloader.dataset)
    model_in.train()
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device_in), y.to(device_in)
        # Compute prediction error
        pred = model_in(x)
        loss = loss_fn_in(pred, y)

        # Backpropagation
        loss.backward()
        optimizer_in.step()
        optimizer_in.zero_grad()
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model_in, loss_fn_in,device_in):
    """Tests the input neural net model
    This was essentially ripped from pytorch tutorials but the loss print
    was modified for my needs.
    
    Modified with David Shin.

    Args:
        dataloader (pytorch.DataLoader): The dataloader for the training data
        model_in (pytorch.Module): The pytorch neural net
        loss_fn_in (pytorch._Loss): The pytorch loss function
        device_in (str): String for pytorch device
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader.dataset)
    model_in.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device_in), y.to(device_in)
            pred = model_in(x)
            pred_percentages = torch.nn.functional.softmax(pred, dim=0)
            test_loss += loss_fn_in(pred, y).item()
            close_values = torch.isclose(y,pred_percentages,0.0,0.1)
            correct += close_values.sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == "__main__":
    training_data = SignalDataSet(cf.get_training_data_file())
    test_data = SignalDataSet(cf.get_test_data_file())
    BATCH_SIZE = 64
    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)
    DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {DEVICE} device")
    model = SignalDetectionNN().to(DEVICE)
    print(model)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    EPOCHS = 5
    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer,DEVICE)
        test(test_dataloader, model, loss_fn,DEVICE)
    print("Done!")
    torch.save(model.state_dict(), cf.get_model_path())
    print("Saved PyTorch Model State to " + str(cf.get_model_path()))