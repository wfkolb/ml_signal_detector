from torch import nn
class SignalDetectionNN(nn.Module):
    """Signal Detection neural net
    """
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        """Forward pass through the network.
    
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Network output.
        """
        logits = self.linear_relu_stack(x)
        return logits
