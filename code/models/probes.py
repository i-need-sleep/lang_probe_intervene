import torch

class LinearProbe(torch.nn.Module):
    def __init__(self, model_dim=2048):
        super().__init__()
        self.linear = torch.nn.Linear(model_dim, 1)

    def forward(self, x):
        return self.linear(x)
    

class MLPProbe(torch.nn.Module):
    def __init__(self, hidden_size=1024, model_dim=2048):
        super().__init__()
        self.linear1 = torch.nn.Linear(model_dim, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.nn.functional.relu(x)
        x = self.linear2(x)
        return x