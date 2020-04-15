import torch
import torch.nn as nn

class FakeModel(nn.Module):
    """Fake model to test GCP scripts."""
    def __init__(self):
        super(FakeModel, self).__init__()

    def forward(self, x):
        return x

def train(loader, epochs):
    """Training"""
    print("Training")
    model = FakeModel()
    model.train()
    for i in range(epochs):
        for item in loader:
            print(model.forward(item))
    torch.save(model.state_dict(), 'output/model')

def test(loader):
    """Testing"""
    print("Testing")
    model = FakeModel()
    model.load_state_dict(torch.load('output/model'))
    model.eval()
    with torch.no_grad():
        for item in loader:
            print(model.forward(item))

