import torch
from torch import nn

class example(nn.Module):
    def __init__(self) :
        super().__init__()

    def forward(self, input):
        output = input+1
        return output
ex = example()
x = torch.tensor(1.0)
output=ex(x)
print(output)