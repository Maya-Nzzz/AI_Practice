import torch
import torch.nn as nn

class FullyConnectedModel(nn.Module):
    def __init__(self, version='', num_classes = 10):
        super(FullyConnectedModel, self).__init__()
        if version=='deep':
            layers = [
                nn.Flatten(), # Flatten превратить многомерный тензор в плоский вектор
                nn.LazyLinear(512), nn.ReLU(),
                nn.LazyLinear(256), nn.ReLU(),
                nn.LazyLinear(128), nn.ReLU()
            ]
        else:
            layers = [
                nn.Flatten(),
                nn.LazyLinear(256), nn.ReLU(),
                nn.LazyLinear(128), nn.ReLU()
            ]
        layers.append(nn.LazyLinear(num_classes)) # слои автоматически подхватывают размер входа при первой прогонке
        self.net = nn.Sequential(*layers) # это контейнер, который объединяет несколько слоёв в одну цепочку

    def forward(self, x: torch.Tensor):
        return self.net(x)