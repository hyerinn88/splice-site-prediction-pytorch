import torch
import torch.nn as nn


class TemplateNetwork(nn.Module):
    def __init__(self):
        super(TemplateNetwork, self).__init__()
        self.feature1 = \
            nn.Sequential(\
                # 1
                nn.Conv2d(in_channels=1, out_channels=12, kernel_size=(5, 3)),
                nn.ReLU(),
                nn.BatchNorm2d(12),
                nn.MaxPool2d(kernel_size=(3, 1)),
                nn.Dropout2d(p=0.1),

                # 2
                nn.Conv2d(in_channels=12, out_channels=24, kernel_size=(3, 1)),
                nn.ReLU(),
                nn.BatchNorm2d(24),
                nn.MaxPool2d(kernel_size=(3, 1)),
                nn.Dropout2d(p=0.15),

                # 3
                nn.Conv2d(in_channels=24, out_channels=48, kernel_size=(3, 1)),
                nn.ReLU(),
                nn.BatchNorm2d(48),
                nn.MaxPool2d(kernel_size=(3, 1)),
                nn.Dropout2d(p=0.2),

                # 4
                nn.Conv2d(in_channels=48, out_channels=64, kernel_size=(3, 1)),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(kernel_size=(2, 1)),
                nn.Dropout2d(p=0.25)
            )

        self.classifier = \
            nn.Sequential(
                nn.Linear(in_features=256, out_features=128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(p=0.3),

                nn.Linear(in_features=128, out_features=64),
                nn.ReLU(),
                nn.Dropout(p=0.35),

                nn.Linear(in_features=64, out_features=2))

    def forward(self, x):
        x = self.feature1(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    x = torch.randn((1, 1, 196, 4))
    model = TemplateNetwork()
    out = model(x)
    print(out.shape)
