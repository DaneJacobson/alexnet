import torch
import torch.nn as nn
import torchvision

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.dense = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=(256 * 6 * 6), out_features=4096),
            nn.ReLU(),

            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=(4096), out_features=4096),
            nn.ReLU(),

            nn.Linear(in_features=4096, out_features=num_classes),
        )

        self.init_bias()

    def init_bias(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(tensor=layer.weight, mean=0, std=0.01)
                nn.init.constant_(tensor=layer.bias, val=0)

        nn.init.constant_(self.net[4].bias, 1)
        nn.init.constant_(self.net[10].bias, 1)
        nn.init.constant_(self.net[12].bias, 1)


    def forward(self, x):
        x = self.net(x)
        x = self.dense(x)
        return x
    

if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = AlexNet().to(device)
    print(model)

    imagenet_data = torchvision.datasets.ImageNet("./data")
    data_loader = torch.utils.data.DataLoader(
        imagenet_data,
        batch_size=4,
        shuffle=True,
    )