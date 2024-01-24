import torch.nn as nn

class EmotionCNN(nn.Module):
    def __init__(self, input_channels=1):  # Parameter added to specify input channels
        super(EmotionCNN, self).__init__()
        print(f"Input channels: {input_channels}")
        self.conv1 = nn.Conv1d(input_channels, 8, 3, stride=1)
        #self.conv1 = nn.Conv1d(8,input_channels,  5000, stride=1)  # Adjusted input channels to 8
        print(f"Conv1: {self.conv1}")
        self.bn1 = nn.BatchNorm1d(8)
        self.lr1 = nn.LeakyReLU(0.3)
        self.mxp = nn.MaxPool1d(4, 1)
        self.conv2 = nn.Conv1d(8, 16, 3, stride=1)  # Adjusted input channels to match the previous layer's output
        self.bn2 = nn.BatchNorm1d(16)
        self.lr2 = nn.LeakyReLU(0.3)
        self.conv3 = nn.Conv1d(16, 32, 3, stride=1)  # Adjusted input channels to match the previous layer's output
        self.bn3 = nn.BatchNorm1d(32)
        self.lr3 = nn.LeakyReLU(0.3)
        self.fn = nn.Flatten()
        self.ms = nn.Mish()
        self.dp = nn.Dropout(0.2)

        self.regressor = nn.Sequential(
            nn.Linear(32000, 1024),  # Adjusted input size to match the previous layer's output
            nn.Mish(),
            nn.Dropout(0.2),
            nn.Linear(1024, 256),
            nn.Mish(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        print(f"Input shape forward: {x.shape}")
        #x = x.type(self.conv1.weight.type())
        #x = x.float()
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.lr1(output)
        output = self.mxp(output)

        output = self.conv2(output)
        output = self.bn2(output)
        output = self.lr2(output)
        output = self.mxp(output)

        output = self.conv3(output)
        output = self.bn3(output)
        output = self.lr3(output)
        output = self.mxp(output)

        output = self.fn(output)
        output = self.regressor(output)

        return output



""" 
class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 4, 3, stride=1)
        self.bn1 = nn.BatchNorm1d(4)
        self.lr1 = nn.LeakyReLU(0.3)
        self.mxp1 = nn.MaxPool1d(4, 1)
        self.conv2 = nn.Conv1d(4, 8, 3, stride=1)
        self.bn2 = nn.BatchNorm1d(8)
        self.lr2 = nn.LeakyReLU(0.3)
        self.mxp2 = nn.MaxPool1d(4, 1)
        self.conv3 = nn.Conv1d(8, 16, 3, stride=1)
        self.bn3 = nn.BatchNorm1d(16)
        self.lr3 = nn.LeakyReLU(0.3)
        self.mxp3 = nn.MaxPool1d(4, 1)
        self.fn = nn.Flatten()
        self.ms = nn.Mish()
        self.dp = nn.Dropout(0.2)
        # Adjusting the regressor layers to fit your output shape
        self.regressor = nn.Sequential(
                            nn.Linear(1856, 512),  # Adjusted input size based on conv output shape
                            nn.ReLU(),
                            nn.Dropout(0.2),
                            nn.Linear(512, 128),
                            nn.ReLU(),
                            nn.Linear(128, 3)  # Assuming 3 output labels for valence, arousal, dominance
                            )

    def forward(self, x):
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.lr1(output)
        output = self.mxp1(output)

        output = self.conv2(output)
        output = self.bn2(output)
        output = self.lr2(output)
        output = self.mxp2(output)

        output = self.conv3(output)
        output = self.bn3(output)
        output = self.lr3(output)
        output = self.mxp3(output)

        output = self.fn(output)
        output = self.regressor(output)

        return output
 """