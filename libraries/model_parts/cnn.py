import torch
import torch.nn as nn

def CNN_28x28(F, FILTER) :
    CNN = nn.Sequential(
            nn.Conv2d(F,   FILTER *1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(FILTER *1),
            nn.ReLU(),
            nn.Conv2d(FILTER *1, FILTER *2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(FILTER *2),
            nn.ReLU(),
            nn.Conv2d(FILTER *2, FILTER *4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(FILTER *4),
            nn.ReLU(),
            nn.Conv2d(FILTER *4, FILTER *8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(FILTER *8),
            nn.ReLU(),
        )
    return CNN

def TCNN_28x28(F, FILTER) :
    CNN_Transpose = nn.Sequential(
            nn.ConvTranspose2d(FILTER *8, FILTER *4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(FILTER *4),
            nn.ReLU(),
            nn.ConvTranspose2d(FILTER *4, FILTER *2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(FILTER *2),
            nn.ReLU(),
            nn.ConvTranspose2d(FILTER *2, FILTER, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(FILTER *1),
            nn.ReLU(),
            nn.Conv2d(FILTER, F, kernel_size=3, stride=1, padding=1),
    )
    return CNN_Transpose

## Models Encoder-Decoder for 28x28 image datas
class Encoder(nn.Module):
    def __init__(self, CNN: nn.Module, D: int, SHAPE: torch.Size):
        super(Encoder, self).__init__()
        self.D = D # latent shape [N x D]

        self.CNN = CNN
        self.Linear = nn.Linear(in_features = SHAPE.numel(), out_features = 2*D)

    def forward(self, X):
        MU_LOGVAR = self.CNN(X)
        
        MU_LOGVAR = nn.Flatten()(MU_LOGVAR)
        MU_LOGVAR = self.Linear(MU_LOGVAR)
        
        MU, LOGVAR = torch.split(MU_LOGVAR, self.D, 1)
        return MU, LOGVAR
    
    
class Decoder(nn.Module):
    def __init__(self, TCNN: nn.Module, D: int, SHAPE: torch.Size, sigmoid: bool = True):
        super(Decoder, self).__init__()
        self.D = D # latent shape [N x D]
        self.SHAPE = SHAPE

        self.Linear = nn.Linear(in_features = D, out_features = SHAPE.numel())
        
        # TCNN (CNN Transpose)
        self.TCNN = TCNN
        self.Sigmoid = nn.Sigmoid()
        self.sigmoid = sigmoid
        
    def forward(self, Z):
        Z = self.Linear(Z)
        Z = Z.view((-1,) + self.SHAPE)
        
        hat_X = self.TCNN(Z)
        hat_X = self.Sigmoid(hat_X) if self.sigmoid else hat_X # sigmoid nécessaire pour BCE_loss
        return hat_X

# Give Encoder and Decoder corresponding to the CNN, TCNN, FILTER and the INPUT_SHAPE
def get_coder(CNN, TCNN, D, F, INPUT_SHAPE) :
    test = torch.randn((1, F) + INPUT_SHAPE)
    test = CNN(test)
    SHAPE =  test.shape[1:]
    Enc = Encoder(CNN, D, SHAPE)
    Dec = Decoder(TCNN, D, SHAPE)
    return Enc, Dec
