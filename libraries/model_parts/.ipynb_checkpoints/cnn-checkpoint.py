import torch
import torch.nn as nn
from collections import OrderedDict

# PAROI_VESICALE
def CNN_PAROI_VESICALE(F: int, FILTER: int, SPACING: tuple = (1, 1, 3)) -> nn.Module :
    deepness = None
    if SPACING == (1.5, )*3 :
        deepness = 5
    if SPACING == (3,)*3 :
        deepness = 4

    sequential_list = OrderedDict()
    sequential_list["Conv3d0"] = nn.Conv3d(F,   FILTER *1, kernel_size=3, stride=1, padding=1)
    for i in range(deepness) : 
        # sequential_list[f"BatchNorm3d{i+1}"] = nn.BatchNorm3d(FILTER *2**i)
        sequential_list[f"ReLU{i+1}"] = nn.ReLU()
        sequential_list[f"Conv3d{i+1}"] = nn.Conv3d(FILTER *2**i,   FILTER *2**(i +1), kernel_size=3, stride=2, padding=1)
        
    # sequential_list[f"BatchNorm3d{deepness +1}"] = nn.BatchNorm3d(FILTER *2**deepness)
    sequential_list[f"ReLU{deepness +1}"] = nn.ReLU()
    sequential_list[f"Conv3d{deepness +1}"] = nn.Conv3d(FILTER *2**deepness,   FILTER *2**deepness, kernel_size=3, stride=1, padding=1)
    return nn.Sequential(sequential_list)

def TCNN_PAROI_VESICALE(F: int, FILTER: int, SPACING: tuple = (1, 1, 3)) -> nn.Module :
    deepness = None
    padding_cond = None
    paddings = []
    sequential_list = OrderedDict()
    if SPACING == (1, )*3 :
        deepness = 5
        padding_cond = (2, 1, 2)
     
    elif SPACING == (1.5, )*3 :
        deepness = 5
        for i in range(deepness) :
            if (deepness-i) == 4 :
                paddings.append(2)
            elif (deepness-i) == 3 :
                paddings.append((1, 2, 1))
            else : 
                paddings.append(1)
    elif SPACING == (3, )*3 :
        deepness = 4
        padding_cond = 2
        for i in range(deepness) : 
            if (deepness-i) != 3 : 
                paddings.append(1)
            else :
                paddings.append(padding_cond)

    
    for i in range(deepness) :
        sequential_list[f"Conv3d_{i+1}"] = nn.ConvTranspose3d(FILTER *2**(deepness -i),   FILTER *2**(deepness -1 -i), kernel_size=4, stride=2, padding=paddings[i])
        sequential_list[f"BatchNorm3d_{i+1}"] = nn.BatchNorm3d(FILTER *2**(deepness -1 -i))
        sequential_list[f"ReLU_{i+1}"] = nn.ReLU()
    sequential_list[f"Conv3d_{deepness +1}"] = nn.Conv3d(FILTER,   F, kernel_size=3, stride=1, padding=1)
    
    return nn.Sequential(sequential_list)


# PAROI_VESICALE
def CNN_PAROI_RECTALE(F: int, FILTER: int, SPACING: tuple = (1, 1, 1)) -> nn.Module :
    deepness = None
    if SPACING == (1.5,)*3 :
        deepness = 5
    if SPACING == (3, 3, 3) :
        deepness = 4

    sequential_list = OrderedDict()
    sequential_list["Conv3d0"] = nn.Conv3d(F,   FILTER *1, kernel_size=3, stride=1, padding=1)
    for i in range(deepness) : 
        # sequential_list[f"BatchNorm3d{i+1}"] = nn.BatchNorm3d(FILTER *2**i)
        sequential_list[f"ReLU{i+1}"] = nn.ReLU()
        sequential_list[f"Conv3d{i+1}"] = nn.Conv3d(FILTER *2**i,   FILTER *2**(i +1), kernel_size=3, stride=2, padding=1)
        
    # sequential_list[f"BatchNorm3d{deepness +1}"] = nn.BatchNorm3d(FILTER *2**deepness)
    sequential_list[f"ReLU{deepness +1}"] = nn.ReLU()
    sequential_list[f"Conv3d{deepness +1}"] = nn.Conv3d(FILTER *2**deepness,   FILTER *2**deepness, kernel_size=3, stride=1, padding=1)
    return nn.Sequential(sequential_list)

def TCNN_PAROI_RECTALE(F: int, FILTER: int, SPACING: tuple = (3,)*3) -> nn.Module :
    deepness = None
    padding_cond = None
    if SPACING == (1.5,)*3 :
        deepness = 5
    elif SPACING == (3,)*3 :
        deepness = 4

    sequential_list = OrderedDict()
    for i in range(deepness) : 
        if deepness == 4 :
            padding = 2 if (deepness-i) in [0] else 1
        if deepness == 5 :
            padding = (2, 1, 2) if (deepness-i) in [0] else 1
            padding = (2, 1, 1) if (deepness-i) in [3] else 1
        
        sequential_list[f"Conv3d_{i+1}"] = nn.ConvTranspose3d(FILTER *2**(deepness -i),   FILTER *2**(deepness -1 -i), kernel_size=4, stride=2, padding=padding)
        sequential_list[f"BatchNorm3d_{i+1}"] = nn.BatchNorm3d(FILTER *2**(deepness -1 -i))
        sequential_list[f"ReLU_{i+1}"] = nn.ReLU()
    sequential_list[f"Conv3d_{deepness +1}"] = nn.Conv3d(FILTER,   F, kernel_size=3, stride=1, padding=1)
    
    return nn.Sequential(sequential_list)

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
