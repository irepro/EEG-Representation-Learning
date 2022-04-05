import torch
import torch.nn as nn

def padding1D(x, kernel_size, dilation=1):
    pad = ((kernel_size-1)*(2**(dilation+1)),0)
    out = torch.nn.functional.pad(x,pad)
    return out

class maxpool_Full_elec(nn.Module):
    def __init__(self):
        super(maxpool_Full_elec, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x): 
        x_sh = x.shape
        batch_size,in_channels, _ = x_sh
        
        #x = x.reshape([batch_size, in_channels, -1])
        x = self.maxpool(x)
        #x = x.reshape([batch_size, in_channels, -1])
        x = x.squeeze(dim=2)

        return x

class CausalBlock_MV(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, skip=False, kernel_size=3, dilation=1):
        super(CausalBlock_MV, self).__init__()
        self.skip = skip
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        if self.skip:
            self.skip_layer = nn.Conv1d(in_channels, out_channels, 1)
            self.layer = nn.Sequential(
                 #filter = K, kernel_size=3
                    torch.nn.utils.weight_norm(nn.Conv1d(in_channels, mid_channels, kernel_size, dilation=2**dilation)),
                    nn.LeakyReLU(),
                    torch.nn.utils.weight_norm(nn.Conv1d(mid_channels, out_channels, kernel_size, dilation=2**dilation)),
                    nn.LeakyReLU(),
                
            )

        else:
            self.layer = nn.Sequential(
                #filter = K, kernel_size=3
                    torch.nn.utils.weight_norm(nn.Conv1d(in_channels, mid_channels, kernel_size, dilation=2**dilation)),
                    nn.LeakyReLU(),
                    torch.nn.utils.weight_norm(nn.Conv1d(mid_channels, out_channels, kernel_size, dilation=2**dilation)),
                    nn.LeakyReLU(),
                
            )
        
    def forward(self, x):
        x_sh = x.shape
        batch_size = x_sh[0]
        x = x.reshape([batch_size, self.in_channels, -1])

        pad_x = padding1D(x, kernel_size=self.kernel_size, dilation=self.dilation)
        if self.skip:
            out = self.layer(pad_x)
            out += self.skip_layer(x)
            out = out.reshape([batch_size, self.out_channels,-1])
            return out
        else:
            out = self.layer(pad_x)
            out = out.reshape([batch_size, self.out_channels,-1])
            return out

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.causalblock1 = CausalBlock_MV(in_channels, 64, 64, skip=True, kernel_size=5, dilation=0)
        self.causalblock2 = CausalBlock_MV(64, 128, 128, skip=True, kernel_size=5, dilation=1)
        self.causalblock3 = CausalBlock_MV(128, 256, 256, skip=True, kernel_size=5, dilation=2)
        self.causalblock4 = CausalBlock_MV(256, out_channels, out_channels, skip=True, kernel_size=5, dilation=3)

        self.maxpool = maxpool_Full_elec()
        self.out_channels = out_channels
        self.maxpool1d = nn.MaxPool1d(2)
        self.in_channels = in_channels

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = self.causalblock1(x)
        x = self.maxpool1d(x)
        x = self.causalblock2(x)
        x = self.maxpool1d(x)
        x = self.causalblock3(x)
        x = self.maxpool1d(x)
        x = self.causalblock4(x)

        out = self.maxpool(x)

        return out

class USRL(nn.Module):
    ####################################################
    # electorde = the number of electrode
    # in_channels, out_channels = the number of channel
    # Full_elec = True or False about whether you use all electrode data in encoding representation
    # During Unsupervised learning, parameter 'Unsupervise' is True. Then, change parameter to False when you start to supervised learning 
    # ################################################### 
    def __init__(self, electrode, out_channels, Unsupervise = True):
        super(USRL, self).__init__()
        self.encoder = Encoder(electrode, out_channels)
        self.softmax = torch.nn.Softmax(dim=1)
        self.activation = torch.nn.LeakyReLU()
        self.Unsupervise = Unsupervise
        self.out_channels = out_channels
        self.electrode = electrode

    def forward(self, x):
        x = self.encoder(x)
        if not self.Unsupervise:
            x = x.unsqueeze(dim=1)
            x = self.classification(x)
            x = self.softmax(x)
        return x

    def set_Unsupervised(self, device, Unsupervise):
        self.Unsupervise = Unsupervise
        self.classification = nn.Sequential(
            torch.nn.utils.weight_norm(nn.Conv1d(1, 4, 3, 1)),
            nn.ReLU(),
            nn.MaxPool1d(2),
            torch.nn.utils.weight_norm(nn.Conv1d(4, 16, 3, 1)),
            nn.ReLU(),
            nn.MaxPool1d(2),
            torch.nn.utils.weight_norm(nn.Conv1d(16, 64, 3, 1)),
            nn.ReLU(),
            nn.MaxPool1d(2),
            torch.nn.Flatten(),
            nn.Dropout(0.5),
            torch.nn.utils.weight_norm(nn.Linear(3968, 5)),
        ).to(device)

