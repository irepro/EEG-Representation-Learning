
import os
from datetime import datetime
import sys
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from model import ERL_DCCN
from util import utilLoader, utils
from loss import TripletSigmoidLoss
import torch

# batch size
batch_size = 4
learning_rate = 0.001
epochs = 5

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
device = "cpu"
#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#"cpu"

#dataset 몇개를 사용할 것인지 결정 ex)1~4
idx = list(range(1,2))
tr, va, te = utils.load_dataset(idx).call(3)

# dataset loader
trainEEG = utilLoader.EEGLoader(tr, torch.device("cpu"), False)

print("tainLoader")
trainLoader = DataLoader(trainEEG, batch_size = batch_size, shuffle=True)

max_norm = 5
#if in_channels == 1: use one channel, in_channels == 62 : use 62 channel
in_channels = 1
#out_channels means the number of features of representation vector 
out_channels = 512
electrode = 64

model = ERL_DCCN.USRL(electrode, out_channels).to(device)
#Custom Tripletloss
criterion = TripletSigmoidLoss.TripletSigmoidLoss(Kcount=5, scale_int=1, sample_margin=200, device=device)

#use Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=3, gamma=0.5)

# save epoch loss
loss_tr = []
loss_val=[] 
for epoch in range(epochs):
    loss_ep = 0 # add batch loss in epoch
    for batch_idx, batch in enumerate(trainLoader):
        loss_batch = criterion.forward(batch, model, trainEEG)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        loss_ep += loss_batch
    
    loss_v = criterion.get_valloss(model, torch.Tensor(va[0]).to(device))
    loss_tr.append(loss_ep.item()/1000)
    loss_val.append(loss_v.item())
    print("epoch : ", epoch, "   train loss : ",loss_ep.item(),"    val loss : ", loss_v.item())

loss_te = criterion.get_valloss(model, torch.Tensor(te[0]).to(device))
print("test loss : ", loss_te.item())

now = datetime.now()
date = now.strftime('%d%H%m')
savepath = '/DataCommon/jhjeon/model/'+"b" + str(batch_size) + "e" + str(epochs) + "la4" +"c" + str(out_channels) + "lo" +str(int(loss_val[-1])) + ".pth"
#"../USRL/save_model/"+date+ "c" + str(out_channels) + "l" +str(int(loss_val[-1])) +"elec"+ fe + ".pth"
torch.save(model, savepath)

plt.plot(range(epochs), loss_tr, label='Loss', color='red')
plt.plot(range(epochs), loss_val, label='Loss', color='blue')

plt.title('Loss history')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

