import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
import torch
from util import utilLoader, utils
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

def accuracy_check(label, pred):
    prediction = np.argmax(pred, axis=1)

    compare = np.equal(label, prediction)
    accuracy = np.sum(compare.tolist()) / len(compare.tolist())

    print(prediction)
    print(labels)

    return accuracy, prediction

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 16
learning_rate = 0.0001
epochs = 80

idx = list(range(11,12)) #dataset 몇개를 사용할 것인지. 1~2
tr, va, te = utils.load_dataset(idx).call(5)

# dataset loader
trainEEG = utilLoader.EEGLoader(tr, device, True)

print("trainLoader")
trainLoader = DataLoader(trainEEG, batch_size = batch_size, shuffle=True)

name = "b16e5la4c512lo3elecT"
PATH = "/DataCommon/jhjeon/model/"+name + ".pth"
model = torch.load(PATH, map_location=device)
model.encoder.requires_grad = False
out_channels = name[2:12]
'''
#if in_channels == 1: use one channel, in_channels == 62 : use 62 channel
in_channels = 1
#out_channels means the number of features of representation vector 
out_channels = 512
electrode = 64
#Full_elec means whether you use all of electrodes or not, if true, then you will use all of electrodes
Full_elec = True
model = USRL.USRL(electrode, in_channels, out_channels, Full_elec).to(device)
'''
model.set_Unsupervised(device, False)

max_norm = 5
CrossEL=torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=50, gamma=0.1)

total_loss = []
val_loss = []
for epoch in range(epochs):

    epoch_loss = 0
    for batch, (inputs, labels) in enumerate(trainLoader):
        optimizer.zero_grad()

        outputs = model.forward(inputs)
        loss = CrossEL(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        epoch_loss += loss.clone().item()

        loss = 0
        torch.cuda.empty_cache()
    
    total_loss.append(epoch_loss/12)
    scheduler.step()

    inputs, labels = va

    outputs = model.forward(torch.Tensor(inputs).to(device))
    loss_va = CrossEL(outputs, torch.Tensor(labels).to(device))

    val_loss.append(loss_va.item())
    print("epoch", epoch + 1, "train loss : ", epoch_loss, "val loss:", loss_va.item())

inputs, labels = te
labels = np.argmax(labels, axis=1)

outputs = model.forward(torch.Tensor(inputs).to(device))
epoch_acc, pred = accuracy_check(labels, outputs.cpu().detach().numpy())

print("val acc : ", epoch_acc)

label=[0, 1] # 라벨 설정
plot = confusion_matrix(labels, pred)
print(plot)
score = str(f1_score(labels, pred, average='micro'))
print("f1 acc : ", score)

plt.plot(range(epochs), total_loss, label='Train_Loss', color='red')
plt.plot(range(epochs), val_loss, label='Val_Loss', color='blue')

plt.title('Loss history')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

if float(score) > 0.67:
    savepath = '/DataCommon/jhjeon/trained/'+ name + "acc" + str(score[2:4]) + ".pth"
    print(savepath)
    torch.save(model, savepath)
