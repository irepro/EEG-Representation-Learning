
import numpy as np
import torch

class TripletSigmoidLoss(torch.nn.modules.loss._Loss):
    def __init__(self, Kcount = 5, sample_margin =10, scale_int = 0.2, device = None):
        super().__init__()
        self.Kcount = Kcount # number of negative sample 
        self.margin = sample_margin # min of signal length
        self.scale_int = scale_int
        self.device = device

    def forward(self, batch, encoder, train, **kwargs):
        train_size = train.x_data.size(0)
        max_length = train.x_data.size(2)
        batch_size = batch.size(0)

        #select negative sample of each batch sample
        samples = np.random.choice(
                    train_size, size=(self.Kcount, batch_size),replace=False
                )
        samples = torch.LongTensor(samples)

        #get max length of batches
        lengths_batch = max_length - torch.sum(
                        torch.isnan(batch[:,0]), 1  #detect nan value
                    ).data.cpu().numpy()
        #get max length of negative samples
        lengths_samples = np.empty(
                        (self.Kcount, batch_size), dtype=int
                    )
        for i in range(self.Kcount):
            lengths_samples[i] = max_length - torch.sum(
                torch.isnan(train.x_data[samples[i],0]), -1
            ).data.cpu().numpy()

        #choice randomly lengths of batches and negative samples
        lengths_pos = np.empty(batch_size, dtype=int)
        lengths_neg = np.empty(
            (self.Kcount, batch_size), dtype=int
        )
        for j in range(batch_size):
            lengths_pos[j] = np.random.randint(
                self.margin, lengths_batch[j] + 1 # minimum margin <= length <= maximum length
            )
            for i in range(self.Kcount):
                lengths_neg[i, j] = np.random.randint(
                    self.margin, lengths_samples[i, j] + 1
                )

        #choice randomly beginning points of batches and negative samples
        beginning_pos = np.empty(batch_size, dtype=int)
        beginning_neg = np.empty(
            (self.Kcount, batch_size), dtype=int
        )
        for j in range(batch_size):
            beginning_pos[j] = np.random.randint(
                0, lengths_batch[j] - lengths_pos[j] + 1 # 0 <= beginning points <= maximum length - length of pos or neg
            )
            for i in range(self.Kcount):
                beginning_neg[i, j] = np.random.randint(
                    0, lengths_samples[i, j] - lengths_neg[i, j] + 1
                )

        #choice randomly length of ref
        lengths_ref = np.empty(batch_size, dtype=int)
        for j in range(batch_size):
            lengths_ref[j] = np.random.randint(
                self.margin, lengths_pos[j] + 1# minimum margin <= length < length of pos
            )
        #choice randomly beginning point of ref
        beginning_ref = np.empty(batch_size, dtype=int)
        for j in range(batch_size):
            beginning_ref[j] = np.random.randint(
                beginning_pos[j], lengths_pos[j] - lengths_ref[j] + beginning_pos[j] + 1
            ) # beginning points of pos <= beginning points <= finishing points of pos - length of ref

        #append representation of each signal to list
        ref = []
        for j in range(batch_size):
            input = batch[j,:,beginning_ref[j]:beginning_ref[j]+lengths_ref[j]]
            ref.append(encoder.forward(torch.unsqueeze(input,0).to(self.device)))
        ref = torch.stack(ref)
        ref = torch.squeeze(ref, 1)
        pos = []
        for j in range(batch_size):
            input = batch[j,:,beginning_pos[j]:beginning_pos[j]+lengths_pos[j]]
            pos.append(encoder.forward(torch.unsqueeze(input,0).to(self.device)))
        pos = torch.stack(pos)
        pos = torch.squeeze(pos, 1)
        neg = []
        for j in range(batch_size):
            for i in range(self.Kcount):
                input = train.x_data[samples[i,j],:,beginning_neg[i,j]:beginning_neg[i,j]+lengths_neg[i,j]]
                neg_tensor = [encoder.forward(torch.unsqueeze(input,0).to(self.device))]
                neg.append(torch.stack(neg_tensor))
        neg = torch.stack(neg)
        neg = neg.reshape([batch_size,self.Kcount, -1])

        batch_size = batch.size(0)
        
        loss, loss_tr = 0, 0
        
        # loss between pos and ref
        loss += -torch.mean(torch.nn.functional.logsigmoid(torch.bmm(
                ref.view(batch_size, 1, -1),pos.view(batch_size, -1, 1)))) 
        loss_tr += loss

        loss.backward(retain_graph=True)
        loss = 0
        del pos
        torch.cuda.empty_cache()

        # loss between ref and each negative samples
        for i in range(self.Kcount):
            neg_tensor = neg[:,i,:]
            loss += -self.scale_int*torch.mean(torch.nn.functional.logsigmoid(
                -torch.bmm(ref.view(batch_size, 1, -1),neg_tensor.view(batch_size, -1, 1))))
                
        loss_tr += loss
        loss.backward(retain_graph=True)
        loss = 0
        del neg, ref
        torch.cuda.empty_cache()

        return loss_tr

    # To evaluate validation set loss
    # almost algorithm is same with forward 
    def get_valloss(self,  encoder, val, **kwargs):
        train_size = val.size(0)
        max_length = val.size(2)

        #In validtion set, pick 6 samples and use first sample as pos.
        samples = np.random.choice(
                    train_size, size=(self.Kcount+1),replace=False #pick 6 samples
                )
        samples = torch.LongTensor(samples)

        lengths_samples = np.empty(
                        (self.Kcount+1), dtype=int
                    )
        for i in range(self.Kcount+1):
            lengths_samples[i] = max_length - torch.sum(
                torch.isnan(val[samples[i],0]), 0
            ).data.cpu().numpy()
        #first sample is used as pos
        lengths_pos = np.random.randint(
                self.margin, lengths_samples[0] + 1
            )
        lengths_neg = np.empty(
            (self.Kcount), dtype=int
        )
        #other samples is used as neg
        for j in range(self.Kcount):
            lengths_neg[j] = np.random.randint(
                self.margin, lengths_samples[j+1] + 1
            )

        beginning_pos = np.random.randint(
                0, lengths_samples[0] - lengths_pos + 1
            )
        beginning_neg = np.empty(
            (self.Kcount), dtype=int
        )
        for j in range(self.Kcount):
            beginning_neg[j] = np.random.randint(
                0, lengths_samples[j+1] - lengths_neg[j] + 1
            )

        lengths_ref = np.random.randint(
                self.margin, lengths_pos + 1
            )
        beginning_ref = np.random.randint(
                beginning_pos, lengths_pos - lengths_ref + beginning_pos + 1
            )

        ref = encoder.forward(torch.unsqueeze(val[samples[0],:, beginning_ref:beginning_ref+lengths_ref],0).to(self.device))
        pos = encoder.forward(torch.unsqueeze(val[samples[0],:, beginning_pos:beginning_pos+lengths_pos],0).to(self.device))
        neg = []
        for j in range(self.Kcount):
            input = val[samples[j+1],:, beginning_neg[j]:beginning_neg[j]+lengths_neg[j]]
            neg_tensor = encoder.forward(torch.unsqueeze(input, 0).to(self.device))
            neg.append(neg_tensor)
        neg = torch.stack(neg)

        loss = 0

        loss += -torch.mean(torch.nn.functional.logsigmoid(torch.bmm(
                ref.view(1, 1, -1),pos.view(1, -1, 1))))

        for i in range(self.Kcount):
            neg_tensor = neg[i,:,:]
            loss += -self.scale_int*torch.mean(torch.nn.functional.logsigmoid(
                -torch.bmm(ref.view(1, 1, -1),neg_tensor.view(1, -1, 1))))
         
        del pos, neg, ref
        torch.cuda.empty_cache()

        #In validation stage, don't use loss.backward()
        return loss