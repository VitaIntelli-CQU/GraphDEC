import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from torch.utils.data import TensorDataset, DataLoader
from scipy.spatial import distance
from tqdm import tqdm

from utils import compute_metrics


class AutoEncoder(nn.Module):

    def __init__(self, input_dim: int):
        super(AutoEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, input_dim)

    def forward(self, x):
        encode = F.relu(self.fc1(x))
        decode = F.relu(self.fc2(encode))
        return encode, decode
    

class GS_block(nn.Module):

    def __init__(self, input_dim: int=50, output_dim: int=50):

        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(self.input_dim*2, output_dim))
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, x, adj=None):
    
        if adj == None: adj = torch.ones((x.shape[0], x.shape[0])).float().cuda()
        neigh_feats = self.aggregate(x, adj)
        combined = torch.cat([x.reshape(-1, self.input_dim), neigh_feats.reshape(-1, self.input_dim)], dim=1)
        combined = F.relu(combined @ self.weight)
        combined = F.normalize(combined,2,1).reshape(x.shape[0], -1)
        return combined
        
    def aggregate(self, x, adj=None):

        n = len(adj)
        adj = adj-torch.eye(n, device=adj.device)
        adj /= (adj.sum(1, keepdim=True)+1e-12).float()
        return adj.mm(x)
    


class RandomRowMaskLayer(nn.Module):

    def __init__(self, mask_ratio=0.0):
        super(RandomRowMaskLayer, self).__init__()
        self.mask_ratio = mask_ratio

    def forward(self, x):
        seq_len, _ = x.shape
        mask = torch.rand(seq_len) > self.mask_ratio
        mask = mask.to(x.device)
        x = x * mask.unsqueeze(-1).float()

        return x
    

class Net(nn.Module):

    def __init__(self, input_dim, output_dim):

        super().__init__()

        self.graphlayer_num = 2
        self.channel_num = 2

        self.encoder = nn.Sequential(nn.Linear(input_dim, 512),
                                        nn.LeakyReLU(0.2, inplace=True),
                                        nn.Linear(512, 256),
                                        nn.LeakyReLU(0.2, inplace=True),
                                        nn.Dropout(p=0.3, inplace=False),
                                        )

        self.predictor = nn.Sequential(nn.Linear(256 * self.channel_num, 128),
                                          nn.LeakyReLU(0.2, inplace=True),
                                          nn.Dropout(p=0.2, inplace=False),
                                          nn.Linear(128, output_dim), 
                                          nn.Softmax(dim=1),
                                          )
        
        self.discriminator = nn.Sequential(nn.Linear(256 * self.channel_num, 128),
                                              nn.LeakyReLU(0.2, inplace=True),
                                              nn.Dropout(p=0.2, inplace=False),
                                              nn.Linear(128, 1), 
                                              nn.Sigmoid(),
                                              )
        
        self.graphlayers = nn.ModuleList(nn.ModuleList([GS_block(256, 256) for _ in range(self.graphlayer_num)]) for _ in range(self.channel_num))
        self.masklayer = RandomRowMaskLayer(mask_ratio=0.0)

    def forward(self, x, adj=None, mask=True):

        if mask:
            x = self.masklayer(x)
        embed = self.encoder(x)

        embed_all = None
        for channel in range(self.channel_num):

            embed_current0 = self.graphlayers[channel][0](embed, adj)
            embed_current1 = self.graphlayers[channel][1](embed_current0, adj)
            embed_current = embed_current0 * 0.7 + embed_current1 * 0.3
            
            embed_all = torch.cat((embed_all, embed_current), dim=1) if embed_all is not None else embed_current

        domain_pred = self.discriminator(embed_all)
        frac_pred = self.predictor(embed_all)
        
        return domain_pred, frac_pred, embed_all


class GraphDec(object):

    def __init__(self, train_data, test_data, num_epochs=3000, learning_rate=1e-3):

        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        
        # GraphDec
        self.ae = AutoEncoder(train_data.shape[1]).cuda()
        self.net = Net(train_data.shape[1], train_data.uns['cell_types'].shape[0]).cuda()

        # Train dataset
        self.train_data_x = torch.FloatTensor(train_data.X.astype(np.float32)).cuda()
        self.train_data_y = torch.FloatTensor(np.array([train_data.obs[ctype] for ctype in train_data.uns['cell_types']], dtype=np.float32).T).cuda()
        self.cell_types = train_data.uns['cell_types']

        # Test dataset
        self.test_data_x = torch.FloatTensor(test_data.X.astype(np.float32)).cuda()
        self.test_data_y = np.array([test_data.obs[ctype] for ctype in train_data.uns['cell_types']], dtype=np.float32).T

        if self.train_data_x.shape[0] + self.test_data_x.shape[0] > 24000:
            train_shape = 24000 - self.test_data_x.shape[0]
            self.train_data_x = self.train_data_x[:train_shape]
            self.train_data_y = self.train_data_y[:train_shape]

        # train : validate = 9 : 1
        self.train_shape = int(self.train_data_x.shape[0] * 0.9) 

        # all 
        self.all_data_x = torch.cat((self.train_data_x, self.test_data_x), dim=0)


    def cal_tri_loss(self, train_embed, p, n):

        margin = 0.3
        p_emb = train_embed[p]
        n_emb = train_embed[n]
        L = (torch.cosine_similarity(train_embed, n_emb, dim=1) - torch.cosine_similarity(train_embed, p_emb, dim=1) + margin) * 0.01
        L[L < 0] = 0.
        return L.mean()


    def train(self):

        self.ae.train()
        train_dataset = TensorDataset(self.train_data_x)
        test_dataset = TensorDataset(self.test_data_x)
        train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True)
        ae_optimizer = torch.optim.Adam(self.ae.parameters(), lr=self.learning_rate, weight_decay=1e-4)

        pbar = tqdm(range(200))
        for _ in pbar:
            for _, (x, ) in enumerate(train_dataloader):
                _, decode = self.ae(x)
                loss = nn.MSELoss(reduction='mean')(decode, x)
                ae_optimizer.zero_grad()
                loss.backward()
                ae_optimizer.step()
            for _, (x, ) in enumerate(test_dataloader):
                _, decode = self.ae(x)
                loss = nn.MSELoss(reduction='mean')(decode, x)
                ae_optimizer.zero_grad()
                loss.backward()
                ae_optimizer.step()
            pbar.set_description(f'Embedding        ')

        self.ae.eval()
        encode, _ = self.ae(self.all_data_x)
        adj_all = torch.tensor(distance.cdist(1-encode.detach().cpu(), encode.detach().cpu(), 'cosine')).float().cuda()
        adj_train = adj_all[:self.train_data_x.shape[0], :self.train_data_x.shape[0]]
        adj_test = adj_all[self.train_data_x.shape[0]:, self.train_data_x.shape[0]:]

        self.net.train()
        net_optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        
        train_domain_y = torch.ones(self.train_data_x.shape[0]).unsqueeze(1).cuda()
        test_domain_y = torch.zeros(self.train_data_x.shape[0]).unsqueeze(1).cuda()

        # E-reweight
        new_adj = torch.tensor(distance.cdist(self.train_data_y.cpu(), self.train_data_y.cpu(), 'cosine')).float().cuda() # euclidean
        new_adj = torch.where(new_adj < 0.5, 1.5, 0.8)
        adj_train = adj_train * new_adj

        # tri
        adj_train_tri = distance.cdist(1-self.train_data_y.cpu(), self.train_data_y.cpu(), 'cosine')
        p = np.argmax(adj_train_tri, axis=1)
        n = np.argmin(adj_train_tri, axis=1)

        begin_time = time.time()
        last_loss, last_epoch = 1e9, 0
        # print('Training...')
        for epoch in range(self.num_epochs):
            
            # 1
            train_domain_pred, train_pred, train_embed = self.net(self.train_data_x, adj_train)
            test_domain_pred, _, _ = self.net(self.test_data_x, adj_test)

            # caculate loss 
            tri_loss = self.cal_tri_loss(train_embed, p, n)
            pred_loss = nn.MSELoss()(train_pred[:self.train_shape], self.train_data_y[:self.train_shape])
            disc_loss = nn.BCELoss()(train_domain_pred, train_domain_y[:train_domain_pred.shape[0],]) + nn.BCELoss()(test_domain_pred, test_domain_y[:test_domain_pred.shape[0],])
            loss1 = pred_loss + disc_loss + tri_loss

            # update weights
            net_optimizer.zero_grad()
            loss1.backward()
            net_optimizer.step()

            # 2
            train_domain_pred, _, _ = self.net(self.train_data_x, adj_train)
            test_domain_pred, _, _ = self.net(self.test_data_x, adj_test)

            # caculate loss 
            disc_loss = nn.BCELoss()(test_domain_pred, train_domain_y[:test_domain_pred.shape[0],]) + nn.BCELoss()(train_domain_pred, test_domain_y[:train_domain_pred.shape[0],])
            loss2 = disc_loss

            # update weights
            net_optimizer.zero_grad()
            loss2.backward()
            net_optimizer.step()
            
            # 3
            all_domain_pred, all_pred, all_embed = self.net(self.all_data_x, adj_all)
            train_domain_pred = all_domain_pred[:self.train_data_x.shape[0]]
            test_domain_pred = all_domain_pred[self.train_data_x.shape[0]:]

            # caculate loss 
            tri_loss = self.cal_tri_loss(all_embed[:self.train_data_x.shape[0]], p, n)
            pred_loss = nn.MSELoss()(all_pred[:self.train_shape], self.train_data_y[:self.train_shape])
            loss_val = nn.MSELoss()(all_pred[:self.train_data_x.shape[0]][self.train_shape:], self.train_data_y[self.train_shape:])
            loss3 = pred_loss * 0.1 + tri_loss

            # update weights
            net_optimizer.zero_grad()
            loss3.backward()
            net_optimizer.step()

            if loss_val < last_loss :
                torch.save(self.net.state_dict(), 'model_state_dict.pth')
                last_loss = loss_val
                last_epoch = epoch
            elif epoch > 1500 and epoch - last_epoch > 100:
                print('early break')
                np.save('embed.npy', all_embed.detach().cpu().numpy())
                break

            if (epoch+1) % 100 == 0:
                print('============= Epoch {:02d}/{:02d} ============='.format(epoch + 1, self.num_epochs))
                pred = self.prediction(self.test_data_x, adj_test)
                epoch_ccc, epoch_rmse, epoch_corr = compute_metrics(pred, self.test_data_y)
                print('CCC:', round(epoch_ccc, 3), ', RMSE:', round(epoch_rmse, 3), ', Pearson:', round(epoch_corr, 3), ', Time:', round(time.time()-begin_time, 2), 's')

        self.net.load_state_dict(torch.load('model_state_dict.pth'))
        pred = self.prediction(self.test_data_x, adj_test)

        del adj_train
        del adj_test


    def prediction(self, test_data_x, adj_test=None):
        
        self.net.eval()
        if adj_test == None:
            adj_test = torch.tensor(distance.cdist(1-test_data_x.cpu(), test_data_x.cpu(), 'cosine')).float().cuda()
        _, pred, _ = self.net(test_data_x, adj_test)

        # E-reweight
        pred = pred.detach().cpu()
        new_adj = torch.tensor(distance.cdist(pred, pred, 'cosine')).float().cuda() # euclidean
        new_adj = torch.where(new_adj < 0.5, 1.5, 0.8)
        new_adj = adj_test * new_adj
        _, pred, _ = self.net(test_data_x, new_adj)

        return pred.detach().cpu().numpy()
