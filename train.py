import torch
from network import Network,MultiHeadAttention, FeedForwardNetwork
from metric import valid
from torch.utils.data import Dataset
import numpy as np
import argparse
import random
from loss import Loss
import dataloader as loader
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import torch.nn as nn
import torch.nn.functional as F



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# proposed local and global contrative calibration loss
def contrastive_train(epoch, p_sample, adaptive_weight):
    tot_loss = 0.
    mes = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()

        hs, qs, xrs, zs = model(xs)
        loss_list = []
        local_pse = [] # local prediction

        # local contrastive calibration
        for v in range(view):
            for w in range(v+1, view):

                # similarity of the samples in any two views
                sim = torch.exp(torch.mm(hs[v], hs[w].t()))
                sim_probs = sim / sim.sum(1, keepdim=True)

                # pseudo matrix
                Q = torch.mm(qs[v], qs[w].t())
                Q.fill_diagonal_(1)
                pos_mask = (Q >= args.threshold).float()
                Q = Q * pos_mask
                Q = Q / Q.sum(1, keepdims=True)

                local_pse.append(Q)
                loss_contrast_local = - (torch.log(sim_probs + 1e-7) * Q).sum(1)
                loss_contrast_local = loss_contrast_local.mean()

                loss_list.append(loss_contrast_local)
            loss_list.append(mes(xs[v], xrs[v]))

        # global contrastive calibration
        hs_tensor = torch.tensor([]).cuda()

        # obtain global view feature
        for v in range(view):
            hs_tensor = torch.cat((hs[v], hs_tensor), 0)
        hs_tensor = torch.tensor([]).cuda()

        for v in range(view):
            hs_tensor = torch.cat((hs_tensor, torch.mean(hs[v], 1).unsqueeze(1)), 1) # d * v

        # transpose
        hs_tensor = hs_tensor.t()

        # process by the attention
        hs_atten = attention_net(hs_tensor, hs_tensor, hs_tensor) # v * 1

        # learn the view sampling distribution
        p_learn = p_net(p_sample) # v * 1

        # regulatory factor
        r = hs_atten * p_learn
        s_p = nn.Softmax(dim=0)
        r = s_p(r)

        # adjust adaptive weight
        adaptive_weight = r * adaptive_weight

        # obtain fusion feature
        fusion_feature = torch.zeros([hs[0].shape[0], hs[0].shape[1]]).cuda()
        for v in range(view):
            fusion_feature = fusion_feature + adaptive_weight[v].item() * hs[v]

        # obtain fusion feature similarity matrix
        sim_fusion_fea = torch.exp(torch.mm(fusion_feature, fusion_feature.t()))
        sim_fusion_fea_probs = sim_fusion_fea / sim_fusion_fea.sum(1, keepdim=True)

        # obtain fusion feature pseudo
        pse_fusion = model.label_contrastive_module(fusion_feature)
        Q_gloal = torch.mm(pse_fusion, pse_fusion.t())
        Q_gloal.fill_diagonal_(1)
        pos_mask_gloal = (Q_gloal >= args.threshold).float()
        Q_gloal = Q_gloal * pos_mask_gloal
        Q_gloal = Q_gloal / Q_gloal.sum(1, keepdims=True)

        loss_contrast_global = - (torch.log(sim_fusion_fea_probs + 1e-7) * Q_gloal).sum(1)
        loss_contrast_global = loss_contrast_global.mean()
        loss_list.append(loss_contrast_global)

        # same prediction loss
        if view == 2:
            loss_each = F.mse_loss(Q_gloal, local_pse[0])
            loss_list.append(loss_each)
        else:
            for v in range(view):
                loss_each = F.mse_loss(Q_gloal, local_pse[v])
                loss_list.append(loss_each)


        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()

    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))


"""
The backbone network for DealMVC is MFLVC.
We adopt their original code for model training.
"""
# pretrain code from MLFVC
def pretrain(epoch):
    tot_loss = 0.
    criterion = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        _, _, xrs, _ = model(xs)
        loss_list = []
        for v in range(view):
            loss_list.append(criterion(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))
def make_pseudo_label(model, device):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=data_size,
        shuffle=False,
    )
    model.eval()
    scaler = MinMaxScaler()
    for step, (xs, y, _) in enumerate(loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        with torch.no_grad():
            hs, _, _, _ = model.forward(xs)
        for v in range(view):
            hs[v] = hs[v].cpu().detach().numpy()
            hs[v] = scaler.fit_transform(hs[v])

    kmeans = KMeans(n_clusters=class_num, n_init=100)
    new_pseudo_label = []
    for v in range(view):
        Pseudo_label = kmeans.fit_predict(hs[v])
        Pseudo_label = Pseudo_label.reshape(data_size, 1)
        Pseudo_label = torch.from_numpy(Pseudo_label)
        new_pseudo_label.append(Pseudo_label)

    return new_pseudo_label
def match(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    new_y = np.zeros(y_true.shape[0])
    for i in range(y_pred.size):
        for j in row_ind:
            if y_true[i] == col_ind[j]:
                new_y[i] = row_ind[j]
    new_y = torch.from_numpy(new_y).long().to(device)
    new_y = new_y.view(new_y.size()[0])
    return new_y
def fine_tuning(epoch, new_pseudo_label):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=data_size,
        shuffle=False,
    )
    tot_loss = 0.
    cross_entropy = torch.nn.CrossEntropyLoss()
    for batch_idx, (xs, _, idx) in enumerate(loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        _, qs, _, _ = model(xs)
        loss_list = []
        for v in range(view):
            p = new_pseudo_label[v].numpy().T[0]
            with torch.no_grad():
                q = qs[v].detach().cpu()
                q = torch.argmax(q, dim=1).numpy()
                p_hat = match(p, q)  #ei yong
            loss_list.append(cross_entropy(qs[v], p_hat))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    if len(data_loader) == 0:
        print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss ))
    else:
        print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))
    # print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))


Dataname = 'BBCSport'
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument("--temperature_f", type=float, default=0.5)
parser.add_argument("--temperature_l", type=float, default=1.0)
parser.add_argument("--threshold", type=float, default=0.8)
parser.add_argument("--learning_rate", type=float, default=0.0003)
parser.add_argument("--weight_decay", type=float, default=0.)
parser.add_argument("--workers", type=int, default=8)
parser.add_argument("--seed", type=int, default=15)
parser.add_argument("--mse_epochs", type=int, default=300)
parser.add_argument("--con_epochs", type=int, default=100)
parser.add_argument("--tune_epochs", type=int, default=50)
parser.add_argument("--feature_dim", type=int, default=512)
parser.add_argument("--high_feature_dim", type=int, default=512)
parser.add_argument('--num_heads', type=int, default=8)
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--ffn_size', type=int, default=32)
parser.add_argument('--attn_bias_dim', type=int, default=6)
parser.add_argument('--attention_dropout_rate', type=float, default=0.5)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset, dims, view, data_size, class_num = loader.load_data(args.dataset)

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
)

setup_seed(args.seed)

#model init
model = Network(view, dims, args.feature_dim, args.high_feature_dim, class_num, device)
attention_net = MultiHeadAttention(args.hidden_dim, args.attention_dropout_rate, args.num_heads, args.attn_bias_dim)
p_net = FeedForwardNetwork(view, args.ffn_size, args.attention_dropout_rate)
attention_net = attention_net.to(device)
p_net = p_net.to(device)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
optimizer_atten_net = torch.optim.Adam(attention_net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
optimizer_p_net = torch.optim.Adam(p_net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
criterion = Loss(args.batch_size, class_num, args.temperature_f, args.temperature_l, device).to(device)

# init p distribution
p_sample = np.ones(view)
weight_history = []
p_sample = p_sample / sum(p_sample)
p_sample = torch.FloatTensor(p_sample).cuda()


# init adaptive weight
adaptive_weight = np.ones(view)
adaptive_weight = adaptive_weight / sum(adaptive_weight)
adaptive_weight = torch.FloatTensor(adaptive_weight).cuda()
adaptive_weight = adaptive_weight.unsqueeze(1)


# training stage
epoch = 1
while epoch <= args.mse_epochs:
    pretrain(epoch)
    epoch += 1
while epoch <= args.mse_epochs + args.con_epochs:
    contrastive_train(epoch, p_sample, adaptive_weight)
    epoch += 1
new_pseudo_label = make_pseudo_label(model, device)
while epoch <= args.mse_epochs + args.con_epochs + args.tune_epochs:
    fine_tuning(epoch, new_pseudo_label)
    if epoch == args.mse_epochs + args.con_epochs + args.tune_epochs:
        acc, nmi, pur, ari = valid(model, device, dataset, view, data_size, class_num, eval_h=False)
    epoch += 1


