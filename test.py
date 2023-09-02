import torch
from metric import valid
from torch.utils.data import Dataset
import argparse
import dataloader as loader


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

model = torch.load('./models/BBCSport.pth')

acc, nmi, pur, ari = valid(model, device, dataset, view, data_size, class_num, eval_h=False)