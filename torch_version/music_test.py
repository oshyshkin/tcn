import sys
sys.path.append('/Users/admin/Documents/Diploma/tcn/')

import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np

from torch_version.model import TCN
from utils import data_generator, get_logger


logger = get_logger("music_test")

parser = argparse.ArgumentParser(description='Sequence Modeling - Polyphonic Music')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA (default: False)')
parser.add_argument('--dropout', type=float, default=0.25,
                    help='dropout applied to layers (default: 0.25)')
parser.add_argument('--clip', type=float, default=0.2,
                    help='gradient clip, -1 means no clip (default: 0.2)')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit (default: 100)')
parser.add_argument('--ksize', type=int, default=5,
                    help='kernel size (default: 5)')
parser.add_argument('--levels', type=int, default=4,
                    help='# of levels (default: 4)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=150,
                    help='number of hidden units per layer (default: 150)')
parser.add_argument('--data', type=str, default='Piano',
                    help='the dataset to run (default: Piano)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')

args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        logger.warning("You have a CUDA device, so you should probably run with --cuda")

logger.info(args)

input_size = 88
X_train, X_valid, X_test = data_generator(args.data)

n_channels = [args.nhid] * args.levels
kernel_size = args.ksize
dropout = args.dropout

model = TCN(input_size=input_size,
            output_size=input_size,
            num_channels=n_channels,
            kernel_size=kernel_size,
            dropout=dropout)


if args.cuda:
    logger.info("Use CUDA")
    model.cuda()

criterion = nn.CrossEntropyLoss()
lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)


def evaluate(X_data):
    model.eval()
    eval_idx_list = np.arange(len(X_data), dtype="int32")
    total_loss = 0.0
    count = 0
    for idx in eval_idx_list:
        data_line = X_data[idx]
        x, y = Variable(data_line[:-1]), Variable(data_line[1:])
        if args.cuda:
            x, y = x.cuda(), y.cuda()
        output = model(x.unsqueeze(0)).squeeze(0)
        loss = -torch.trace(torch.matmul(y, torch.log(output).float().t()) +
                            torch.matmul((1-y), torch.log(1-output).float().t()))
        total_loss += loss.data.item()  # loss.data[0]
        count += output.size(0)
    eval_loss = total_loss / count
    return eval_loss


def train(ep):
    model.train()
    total_loss = 0
    count = 0
    train_idx_list = np.arange(len(X_train), dtype="int32")
    np.random.shuffle(train_idx_list)
    for idx in train_idx_list:
        data_line = X_train[idx]
        x, y = Variable(data_line[:-1]), Variable(data_line[1:])
        if args.cuda:
            x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        output = model(x.unsqueeze(0)).squeeze(0)
        loss = -torch.trace(torch.matmul(y, torch.log(output).float().t()) +
                            torch.matmul((1 - y), torch.log(1 - output).float().t()))
        total_loss += loss.data.item()  # loss.data[0]
        count += output.size(0)

        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        loss.backward()
        optimizer.step()
        if idx > 0 and idx % args.log_interval == 0:
            cur_loss = total_loss / count
            logger.info("Epoch: {:2d}. Idx {} | lr {:.5f} | loss {:.5f}".format(ep, idx, lr, cur_loss))
            total_loss = 0.0
            count = 0


if __name__ == "__main__":
    best_vloss = 1e8
    vloss_list = []
    model_name = "poly_music_{0}.pt".format(args.data)
    logger.info("Start training...")
    for ep in range(1, args.epochs+1):
        train(ep)
        vloss = evaluate(X_valid)
        logger.info("Epoch: {:2d}. Validation loss: {:.5f}".format(ep, vloss))
        # tloss = evaluate(X_test)
        # logger.info("Epoch:{}. Test loss: {:.5f}".format(ep, tloss))
        if vloss < best_vloss:
            with open(model_name, "wb") as f:
                torch.save(model, f)
                logger.info("Saved model!")
            best_vloss = vloss
        if ep > 10 and vloss > max(vloss_list[-3:]):
            lr /= 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        vloss_list.append(vloss)
    logger.info("Training is done")
    logger.info("vloss_list: {}".format(vloss_list))

    logger.info("Training is done")
    model = torch.load(open(model_name, "rb"))
    logger.info("Test model")
    tloss = evaluate(X_test)
    logger.info("Test loss: {:.5f}".format(tloss))
