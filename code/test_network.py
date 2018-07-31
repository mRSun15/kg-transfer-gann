import random
import os

import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from test import test
import sys
from model import GANModel
from model import PCNNGANmodel

model_root = os.path.join('..', 'models')
cuda = True

lr = 1e-4
batch_size = 128
n_epoch = 100
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)
source_data_root = os.path.join('..', 'data')
# target_data_root = source_data_root
source_data_train = os.path.join(source_data_root, 'nyt_train_data_.npy')
source_label_train = os.path.join(source_data_root, 'nyt_train_label_.npy')
source_mask_train = os.path.join(source_data_root, 'nyt_train_mask.npy')
source_data_test = os.path.join(source_data_root, 'nyt_test_data_.npy')
source_label_test = os.path.join(source_data_root, 'nyt_test_label_.npy')
source_mask_test = os.path.join(source_data_root, 'nyt_test_mask.npy')
# target_data_train = os.path.join(target_data_root, 'nyt_train_data.npy')
# target_data_test = os.path.join(target_data_root, 'nyt_test_data.npy')
# target_label_test = os.path.join(target_data_root, 'nyt_test_label.npy')
# load data

#source data: wiki_data, target_data:nyt
print("load data")

source_train_data = np.load(source_data_train)
source_train_label = np.load(source_label_train)
source_train_mask = np.load(source_mask_train)
source_test_data = np.load(source_data_test)
source_test_label = np.load(source_label_test)
source_test_mask = np.load(source_mask_test)
source_dataset = torch.utils.data.TensorDataset(torch.Tensor(source_train_data),torch.Tensor(source_mask_train),torch.LongTensor(source_train_label))

dataloader_source = torch.utils.data.DataLoader(
    dataset=source_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8)

# load model
print("load model")
max_length = 100
input_dim = 55
embedding_dim = 100
class_num = np.max(source_train_label)+1
print("Total relations class number: ",class_num)
# my_net = GANModel(max_length=max_length,
#                   input_dim=input_dim,
#                   embedding_dim=100,
#                   class_num=class_num)
my_net = PCNNGANmodel(max_length = max_length,
                      input_dim=1,
                      embedding_dim=embedding_dim,
                      class_num = class_num)

# setup optimizer

optimizer = optim.Adam(my_net.parameters(), lr=lr)

loss_class = torch.nn.NLLLoss()
loss_domain = torch.nn.NLLLoss()

if cuda:
    my_net = my_net.cuda()
    loss_class = loss_class.cuda()
    loss_domain = loss_domain.cuda()

for p in my_net.parameters():
    p.requires_grad = True
print("train")
# training
# f = open('../test_output.txt', 'w')
# old = sys.stdout
# sys.stdout = f
for epoch in range(n_epoch):

    len_dataloader = len(dataloader_source)
    data_source_iter = iter(dataloader_source)

    i = 0
    epoch_err_s_label = 0
    while i < len_dataloader:

        p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # training model using source data
        data_source = data_source_iter.next()
        s_data, s_mask,s_label = data_source

        my_net.zero_grad()

        batch_size = len(s_label)
        input_data = torch.FloatTensor(batch_size,max_length,input_dim)
        input_mask = torch.FloatTensor(batch_size,max_length,3)
        class_label = torch.LongTensor(batch_size)
        domain_label = torch.zeros(batch_size)
        domain_label = domain_label.long()

        if cuda:
            s_data = s_data.cuda()
            s_label = s_label.cuda()
            s_mask = s_mask.cuda()
            input_data = input_data.cuda()
            input_mask = input_mask.cuda()
            class_label = class_label.cuda()
            domain_label = domain_label.cuda()

        input_data.resize_as_(s_data).copy_(s_data)
        input_mask.resize_as_(s_mask).copy_(s_mask)
        class_label.resize_as_(s_label).copy_(s_label)
        inputv_data = Variable(input_data)
        inputv_mask = Variable(input_mask)
        classv_label = Variable(class_label)
        domainv_label = Variable(domain_label)

        class_output, domain_output = my_net(input_data=inputv_data, mask=input_mask)
        err_s_label = loss_class(class_output, classv_label)
        epoch_err_s_label += err_s_label.cpu().data.numpy()
        err = err_s_label
        err.backward()
        optimizer.step()

        i += 1
    epoch_err_s_label = epoch_err_s_label * 1.0 / len_dataloader
    print('epoch: %d,  err_s_label: %f'%(epoch, epoch_err_s_label))

    torch.save(my_net, '{0}/PCNN_model_epoch_{1}.pth'.format(model_root, epoch))
    test('source_data_nyt', source_test_data,source_test_mask,source_test_label, epoch)

# sys.stdout = old
# f.close()
print('done\n')