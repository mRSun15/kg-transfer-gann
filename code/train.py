import random
import os

import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from test import test
import sys
from model import GANModel

model_root = os.path.join('..', 'models')
cuda = True

lr = 1e-3
batch_size = 128
n_epoch = 100
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)
source_data_root = os.path.join('..', 'data')
target_data_root = source_data_root
source_data_train = os.path.join(source_data_root, 'wiki_training_data.npy')
source_label_train = os.path.join(source_data_root, 'wiki_training_label.npy')
source_data_test = os.path.join(source_data_root, 'wiki_validation_data.npy')
source_label_test = os.path.join(source_data_root, 'wiki_validation_label.npy')
target_data_train = os.path.join(target_data_root, 'nyt_train_data.npy')
target_data_test = os.path.join(target_data_root, 'nyt_test_data.npy')
target_label_test = os.path.join(target_data_root, 'nyt_test_label.npy')
# load data

#source data: wiki_data, target_data:nyt
print("load data")

source_train_data = np.load(source_data_train)
source_train_label = np.load(source_label_train)
source_test_data = np.load(source_data_test)
source_test_label = np.load(source_label_test)
target_train_data = np.load(target_data_train)
target_test_data = np.load(target_data_test)
target_test_label = np.load(target_label_test)
source_dataset = torch.utils.data.TensorDataset(torch.Tensor(source_train_data),torch.Tensor(source_train_label))

dataloader_source = torch.utils.data.DataLoader(
    dataset=source_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8)

dataloader_target = torch.utils.data.DataLoader(
    dataset=target_train_data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8)
# load model
print("load model")
max_length = 100
input_dim = 55
embedding_dim = 100
class_num = len(np.unique(source_train_label))
my_net = GANModel(max_length=max_length,
                  input_dim=input_dim,
                  embedding_dim=100,
                  class_num=class_num)

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
f = open('../output.txt', 'w')
old = sys.stdout
sys.stdout = f
for epoch in range(n_epoch):

    len_dataloader = min(len(dataloader_source), len(dataloader_target))
    data_source_iter = iter(dataloader_source)
    data_target_iter = iter(dataloader_target)

    i = 0
    while i < len_dataloader:

        p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # training model using source data
        data_source = data_source_iter.next()
        s_data, s_label = data_source

        my_net.zero_grad()

        batch_size = len(s_label)
        input_data = torch.FloatTensor(batch_size,max_length,input_dim)
        class_label = torch.LongTensor(batch_size)
        domain_label = torch.zeros(batch_size)
        domain_label = domain_label.long()

        if cuda:
            s_data = s_data.cuda()
            s_label = s_label.cuda()
            input_img = input_data.cuda()
            class_label = class_label.cuda()
            domain_label = domain_label.cuda()

        input_data.resize_as_(s_data).copy_(s_data)
        class_label.resize_as_(s_label).copy_(s_label)
        inputv_img = Variable(input_data)
        classv_label = Variable(class_label)
        domainv_label = Variable(domain_label)

        class_output, domain_output = my_net(input_data=inputv_img, alpha=alpha)
        err_s_label = loss_class(class_output, classv_label)
        err_s_domain = loss_domain(domain_output, domainv_label)

        # training model using target data
        data_target = data_target_iter.next()
        t_data, _ = data_target

        batch_size = len(t_data)

        input_img = torch.FloatTensor(batch_size,max_length,input_dim)
        domain_label = torch.ones(batch_size)
        domain_label = domain_label.long()

        if cuda:
            t_img = t_img.cuda()
            input_img = input_img.cuda()
            domain_label = domain_label.cuda()

        input_img.resize_as_(t_img).copy_(t_img)
        inputv_img = Variable(input_img)
        domainv_label = Variable(domain_label)

        _, domain_output = my_net(input_data=inputv_img, alpha=alpha)
        err_t_domain = loss_domain(domain_output, domainv_label)
        err = err_t_domain + err_s_domain + err_s_label
        err.backward()
        optimizer.step()

        i += 1

        print('epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f'%
              (epoch, i, len_dataloader, err_s_label.cpu().data.numpy(),
                 err_s_domain.cpu().data.numpy(), err_t_domain.cpu().data.numpy()))

    torch.save(my_net, '{0}/DANN_model_epoch_{1}.pth'.format(model_root, epoch))
    test('source_data_wiki', source_test_data,source_test_label, epoch)
    test('test_data_nyt', target_test_data,target_test_label,epoch)

sys.stdout = old
f.close()
print('done')
