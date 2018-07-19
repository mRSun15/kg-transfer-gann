import os
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.autograd import Variable


def test(dataset_name,data, label, epoch):

    model_root = os.path.join('..', 'models')
    cuda = True
    cudnn.benchmark = True
    batch_size = 128
    max_length = 100
    input_dim = 55
    alpha = 0

    """load data"""
    dataset = torch.utils.data.TensorDataset(torch.Tensor(data), torch.LongTensor(label))
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8
    )

    """ training """

    my_net = torch.load(os.path.join(
        model_root, 'DANN_model_epoch_' + str(epoch) + '.pth'
    ))
    my_net = my_net.eval()

    if cuda:
        my_net = my_net.cuda()

    len_dataloader = len(dataloader)
    data_target_iter = iter(dataloader)

    i = 0
    n_total = 0
    n_correct = 0

    while i < len_dataloader:

        # test model using target data
        data_target = data_target_iter.next()
        t_img, t_label = data_target

        batch_size = len(t_label)

        input_img = torch.FloatTensor(batch_size, max_length, input_dim)
        class_label = torch.LongTensor(batch_size)

        if cuda:
            t_img = t_img.cuda()
            t_label = t_label.cuda()
            input_img = input_img.cuda()
            class_label = class_label.cuda()

        input_img.resize_as_(t_img).copy_(t_img)
        class_label.resize_as_(t_label).copy_(t_label)
        inputv_img = Variable(input_img)
        classv_label = Variable(class_label)

        class_output, _ = my_net(input_data=inputv_img, alpha=alpha)
        pred = class_output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(classv_label.data.view_as(pred)).cpu().sum()
        n_total += batch_size
        i += 1

    accu = n_correct.double()/ n_total

    print('epoch: %d, accuracy of the %s dataset: %f' % (epoch, dataset_name, accu))
