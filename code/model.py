import torch.nn as nn
import torch
from functions import ReverseLayerF


class PCNNGANmodel(nn.Module):
    def __init__(self, max_length, input_dim, embedding_dim, class_num ,kernel_size = 3, stride_size = 1):
        super(PCNNGANmodel, self).__init__()
        self.max_length = max_length
        self.input_dim =input_dim
        self.class_num = class_num
        self.hidden_size = embedding_dim
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv', nn.Conv2d(1, embedding_dim, kernel_size=(1,kernel_size),
                                                    stride = (1,stride_size),padding=(0,1)))
        self.feature.add_module('f_bn', nn.BatchNorm2d(embedding_dim))
        self.feature_ = nn.Sequential()
        self.feature_.add_module('f_relu', nn.ReLU(True))
        self.feature_.add_module('f_drop', nn.Dropout())

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(self.hidden_size*3, self.class_num))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax())

    def forward(self,input_data, mask):
        input_data = input_data.view(input_data.shape[0], 1, self.max_length, self.input_dim)
        feature1 = self.feature(input_data)
        feature2 = self.piece_pooling(feature1, mask=mask)
        feature = self.feature_(feature2)
        class_output = self.class_classifier(feature)

        return class_output

    def piece_pooling(self, x, mask):
        x = x.view(-1,self.max_length,self.hidden_size, 1)
        new_mask = mask.view(-1,self.max_length, 1,3)*100
        pooling_x,_ = torch.max(x+new_mask,1)-100
        return pooling_x.view(-1,self.hidden_size*3)



class GANModel(nn.Module):

    def __init__(self, max_length, input_dim, embedding_dim, class_num):
        super(GANModel, self).__init__()
        self.max_length = max_length
        self.input_dim = input_dim
        # self.embedding_dim = embedding_dim
        self.class_num = class_num
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(1, 64, kernel_size=(5,5)))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(64))
        self.feature.add_module('f_pool1', nn.MaxPool2d((4,3)))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_conv2', nn.Conv2d(64, 30, kernel_size=(5,3)))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(30))
        self.feature.add_module('f_drop1', nn.Dropout2d())
        self.feature.add_module('f_pool2', nn.MaxPool2d((4,3)))
        self.feature.add_module('f_relu2', nn.ReLU(True))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(30 * 5 * 5, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, self.class_num))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax())

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(30 * 5 * 5, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax())

    def forward(self, input_data, alpha):
        input_data = input_data.view(input_data.data.shape[0],1,self.max_length, self.input_dim)
        feature = self.feature(input_data)
        feature = feature.view(-1, 30*5*5)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output
