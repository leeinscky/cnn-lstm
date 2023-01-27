import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from torchvision.models import resnet18, resnet101


class CNNLSTM(nn.Module):
    def __init__(self, num_classes=2):
        super(CNNLSTM, self).__init__()
        self.resnet = resnet101(pretrained=True)
        self.resnet.fc = nn.Sequential(nn.Linear(self.resnet.fc.in_features, 300))
        self.lstm = nn.LSTM(input_size=300, hidden_size=256, num_layers=3) # input_size表示输入的特征维度(embedding_size)，hidden_size表示隐藏层的维度，num_layers表示LSTM的层数
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)
       
    def forward(self, x_3d): # 整个逻辑是将图片序列（一共16帧，这个16是自己设置的）输入到CNN中，得到每一帧的特征，然后将这些特征输入到LSTM中，得到最后的分类结果
        # print(f'[cnnlstm.py] 正在执行CNNLSTM类的forward函数，输入的x_3d的shape为{x_3d.shape}') # torch.Size([8, 16, 3, 150, 150]) 8表示batch_size，16表示16帧，3表示3个通道，150表示150*150的图像
        hidden = None
        for t in range(x_3d.size(1)): # x_3d.size(1)表示16帧，即将16帧的图像输入到CNN中，得到16帧的特征, t表示第t帧
            with torch.no_grad():
                # print(f'[cnnlstm.py] 正在执行CNNLSTM类的forward函数，x_3d[:, t, :, :, :].shape={x_3d[:, t, :, :, :].shape}, t={t}') 
                # 打印结果：x_3d[:, t, :, :, :].shape=torch.Size([8, 3, 150, 150]) 8表示batch_size，3表示3个通道，150表示150*150的图像
                
                # x是每一帧的特征，x.shape = [batch_size, 300]
                x = self.resnet(x_3d[:, t, :, :, :])  # x_3d[:, t, :, :, :] 表示取出第t帧的图像
                
                # print(f'[cnnlstm.py] 正在执行CNNLSTM类的forward函数，经过resnet后，x.shape={x.shape}, 输入到LSTM的数据: x.unsqueeze(0).shape={x.unsqueeze(0).shape}, t={t}') 
                # x.shape=torch.Size([8, 300]), 输入到LSTM的数据: x.unsqueeze(0).shape=torch.Size([1, 8, 300])，1表示1帧，8表示batch_size，300表示特征维度  
                
            # LSTM的输入维度为：（seq_len, batch_size, input_size），seq_len表示序列长度=1，batch_size表示batch_size=8，input_size表示特征维度=300
            out, hidden = self.lstm(x.unsqueeze(0), hidden) # unsqueeze(0)表示在第0维度增加一个维度，即将x的shape从[8, 300]变为[1, 8, 300]
            # print(f'[cnnlstm.py] 正在执行CNNLSTM类的forward函数，经过lstm后，out.shape={out.shape}, len(hidden)={len(hidden)}, hidden[0].shape={hidden[0].shape}, hidden[1].shape={hidden[1].shape}, t={t}')
            # 打印结果：out.shape=torch.Size([1, 8, 256]), len(hidden)=2, hidden[0].shape=torch.Size([3, 8, 256]), hidden[1].shape=torch.Size([3, 8, 256]), t=0
            # 打印结果解析：hidden[0] 表示h_n，hidden[1]表示c_n，h_n和c_n的shape都是[3, 8, 256]，3表示3层LSTM，8表示batch_size，256表示LSTM的隐藏层维度

        x = self.fc1(out[-1, :, :])
        # print(f'[cnnlstm.py] 正在执行CNNLSTM类的forward函数，经过fc1后，x.shape={x.shape}') # x.shape=torch.Size([8, 128])
        x = F.relu(x)
        # print(f'[cnnlstm.py] 正在执行CNNLSTM类的forward函数，经过relu后，x.shape={x.shape}') # x.shape=torch.Size([8, 128])
        x = self.fc2(x)
        # print(f'[cnnlstm.py] 正在执行CNNLSTM类的forward函数，经过fc2后，最终返回的x.shape={x.shape}') # x.shape=torch.Size([8, 2])
        return x


"""  维度变化总结
    1、resnet的输入数据：torch.Size([8, 3, 150, 150]) 8表示batch_size，3表示3个通道，150表示150*150的图像
    2、经过CNN模型resnet处理后，输出数据：torch.Size([8, 300])，8表示batch_size，300表示特征维度。 300是在初始化resnet时定义的：self.resnet.fc = nn.Sequential(nn.Linear(self.resnet.fc.in_features, 300))
    
    进行步骤3前的数据预处理：将第2步的输出数据torch.Size([8, 300])变为：torch.Size([1, 8, 300])，1表示1帧，8表示batch_size，300表示特征维度
    
    3、LSTM的输入数据：
        第1个参数：input，torch.Size([1, 8, 300])，维度为：（seq_len, batch_size, input_size），seq_len表示序列长度=1，batch_size表示batch_size=8，input_size表示特征维度=300
        第2个参数：hidden，包含h_0和c_0，初始时都为None
    4、经过LSTM处理后，输出数据：
        第1个输出：out，torch.Size([1, 8, 256])，维度为：（seq_len, batch_size, hidden_size），seq_len表示序列长度=1，batch_size表示batch_size=8，hidden_size表示LSTM的隐藏层维度=256
        第2个输出：hidden，包含h_n和c_n，维度为：[3, 8, 256]，即（num_layers * num_directions, batch_size, hidden_size），num_layers表示LSTM的层数=3，num_directions表示LSTM的方向=1，batch_size表示batch_size=8，hidden_size表示LSTM的隐藏层维度=256
"""



""" 参考LSTM的测试代码加深理解
假设有100个句子（sequence）,每个句子里有5个词，batch_size=3，embedding_size=10

此时，LSTM forward函数的第一个参数inputs的各个参数为：
    1、seq_len=5
    2、batch=batch_size=3
    3、input_size=embedding_size=10

另外设置hidden_size=20, num_layers=1
————————————————
原文链接：https://blog.csdn.net/weixin_42713739/article/details/108746797

"""

"""
rnn = nn.LSTM(input_size=10,hidden_size=20,num_layers=2) #输入向量维数10, 隐藏元维度20, 2个LSTM层串联(若不写则默认为1）
inputs = torch.randn(5,3,10) #输入（seq_len, batch_size, input_size） 序列长度为5(每个句子里有5个词) batch_size为3 输入维度为10
h_0 = torch.randn(2,3,20) #(num_layers * num_directions, batch, hidden_size)  num_layers = 2 ，batch_size=3 ，hidden_size = 20,如果LSTM的bidirectional=True,num_directions=2,否则就是１，表示只有一个方向
c_0 = torch.randn(2,3,20) #c_0和h_0的形状相同，它包含的是在当前这个batch_size中的每个句子的初始细胞状态。h_0,c_0如果不提供，那么默认是０
num_directions=1 #   因为是单向LSTM

#输出格式为(output,(h_n,c_n))
output,(h_n,c_n) = rnn(inputs,(h_0,c_0))#输入格式为lstm(input,(h_0, c_0))
print("out:", output.shape)
print("h_n:", h_n.shape)
print("c_n:", c_n.shape)

# 输出结果：
# out: torch.Size([5, 3, 20])
# h_n: torch.Size([2, 3, 20])
# c_n: torch.Size([2, 3, 20])

"""