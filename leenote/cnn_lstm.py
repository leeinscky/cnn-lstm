# 参考：https://www.kaggle.com/code/orkatz2/cnn-lstm-pytorch-train
from torch import nn
from torch.nn import functional as F

class NeuralNet(nn.Module):
    def __init__(self,cnn, embed_size=1280, LSTM_UNITS=64, DO = 0.3):
        super(NeuralNet, self).__init__()
        self.cnn = cnn.module
        self.cnn.eval().cuda()
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)

        self.linear1 = nn.Linear(LSTM_UNITS*2, LSTM_UNITS*2)
        self.linear2 = nn.Linear(LSTM_UNITS*2, LSTM_UNITS*2)

        self.linear_pe = nn.Linear(LSTM_UNITS*2, 1)
        self.linear_global = nn.Linear(LSTM_UNITS*2, 9)

    def forward(self, x, lengths=None):
        with torch.no_grad():
            embedding = self.cnn.extract_features(x)
            embedding = self.avgpool(embedding)
            b,f,_,_ = embedding.shape
            embedding = embedding.reshape(1,b,f)
        
        self.lstm1.flatten_parameters() # flatten_parameters 作用是为了提高内存的利用率和效率，调用flatten_parameters让parameter的数据存放成contiguous chunk(连续的块)。类似我们调用tensor.contiguous
        h_lstm1, _ = self.lstm1(embedding)
        self.lstm2.flatten_parameters()
        h_lstm2, _ = self.lstm2(h_lstm1)
        
        h_conc_linear1  = F.relu(self.linear1(h_lstm1))
        h_conc_linear2  = F.relu(self.linear2(h_lstm2))
        
        hidden = h_lstm1 + h_lstm2 + h_conc_linear1 + h_conc_linear2

        output = self.linear_pe(hidden)
        output_global = self.linear_global(hidden.mean(1))
        return output,output_global