# euclidean dist: 5w times, dist = 0.5376

import torch
import numpy as np
import scipy.io as scio
from torch import nn
from torch.autograd import Variable
from sklearn.decomposition import KernelPCA

if __name__ == '__main__':
    data6_14 = scio.loadmat('data6_14.mat')
    data1 = data6_14['data6_14']
    data = data1.T  # 500x1074

    sklearn_kpca = KernelPCA(n_components=500, kernel='linear', gamma=15)
    data_kpca = sklearn_kpca.fit_transform(data)

    train_data = data_kpca[:, :100]  # 500x100
    x1 = train_data[0]
    x = x1.reshape(10, 10)
    x = torch.from_numpy(x)
    x = torch.reshape(x, (1, 1, 10, 10))
    x = x.type(torch.float32)

    num_epochs = 50000
    learning_rate = 1e-3


    def cos_sim(vector_a, vector_b):
        vector_a = np.mat(vector_a)
        vector_b = np.mat(vector_b)
        num = float(vector_a * vector_b.T)
        denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
        sim = num / denom
        return sim



    class autoencoder(nn.Module):
        def __init__(self):
            super(autoencoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 8, 2),
                nn.ReLU(True),
                nn.MaxPool2d(2, stride=1),  # [1, 16, 39, 58]
                nn.Conv2d(8, 4, 2),
                nn.ReLU(True),
                nn.MaxPool2d(2, stride=1),  # [1, 8, 12, 18]
                nn.Conv2d(4, 2, 2),
                nn.ReLU(True),
                nn.MaxPool2d(2, stride=1),  # [1, 4, 9, 15]
                nn.Conv2d(2, 1, (1, 2)),
                nn.ReLU(True)
            )
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(1, 2, (1, 2), stride=2),  # [1, 1, 5, 11]
                nn.ReLU(True),
                nn.ConvTranspose2d(2, 4, 2, stride=1),  # [1, 2, 7, 13]
                nn.ReLU(True),
                nn.ConvTranspose2d(4, 8, 2, stride=1),  # [1, 4, 9, 15]
                nn.ReLU(True),
                nn.ConvTranspose2d(8, 1, 2, stride=1),
                nn.ReLU(True),
                nn.ConvTranspose2d(1, 1, (1, 2), stride=1),
                nn.ReLU(True)
            )

        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)

    model = autoencoder().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=1e-5)
    loss = 0

    for epoch in range(num_epochs):
        # forward
        output = model(x)
        x_sqz = x.squeeze()
        output_sqz = output.squeeze()
        x1 = x_sqz.reshape(10*10)
        output1 = output_sqz.reshape(5*20)
        loss = torch.dist(x[0], output[0], 5)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(epoch+1)
        print(loss)
        # print('epoch [{}/{}], loss:{:.4f}'
        #       .format(epoch+1, num_epochs, loss))