# MSE: 5w times, loss = 0.0112

import torch
import numpy as np
import scipy.io as scio
from torch import nn
from torch.autograd import Variable
from sklearn.decomposition import PCA

if __name__ == '__main__':
    data6_14 = scio.loadmat('data6_14.mat')
    data1 = data6_14['data6_14']
    data = data1.T  # 500x1074

    pca = PCA(n_components=500)
    data_pca = pca.fit_transform(data)  # 500x500
    # sklearn_kpca = KernelPCA(n_components=500, kernel='linear', gamma=15)
    # data_kpca = sklearn_kpca.fit_transform(data)

    train_data = data_pca[:, :100]  # 500x100
    x = torch.from_numpy(train_data)
    x = torch.reshape(x, (1, 1, 500, 100))
    x = x.type(torch.float32)


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
                nn.Conv2d(1, 32, (10, 5)),
                nn.ReLU(True),
                nn.MaxPool2d(5, stride=2),
                nn.Conv2d(32, 16, (8, 5)),
                nn.ReLU(True),
                nn.MaxPool2d(5, stride=1),
                nn.Conv2d(16, 8, (6, 4)),
                nn.ReLU(True),
                nn.MaxPool2d(4, stride=1),
                nn.Conv2d(8, 4, 4),
                nn.ReLU(True),
                nn.MaxPool2d(3, stride=1),
                nn.Conv2d(4, 2, 2),
                nn.ReLU(True),
                nn.MaxPool2d(2, stride=1),
                nn.Conv2d(2, 1, 2),
                nn.ReLU(True),
                nn.MaxPool2d(2, stride=1)
            )
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(1, 2, 16, stride=2),
                nn.ReLU(True),
                nn.ConvTranspose2d(2, 4, (14, 10), stride=1),
                nn.ReLU(True),
                nn.ConvTranspose2d(4, 8, (12, 8), stride=1),
                nn.ReLU(True),
                nn.ConvTranspose2d(8, 16, (12, 8), stride=1),
                nn.ReLU(True),
                nn.ConvTranspose2d(16, 32, 12, stride=1),
                nn.ReLU(True),
                nn.ConvTranspose2d(32, 1, 12, stride=1),
                nn.ReLU(True),
                nn.Conv2d(1, 1, (4, 6)),
                nn.Tanh()
            )

        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)

    num_epochs = 50000
    learning_rate = 1e-3

    model = autoencoder().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=1e-5)
    loss = nn.MSELoss()

    for epoch in range(num_epochs):
        # forward
        output = model(x)
        data_loss = loss(x, output)
        # backward
        optimizer.zero_grad()
        data_loss.backward()
        optimizer.step()
        print(epoch+1)
        print(data_loss)
        # print('epoch [{}/{}], loss:{:.4f}'
        #       .format(epoch+1, num_epochs, loss))