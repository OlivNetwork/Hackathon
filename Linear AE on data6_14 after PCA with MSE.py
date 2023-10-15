# MSE: 5w times, loss = 0.0085
# MSE: 5w times,svd = full, loss = 0.0067
# MSE: 5w times,svd = randomized, loss = 0.0076
# MSE: 5w times,svd = auto, loss = 0.0070

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import manifold
import torch
import numpy as np
import scipy.io as scio
from torch import nn
from torch.autograd import Variable
from sklearn.decomposition import PCA

data6_14 = scio.loadmat('data6_14.mat')
data1 = data6_14['data6_14']
data = data1.T  # 500x1074
print('the shape of train_data is: ', data.shape)


# data1d = train_data[0]
# plt.plot(data1d)
# plt.show()


# KPCA: different kernel's performance test
# for index, kernel in enumerate(['linear', 'poly', 'rbf', 'cosine']):
#     plt.subplot(2, 2, index+1)
#     sklearn_kpca = KernelPCA(n_components=500, kernel=kernel, gamma=15)
#     DATA_KPCA = sklearn_kpca.fit_transform(data)
#     KPCA1D = DATA_KPCA[0]
#     plt.plot(KPCA1D)
#     plt.text(.011, .99, ('kernel: ', kernel), transform=plt.gca().transAxes, size=10, horizontalalignment='left')
#     plt.text(.99, .011, 'KPCA', transform=plt.gca().transAxes, size=10, horizontalalignment='right')
# plt.show()


# linear's performance is the best
pca = PCA(n_components=500, svd_solver='randomized')
data_pca = pca.fit_transform(data)  # 500x500


train_data = data_pca[:, :100]  # 500x100
x = torch.from_numpy(train_data)
x = x.type(torch.float32)
#
#
#
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(100, 80),
            nn.ReLU(True),
            nn.Linear(80, 40),
            nn.ReLU(True),
            nn.Linear(40, 20),
            nn.ReLU(True),
            nn.Linear(20, 12),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(12, 20),
            nn.ReLU(True),
            nn.Linear(20, 40),
            nn.ReLU(True),
            nn.Linear(40, 80),
            nn.ReLU(True),
            nn.Linear(80, 100),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = x.to(device)

num_epochs = 50000
learning_rate = 0.0001
model = autoencoder().cuda()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
loss = nn.MSELoss()

for epoch in range(num_epochs):
    input = Variable(x).cuda()
    # forward
    output = model(x)
    data_loss = loss(x, output)
    # backward
    optimizer.zero_grad()
    data_loss.backward()
    optimizer.step()
    print(epoch+1)
    print(data_loss)



