# MSE: 5w times, n_neighbors = 5, loss = 0.0353
# MSE: 5w times, n_neighbors = 10, loss = 0.0246
# MSE: 5w times, n_neighbors = 20, loss = 0.0190
# MSE: 5w times, n_neighbors = 30, loss = 0.0167
# MSE: 5w times, n_neighbors = 40, loss = 0.0155
# MSE: 5w times, n_neighbors = 50, loss = 0.0146
# MSE: 5w times, n_neighbors = 60, loss = 0.0139
# MSE: 5w times, n_neighbors = 70, loss = 0.0134
# MSE: 5w times, n_neighbors = 80, loss = 0.0129
# MSE: 5w times, n_neighbors = 90, loss = 0.0126
# MSE: 5w times, n_neighbors = 100, loss = 0.0123
# MSE: 5w times, n_neighbors = 150, loss = 0.0111
# MSE: 5w times, n_neighbors = 200, loss = 0.0103
# MSE: 5w times, n_neighbors = 300, loss = 0.0092
# MSE: 5w times, n_neighbors = 400, loss = 0.0078
# MSE: 5w times, n_neighbors = 480, loss = 0.0067



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

n_neighbors = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 300, 400, 480]
loss = [0.0353, 0.0246, 0.019, 0.0167, 0.0155, 0.0146, 0.0139, 0.0134, 0.0129, 0.0126, 0.0123, 0.0111, 0.0103, 0.0092, 0.0078, 0.0067]
plt.plot(n_neighbors, loss)
plt.xlabel('n_neighbors of ISOMAP')
plt.ylabel('loss')
plt.title('Loss of different n_neighbors')
plt.show()
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
data_ISOMAP = manifold.Isomap(n_neighbors=480, n_components=200, eigen_solver='arpack').fit_transform(data)

# 计算data_pca的合理长度，删除过小的值比如0，选取前100个
# a = []
# for i in range(len(data_ISOMAP[0])):
#     if abs(data_ISOMAP[0][i]) > 0.0001:
#         a.append(data_ISOMAP[0][i])
# print(a)
# print(np.shape(a))

x = torch.from_numpy(data_ISOMAP)
x = x.type(torch.float32)



class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(200, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, 16),
            nn.ReLU(True),
            nn.Linear(16, 12),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(12, 16),
            nn.ReLU(True),
            nn.Linear(16, 32),
            nn.ReLU(True),
            nn.Linear(32, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 200),
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


