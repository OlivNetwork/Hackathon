# MSE: 5w times, loss = 0.0020


import torch
import numpy as np
import scipy.io as scio
from torch import nn
from torch.autograd import Variable
from sklearn.decomposition import PCA
from sklearn import manifold
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

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
data_LLE, err = manifold.locally_linear_embedding(data, n_neighbors=40, n_components=300)
print(data_LLE.shape)
# data_pca = pca.fit_transform(data)  # 500x500

# a = []
# for i in range(len(data_pca[0])):
#     if abs(data_pca[0][i]) > 0.0001:
#         a.append(data_pca[0][i])
# print(a)
# print(np.shape(a))


# train_data = data_pca[:, :100]  # 500x100
x = torch.from_numpy(data_LLE)
x = x.type(torch.float32)
# #
# #
# #
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(300, 200),
            nn.ReLU(True),
            nn.Linear(200, 100),
            nn.ReLU(True),
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
            nn.ReLU(True),
            nn.Linear(100, 200),
            nn.ReLU(True),
            nn.Linear(200, 300),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# x = x.to(device)
#
# num_epochs = 1000
# learning_rate = 0.0001
# model = autoencoder().cuda()
# # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
# loss = nn.MSELoss()
#
# for epoch in range(num_epochs):
#     input = Variable(x).cuda()
#     # forward
#     output = model(x)
#     data_loss = loss(x, output)
#     # backward
#     optimizer.zero_grad()
#     data_loss.backward()
#     optimizer.step()
#     print(epoch+1)
#     print(data_loss)




def loss_function(recon_x, x, mu, logvar):
    """
    : param recon_x: generate image
    : param x: original image
    : param mu: latent mean of z
    : param logvar: latent log variance of z
    """
    loss = nn.MSELoss()
    # loss = nn.BCELoss(reduction='sum')
    reconstruction_loss = loss(recon_x, x)
    KL_divergence = -0.5 * torch.sum(1 + logvar - torch.exp(logvar) - mu**2)
    print(reconstruction_loss, KL_divergence)
    return reconstruction_loss + KL_divergence


class VAE1(nn.Module):
    def __init__(self):
        super(VAE1, self).__init__()
        self.fc1 = nn.Linear(300, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 80)
        self.fc4 = nn.Linear(80, 40)
        self.fc5_mean = nn.Linear(40, 12)
        self.fc5_logvar = nn.Linear(40, 12)
        self.fc6 = nn.Linear(12, 40)
        self.fc7 = nn.Linear(40, 80)
        self.fc8 = nn.Linear(80, 100)
        self.fc9 = nn.Linear(100, 200)
        self.fc10 = nn.Linear(200, 300)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        h4 = F.relu(self.fc4(h3))
        return self.fc5_mean(h4), self.fc5_logvar(h4)

    def reparametrization(self, mu, logvar):
        std = 0.5 * torch.exp(logvar)
        z = torch.randn(std.size()) * std +mu
        return z

    def decode(self, z):
        h5 = F.relu(self.fc6(z))
        h6 = F.relu(self.fc7(h5))
        h7 = F.relu(self.fc8(h6))
        h8 = F.relu(self.fc9(h7))
        return torch.sigmoid(self.fc10(h8))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrization(mu, logvar)
        return self.decode(z), mu, logvar

num_epochs1 = 1000
learning_rate1 = 0.0003
all_loss = 0
vae1 = VAE1()
optimizer1 = torch.optim.Adam(vae1.parameters(), lr=learning_rate1)
p11 = torch.from_numpy(data_LLE)
p11 = p11.type(torch.float32)

for epoch in range(num_epochs1):
    gen_imgs, mu, logvar = vae1(p11)
    loss = loss_function(gen_imgs, p11, mu, logvar)

    optimizer1.zero_grad()
    loss.backward()
    optimizer1.step()
    if epoch == num_epochs - 1:
        mean1 = mu
        sigma1 = logvar
    #print('Epoch {}, loss: {:.6f}'.format(epoch, loss))
    print(epoch+1)
    print('loss: ', loss)
    print('mu: ', mu)
    print('logvar: ', logvar)


def normal_distribution(x, mean, sigma):
    return np.exp(-1*((x-mean)**2)/(2*(sigma**2)))/(math.sqrt(2*np.pi) * sigma)


mean1 = mean1.detach().numpy()
m = mean1[0][0]
# print('the shape of mean1 is: ', np.shape(mean1))
# print(mean1[0][0])
sigma1 = sigma1.detach().numpy()
s = sigma1[0][0]
# print('mean1: ', mean1)
xx1 = np.linspace(m - 6*s, m + 6*s, 100)
yy1 = normal_distribution(xx1, m, s)
plt.plot(xx1, yy1, 'r')
plt.grid()
plt.show()