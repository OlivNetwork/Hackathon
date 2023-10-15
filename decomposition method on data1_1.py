# self-def-PCA， sklearn-PCA, KPCA, ISOMAP, LLE, NMF, UMAP

import torch
import scipy.io as scio
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.decomposition import KernelPCA
import numpy as np
from sklearn import manifold
from sklearn.decomposition import NMF
import umap
import re
import pandas as pd
import matplotlib.pyplot as plt


# *******************  self-def-PCA  *******************
def pca_def(datasets, k):
    x = datasets
    x_mean = torch.mean(x, 0)
    x = x - x_mean.expand_as(x)
    u, s, v = torch.svd(torch.t(x))
    return torch.mm(x, u[:, :k])


k = 27
data1_1 = scio.loadmat('data1_1.mat')
data1 = data1_1['data1']
data1 = torch.from_numpy(data1)
data = torch.t(data1)
print('the shape of data11 is: ', data.shape)

data_pca_def = pca_def(data, k)
print('data_pca_def is: ', data_pca_def)
print('the shape of data_pca_def is: ', data_pca_def.shape)


# *******************  sklearn-PCA  *******************
pca = PCA(n_components=27)
data_pca = pca.fit_transform(data)
print('data_pca is: ', data_pca)
print('the shape of data_pca is: ', data_pca.shape)
np.savetxt('data_pca.out', data_pca, delimiter=',')


# *******************  sklearn-KPCA  *******************
sklearn_kpca = KernelPCA(n_components=33, kernel="rbf", gamma=15)
data_kpca = sklearn_kpca.fit_transform(data)
print('data_kpca is: ', data_kpca)
print('the shape of data_kpca is: ', data_kpca.shape)
np.savetxt('data_kpca.out', data_kpca, delimiter=',')



# *******************  sklearn-ISOMAP  *******************
data_ISOMAP = manifold.Isomap(n_neighbors=5, n_components=27, n_jobs=-1).fit_transform(data)
print('data_ISOMAP is: ', data_ISOMAP)
print('the shape of data_ISOMAP is: ', data_ISOMAP.shape)
np.savetxt('data_ISOMAP', data_ISOMAP, delimiter=',')

txt_ISOMAP = []
f = open('data_ISOMAP', encoding='gbk')
for line in f:
    txt_ISOMAP.append(line.strip('\n'))
for i in range(500):
    txt_ISOMAP[i] = '[' + txt_ISOMAP[i] + '],'
txt_ISOMAP = np.array(txt_ISOMAP)
txt_ISOMAP = txt_ISOMAP.reshape(500, )


def save(filename, contents):
    fh = open(filename, 'w')
    # fh.write("\r\n".join(filename))
    fh.write(str(contents).replace('\'', ''))
    fh.close()


save('txt_ISOMAP', txt_ISOMAP)


# *******************  sklearn-LLE  *******************
# method{‘standard’, ‘hessian’, ‘modified’, ‘ltsa’}, default=’standard’
data_LLE, err = manifold.locally_linear_embedding(data, n_neighbors=40, n_components=27)
print('data_HLLE is: ', data_LLE)
# print('the shape of data_HLLE is: ', np.shape(data_LLE))
print("Done. Reconstruction error: %g" % err)


# *******************  sklearn-NMF  *******************
model = NMF(n_components=27, init='random', random_state=0)
# data_NMF = model.fit_transform(data)
# H = model.components_
# print('the shape of data_NMF is: ', data_NMF.shape)
# print('the shape of H is: ', H)


# *******************  umap-learn-umap  *******************
data_UMAP = umap.UMAP(n_neighbors=2, min_dist=0.3, metric='correlation').fit_transform(data)

# print('data_UMAP is: ', data_UMAP)
print('the shape of data_UMAP is: ', data_UMAP.shape)




