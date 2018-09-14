import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

def draw_plot(data,title):
    fig=plt.figure()
    plt.scatter(data[:,0],data[:,1])
    fig.suptitle(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def pca_whitening(data):
    draw_plot(data,'Raw data')
    #translating data to have zero mean
    data=data-torch.mean(data,0)
    draw_plot(data,'Translated data')
    #SVD on covariance matrix
    cov_data=torch.mm(torch.transpose(data,0,1),data)/data.shape[0]
    U,S,V=torch.svd(cov_data)
    #rotating data on to the eigenvectors
    dataRot=torch.mm(data,U)
    #dividing by square root of eigenvalues to get variance of all dimensions to be 1 and covariance matrix identity
    datawhite=dataRot/torch.sqrt(S)
    cov_white=torch.mm(torch.transpose(datawhite,0,1),datawhite)/datawhite.shape[0]
    print("covariance matrix of whitened data is:",cov_white)
    draw_plot(datawhite,"Whitened data")

data=torch.load("assign0_data.py")
pca_whitening(data)
