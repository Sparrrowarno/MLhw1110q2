import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
import math

def data(N, prior, mu, Sigma, radius, angle):
    dist = [0]
    dist.extend(np.cumsum(prior))
    random0 = np.random.rand(N)
    L = np.zeros((N))
    x = np.zeros((N, 2))
    i = 0
    indices = np.squeeze(np.where(np.logical_and(random0 >= dist[i], random0 < dist[i+1]) == True))#dimentions
    L[indices] = i * np.ones(len(indices))
    for k in indices:
        x[k,:] = np.random.multivariate_normal(mu, Sigma)
    i = 1
    indices = np.squeeze(np.where(np.logical_and(random0 >= dist[i], random0 < dist[i+1]) == True))
    L[indices] = i * np.ones(len(indices))
    for k in indices:
        r = (radius[1] - radius[0]) * np.random.rand() + radius[0]
        ang = (angle[1] - angle[0]) * np.random.rand() + angle[0]
        x[k,0] = r*math.cos(ang)
        x[k,1] = r*math.sin(ang)
    return x, L

#true datas
N = 1000
prior = np.array([0.35,0.65])
mu = np.array([0,0])
Sigma = np.array([[1,0.5],
                  [0.5,1]])
radius = np.array([2,3])
angle = np.array([-math.pi,math.pi])
K = 10

x, L = data(N, prior, mu, Sigma, radius, angle)
plt.figure(figsize = (10,10))
plt.scatter(x[np.where(L == 0),0],x[np.where(L == 0),1], marker='.', label='Class -')
plt.scatter(x[np.where(L == 1),0],x[np.where(L == 1),1], marker='+', label='Class +')
plt.title('Training data',fontsize=20)
plt.legend(fontsize = 20)

x_t, L_t = data(N, prior, mu, Sigma, radius, angle)

C_list = 10 ** (np.linspace(-3, 3, 7))
error_mean = np.zeros((len(C_list)))
fold_mark = np.linspace(0, N, K + 1)
index_partition_limits = np.zeros((K, 2))
for k in range(K):
    index_partition_limits[k, 0] = fold_mark[k]
    index_partition_limits[k, 1] = fold_mark[k + 1] - 1
for i in range(len(C_list)):
    C = C_list[i]
    P_error = np.zeros(K)
    for k in range(K):
        ind_validate = np.arange(index_partition_limits[k, 0], index_partition_limits[k, 1] + 1)
        ind_validate = ind_validate.astype(int)
        x_validate = x[ind_validate, :]
        L_validate = L[ind_validate]
        ind_train = np.hstack(
            (np.arange(0, index_partition_limits[k, 0]), np.arange(index_partition_limits[k, 1] + 1, N)))
        ind_train = ind_train.astype(int)
        x_train = x[ind_train, :]
        L_train = L[ind_train]
        SVM = svm.SVC(C=C, kernel='linear').fit(x_train, L_train)
        # print(SVM)
        D_validate = SVM.predict(x_validate)
        P_error[k] = np.count_nonzero(L_validate != D_validate) / len(L_validate)
    error_mean[i] = np.mean(P_error)
min_error_mean = min(error_mean)
for i in range(len(C_list)):
    if error_mean[i] == min_error_mean:
        opt_C1 = C_list[i]
        break

print(opt_C1)

plt.figure(figsize = (10,10))
plt.semilogx(C_list, error_mean, marker='x')
plt.title('Linear kernel opt',fontsize=20)
plt.xlabel('C', fontsize=20)
plt.ylabel('Probability of error', fontsize=20)
print('optimized C is %f'%opt_C1)
print('smallest probability of error is %f'%min_error_mean)

C_list = 10 ** (np.linspace(-4, 4, 9))
Sigma_list = 10 ** (np.linspace(-2, 2, 5))
error_mean = np.zeros((len(Sigma_list), len(C_list)))
fold_mark = np.linspace(0, N, K + 1)
index_partition_limits = np.zeros((K, 2))
for k in range(K):
    index_partition_limits[k, 0] = fold_mark[k]
    index_partition_limits[k, 1] = fold_mark[k + 1] - 1
for j in range(len(Sigma_list)):
    Sigma = Sigma_list[j]
    for i in range(len(C_list)):
        C = C_list[i]
        P_error = np.zeros((K,))
        for k in range(K):
            ind_validate = np.arange(index_partition_limits[k, 0], index_partition_limits[k, 1] + 1)
            ind_validate = ind_validate.astype(int)
            x_validate = x[ind_validate, :]
            L_validate = L[ind_validate]
            ind_train = np.hstack(
                (np.arange(0, index_partition_limits[k, 0]), np.arange(index_partition_limits[k, 1] + 1, N)))
            ind_train = ind_train.astype(int)
            x_train = x[ind_train, :]
            L_train = L[ind_train]
            SVM = svm.SVC(C=C, kernel='rbf', gamma=Sigma).fit(x_train, L_train)
            D_validate = SVM.predict(x_validate)
            P_error[k] = np.count_nonzero(L_validate != D_validate) / len(L_validate)
        error_mean[j, i] = np.mean(P_error)
min_error_mean = np.min(error_mean)
for j in range(len(Sigma_list)):
    for i in range(len(C_list)):
        if error_mean[j, i] == min_error_mean:
            opt_C2 = C_list[i]
            opt_Sigma2 = Sigma_list[j]
            print(error_mean[j, i])
            print(j, i)
            break

print(opt_C2)
print(opt_Sigma2)
print(C_list)

print(error_mean)
print(min_error_mean)

plt.figure(figsize = (10,10))
plt.xscale('log')
plt.yscale('log')
a = plt.contour(C_list,Sigma_list,error_mean)
a.clabel(fontsize=10)
plt.xlabel('C',fontsize=20)
plt.ylabel('Sigma',fontsize=20)
plt.title('Gaussian kernel opt',fontsize=20)
print('optimized C is %f'%opt_C2)
print('optimized Sigma is %f'%opt_Sigma2)
print('smallest probability of error is %f'%min_error_mean)

opt_SVM = svm.SVC(C=opt_C1, kernel='linear').fit(x, L)
D = opt_SVM.predict(x)
count1 = np.count_nonzero(L != D)
P_error_linear = np.count_nonzero(L != D) / len(D)
plt.figure(figsize = (10,10))
plt.scatter(x[np.where(np.logical_and(D == 0, L == 0)),0],x[np.where(np.logical_and(D == 0, L == 0)),1], marker='o', c='g',label='D = -, L = -')
plt.scatter(x[np.where(np.logical_and(D == 0, L == 1)),0],x[np.where(np.logical_and(D == 0, L== 1)),1], marker='o', c='r',label='D = -, L = +')
plt.scatter(x[np.where(np.logical_and(D == 1, L == 0)),0],x[np.where(np.logical_and(D == 1, L == 0)),1], marker='x', c='r',label='D = +, L = -')
plt.scatter(x[np.where(np.logical_and(D == 1, L == 1)),0],x[np.where(np.logical_and(D == 1, L == 1)),1], marker='x', c='g',label='D = +, L = +')
plt.legend(fontsize = 20)
plt.title('SVM (linear kernel) classification on training data',fontsize=20)
print('errors: %f'%count1)
print('probability of error: %f'%P_error_linear)

opt_SVM = svm.SVC(C=opt_C2, kernel='rbf',gamma=opt_Sigma2).fit(x, L)
D = opt_SVM.predict(x)
count2 = np.count_nonzero(L != D)
P_error_gaussian = np.count_nonzero(L != D) / len(D)
plt.figure(figsize = (10,10))
plt.scatter(x[np.where(np.logical_and(D == 0, L == 0)),0],x[np.where(np.logical_and(D == 0, L == 0)),1], marker='.', c='g', label='D = -, L = -')
plt.scatter(x[np.where(np.logical_and(D == 0, L == 1)),0],x[np.where(np.logical_and(D == 0, L== 1)),1], marker='.', c='r', label='D = -, L = +')
plt.scatter(x[np.where(np.logical_and(D == 1, L == 0)),0],x[np.where(np.logical_and(D == 1, L == 0)),1], marker='x', c='r', label='D = +, L = -')
plt.scatter(x[np.where(np.logical_and(D == 1, L == 1)),0],x[np.where(np.logical_and(D == 1, L == 1)),1], marker='x', c='g', label='D = +, L = +')
plt.legend(fontsize = 20)
plt.title('SVM (Gaussian kernel) classification on training data',fontsize=20)
print('errors: %f'%count2)
print('probability of error: %f'%P_error_gaussian)

opt_SVM = svm.SVC(C=opt_C1, kernel='linear').fit(x, L)
D = opt_SVM.predict(x_t)
count1 = np.count_nonzero(L_t != D)
P_error_linear = np.count_nonzero(L_t != D) / len(D)
plt.figure(figsize = (10,10))
plt.scatter(x[np.where(np.logical_and(D == 0, L_t == 0)),0],x[np.where(np.logical_and(D == 0, L_t == 0)),1], marker='.', c='g', label='D = -, L = -')
plt.scatter(x[np.where(np.logical_and(D == 0, L_t == 1)),0],x[np.where(np.logical_and(D == 0, L_t == 1)),1], marker='.', c='r', label='D = -, L = +')
plt.scatter(x[np.where(np.logical_and(D == 1, L_t == 0)),0],x[np.where(np.logical_and(D == 1, L_t == 0)),1], marker='x', c='r', label='D = +, L = -')
plt.scatter(x[np.where(np.logical_and(D == 1, L_t == 1)),0],x[np.where(np.logical_and(D == 1, L_t == 1)),1], marker='x', c='g', label='D = +, L = +')
plt.legend(fontsize = 20)
plt.title('SVM (linear kernel) classification on training data',fontsize=20)
print('errors: %f'%count1)
print('probability of error: %f'%P_error_linear)

opt_SVM = svm.SVC(C=opt_C2, kernel='rbf',gamma=opt_Sigma2).fit(x, L)
D = opt_SVM.predict(x_t)
count2 = np.count_nonzero(L_t != D)
P_error_gaussian = np.count_nonzero(L_t != D) / len(D)
plt.figure(figsize = (10,10))
plt.scatter(x[np.where(np.logical_and(D == 0, L_t == 0)),0],x[np.where(np.logical_and(D == 0, L_t == 0)),1], marker='.', c='g', label='D = -, L = -')
plt.scatter(x[np.where(np.logical_and(D == 0, L_t == 1)),0],x[np.where(np.logical_and(D == 0, L_t== 1)),1], marker='.', c='r', label='D = -, L = +')
plt.scatter(x[np.where(np.logical_and(D == 1, L_t == 0)),0],x[np.where(np.logical_and(D == 1, L_t == 0)),1], marker='x', c='r', label='D = +, L = -')
plt.scatter(x[np.where(np.logical_and(D == 1, L_t == 1)),0],x[np.where(np.logical_and(D == 1, L_t == 1)),1], marker='x', c='g', label='D = +, L = +')
plt.legend(fontsize = 20)
plt.title('SVM (Gaussian kernel) classification on training data',fontsize=20)
print('errors: %f'%count2)
print('probability of error: %f'%P_error_gaussian)