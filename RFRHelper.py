import blurhelper
import numpy as np
from sklearn import svm
import vtk
import os
from mpl_toolkits.mplot3d import Axes3D
from numpy.core.fromnumeric import size
import pandas as pd
import math

import random
###########4.预设回归方法##########
####随机森林回归####
from sklearn import ensemble
import cv2 as cv
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
import matplotlib.image as mpimg  # mpimg 用于读取图片
from concurrent.futures import ThreadPoolExecutor
import threading
import scipy.interpolate as si
import threadpool
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.svm import LinearSVC
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model
from sklearn.ensemble import AdaBoostRegressor
from sklearn.manifold import MDS


def test_RandomForestRegressor_num(*data):
    '''
    测试 RandomForestRegressor 的预测性能随  n_estimators 参数的影响
    '''
    X_train, y_train = data
    nums = np.arange(1, 100, step=1)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    testing_scores = []
    training_scores = []
    for num in nums:
        regr = ensemble.RandomForestRegressor(n_estimators=num, max_depth=10)
        regr.fit(X_train, y_train)
        training_scores.append(regr.score(X_train, y_train))
    ax.plot(nums, training_scores, label="Training Score")

    ax.set_xlabel("estimator num")
    ax.set_ylabel("score")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1)
    plt.suptitle("RandomForestRegressor")
    plt.show()

# 调用 test_RandomForestRegressor_num
# test_RandomForestRegressor_num(X_train,y_train)


# 新建空白网格
def new3Dgrid(size_grid):
    xyzs = []
    for i in range(size_grid):
        for j in range(size_grid):
            for k in range(size_grid):
                xyzs.append([i, j, k])
    return xyzs


def new3Dgrid_xyzv(size_grid):

    xyzs = []
    for i in range(size_grid):
        for j in range(size_grid):
            for k in range(size_grid):
                xyzs.append([i, j, k])
    return xyzs
# 读取样本点的坐标和属性值，并输入xyz方向上的步长


def readsamplepoints(samplepoints_path, x_step, y_step, z_step):
    sample_points = []
    xyz_points = []
    fp1 = open(samplepoints_path)
    for line1 in fp1:
        t1 = line1.replace(' \n', '').split(' ')
        _x = int(int(t1[0])/x_step)
        _y = int(int(t1[1])/y_step)
        _z = int(int(t1[2])/z_step)
        _v = float(t1[3])*0.01
        xyz_points.append([_x, _y, _z])
        sample_points.append([_x, _y, _z, _v])
    return sample_points, xyz_points
# 计算两点之间的距离


def distance(x1, y1, z1, x2, y2, z2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2+(z1-z2)**2)
# 计算两点的角度


def calangle(x0, y0, x1, y1):

    deltaX = x1 - x0+0.000001
    deltaY = y1 - y0+0.000001
    tan = deltaX/deltaY
    formula = math.atan(tan)

    formula = formula ** 2
    formula = math.sqrt(formula)

    formula = math.degrees(formula)
    return formula

# RFR建模并插值
#xyz0表示未知点的坐标，xyz_nei表示未知点附近的n个已知样本点
#返回的是xyz0的属性值和xyz0的坐标
def RandomForestInterpolation3D(xyz0, xyz_nei):
    #获取已知点样本的数量
    neighbour = len(xyz_nei)

    #获取已知点样本对未知点的影响，随机森林预测的输入集（包含特征n-1个）
    _train_xs = []
    for i in range(len(xyz_nei)-1):
        _train_x = [float(xyz_nei[i][3]), distance(xyz0[0], xyz0[1], xyz0[2], xyz_nei[i][0],
                                                   xyz_nei[i][1], xyz_nei[i][2]), calangle(xyz0[0], xyz0[1], xyz_nei[i][0], xyz_nei[i][1])]
        _train_xs.append(_train_x)

    tr_x = np.array(_train_xs)
    tsvd = TruncatedSVD(n_components=1)  # 降维
    X_tsvd = tsvd.fit_transform(tr_x)
    X_tsvd = X_tsvd.reshape(-1)
    t1 = X_tsvd.tolist()

    unknownpoints = np.array(t1)


    # 训练模型
    train_x = []
    train_y = []
    for i in range(len(xyz_nei)):
        _train_xs = []
        _train_ys = []
        for j in range(len(xyz_nei)):
            if i != j:
                _train_x = [xyz_nei[j][3], distance(xyz_nei[i][0], xyz_nei[i][1], xyz_nei[i][2], xyz_nei[j][0], xyz_nei[j][1], xyz_nei[j][2]), calangle(
                    xyz_nei[i][0], xyz_nei[i][1], xyz_nei[j][0], xyz_nei[j][1])]
                a = _train_xs.append(_train_x)
        _train_ys = xyz_nei[i][3]
        tr_x = np.array(_train_xs)
        tr_x = tr_x[np.argsort(tr_x[:, 1])] #这些特征根据到未知点的距离由远到近排序
        tsvd = TruncatedSVD(n_components=1)  # 降维
        X_pca = tsvd.fit_transform(tr_x)
        X_pca = X_pca.reshape(-1)
        t1 = X_pca.tolist()
        train_x.append(t1)
        train_y.append(_train_ys)

    # LinearRegression,Ridge
    model_RFR = ensemble.RandomForestRegressor(n_estimators=int(neighbour*5))
    # starttime=time.clock()

    train_x = np.array(train_x)
    pcr01 = np.insert(train_x, len(train_x), values=unknownpoints, axis=0)
    pca = PCA(n_components=neighbour-1)  # PCR回归，消除训练集中特征的线性相关关系，把训练集的输入的特征从线性相关转换为线性无关
    pcr01 = pca.fit_transform(pcr01)

    # pcr01[0:neighbour,:]  取前n行，最后一行为训练样本集
    model_RFR.fit(pcr01[0:neighbour, :], train_y)
    # endtime=time.clock()
    #print('Train time: %s Seconds'%(endtime-starttime))

    # test_RandomForestRegressor_num(train_x,train_y)   #看每个模型拟合程度
    # y_test = model_RFR.predict(train_x)
    # print("test_y:"+str(y_test))
    # print("train_y:"+str(train_y))
    #print("得分:"+str(r2_score(y_test, train_y)))
    # for i in y_test:
    #     if i < 0:
    #         print("???????")

    t1 = pcr01[neighbour:neighbour+1, :]
    result = model_RFR.predict(t1) #预测
    #print(y_test, train_y,result[0])
    # print(str([xyz0[0],xyz0[1],xyz0[2],result[0]])+"over")
    # with open("r2_score.txt", "a") as file:  # 只需要将之前的”w"改为“a"即可，代表追加内容#统计得分
    #     file.write(str(r2_score(y_test, train_y))+"\n")
    #print("得分:"+str(r2_score(y_test, train_y))+"---"+str([xyz0[0],xyz0[1],xyz0[2],result[0]]))
    return [xyz0[0], xyz0[1], xyz0[2], result[0]]

#写入txt
def save_interpolation_result_txt(result_grids, save_txt_path_unblur):
    with open(save_txt_path_unblur, "w") as f:
        for i in range(30):
            for j in range(30):
                for k in range(30):
                    f.write(str(i)+' '+str(j)+' '+str(k)+' ' +
                            str(result_grids[i][j][k])+'\n')


def read_txtpoints(file_path):
    res_txt = []
    fp1 = open(file_path)
    for line1 in fp1:
        t1 = line1.replace(' \n', '').split(' ')
        x = int(float(t1[0]))
        y = int(float(t1[1]))
        z = int(float(t1[2]))
        v = float(t1[3])
        res_txt.append([x, y, z, v])
    return res_txt

if __name__ == '__main__':
    xyz0=[0, 0, 0]#未知点的坐标
    xyz_nei=[[5, 9, 0, 0.11], [5, 9, 1, 0.11], [5, 9, 2, 0.1], [5, 9, 3, 0.1], [5, 9, 4, 0.1]]#附近的n个样本点
    res=RandomForestInterpolation3D(xyz0,xyz_nei)
    print(res)
   