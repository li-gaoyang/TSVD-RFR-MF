import numpy as np
from sklearn import svm
import vtk
import os
from numpy.core.fromnumeric import size
import math
import random
from sklearn import ensemble
import cv2 as cv
from sklearn.decomposition import TruncatedSVD
import scipy.interpolate as si
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# new empty 3D GRID
def new3Dgrid(size_grid):
    xyzs = []
    for i in range(size_grid):
        for j in range(size_grid):
            for k in range(size_grid):
                xyzs.append([i, j, k])
    return xyzs


# Read the coordinates and attribute values of the sample points and enter the step in the XYZ direction
def readsamplepoints(samplepoints_path, x_step, y_step, z_step):
    sample_points = []
    xyz_points = []
    fp1 = open(samplepoints_path)
    for line1 in fp1:
        t1 = line1.replace(' \n', '').split(' ')
        _x = int(int(t1[0])/x_step)
        _y = int(int(t1[1])/y_step)
        _z = int(int(t1[2])/z_step)
        _v = float(t1[3])
        xyz_points.append([_x, _y, _z])
        sample_points.append([_x, _y, _z, _v])
    return sample_points, xyz_points
# 计算两点之间的距离


def distance(x1, y1, z1, x2, y2, z2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2+(z1-z2)**2)



def calangle(x0, y0, x1, y1):

    deltaX = x1 - x0+0.000001
    deltaY = y1 - y0+0.000001
    tan = deltaX/deltaY
    formula = math.atan(tan)

    formula = formula ** 2
    formula = math.sqrt(formula)

    formula = math.degrees(formula)
    return formula

# make RFRI  interpolation model 
# "xyz0" is the coordinate of the unknown point
# "xyz_nei" is some known sample points near unknown points
# return is the attribute value of xyz0 and the coordinates of xyz0
# You can find the input format and output format of data in the "__main__" function.
def RFSI(xyz0, xyz_nei):
    #获取已知点样本的数量
    neighbour = len(xyz_nei)

     #  get train dataset as the input of model, the number of variable is n-1    
    _train_xs = []
    for i in range(len(xyz_nei)-1):
        _train_x = [float(xyz_nei[i][3]), distance(xyz0[0], xyz0[1], xyz0[2], xyz_nei[i][0],
                                                   xyz_nei[i][1], xyz_nei[i][2])]
        _train_xs.append(_train_x)

    tr_x = np.array(_train_xs)

    tr_x=[*tr_x.flat]
    unknownpoints = np.array(tr_x)

    # 训练模型
    train_x = []
    train_y = []
    for i in range(len(xyz_nei)):
        _train_xs = []
        _train_ys = []
        for j in range(len(xyz_nei)):
            if i != j:
                _train_x = [xyz_nei[j][3], distance(xyz_nei[i][0], xyz_nei[i][1], xyz_nei[i][2], xyz_nei[j][0], xyz_nei[j][1], xyz_nei[j][2])]
                _train_xs.append(_train_x)
        _train_ys = xyz_nei[i][3]
        tr_x = np.array(_train_xs)
        tr_x = tr_x[np.argsort(tr_x[:, 1])] # # # this variable sort by distance between unknown and known points 
        tr_x=[*tr_x.flat]
        train_x.append(tr_x)
        train_y.append(_train_ys)

    
    model_RFR = ensemble.RandomForestRegressor(n_estimators=int(neighbour*5),max_depth=5)
 

    train_x = np.array(train_x)
    pcr01 = np.insert(train_x, len(train_x), values=unknownpoints, axis=0)
   


    model_RFR.fit(pcr01[0:neighbour, :], train_y)
    # endtime=time.clock()
    #print('Train time: %s Seconds'%(endtime-starttime))

    
    y_test = model_RFR.predict(pcr01[0:neighbour, :])
   

    t1 = pcr01[neighbour:neighbour+1, :]
    result = model_RFR.predict(t1) # predict
    
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
    xyz0=[0, 0, 0]# Coordinates of unknownpoint 
    xyz_nei=[[5, 9, 0, 0.11], [5, 9, 1, 0.11], [5, 9, 2, 0.1], [5, 9, 3, 0.1], [5, 9, 4, 0.1]]# Coordinates and values of five known sample points
    res=RFSI(xyz0,xyz_nei)
    print(res)
   