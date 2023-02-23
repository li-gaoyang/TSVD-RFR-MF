import Blurhelper
import numpy as np
import vtk
import os
from numpy.core.fromnumeric import size
import math
import random
from sklearn import ensemble
from sklearn.decomposition import TruncatedSVD
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import math



#New empty 3D geological grid
def new3Dgrid(size_grid):
    xyzs = []
    for i in range(size_grid):
        for j in range(size_grid):
            for k in range(size_grid):
                xyzs.append([i, j, k])
    return xyzs


# Read the coordinates and attribute values of the sample point, and enter the step size in the xyz direction
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

#Calculate distance
def distance(x1, y1, z1, x2, y2, z2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2+(z1-z2)**2)
 



#Calculate inclination
def cal_incl(x0,y0,z0,x1,y1,z1):
    dx=x0-x1
    dy=y0-y1
    dz=z0-z1
    t=pow(pow(dx,2)+pow(dy,2),0.5) 
    incl=0
    if t==0:
        incl= 0
    else:
        incl=math.atan(dz/t)
    return incl

    

# Calculate azimuth   
def cal_azim(x1,y1,x2,y2):
    angle = 0.0
    dx = x2 - x1
    dy = y2 - y1
    if x2 == x1:
        angle = math.pi / 2.0
    if y2 == y1 :
        angle = 0.0
    elif y2 < y1 :
        angle = 3.0 * math.pi / 2.0
    elif x2 > x1 and y2 > y1:
        angle = math.atan(dx / dy)
    elif x2 > x1 and y2 < y1 :
        angle = math.pi / 2 + math.atan(-dy / dx)
    elif x2 < x1 and y2 < y1 :
        angle = math.pi + math.atan(dx / dy)
    elif x2 < x1 and y2 > y1 :
        angle = 3.0 * math.pi / 2.0 + math.atan(dy / -dx)
    return (angle * 180 / math.pi)



# make TSVD-RFR  interpolation model 
# "xyz0" is the coordinate of the unknown point
# "xyz_nei" is some known sample points near unknown points
# return is the attribute value of xyz0 and the coordinates of xyz0
# You can find the input format and output format of data in the "__main__" function.
def RandomForestInterpolation3D(xyz0, xyz_nei):
   
    neighbour = len(xyz_nei)

    #  get train dataset as the input of model, the number of variable is n-1     
    _train_xs = []
    for i in range(len(xyz_nei)-1):
        _v=float(xyz_nei[i][3])
        dis=distance(xyz0[0], xyz0[1], xyz0[2], xyz_nei[i][0], xyz_nei[i][1], xyz_nei[i][2])
        azim=cal_azim(xyz0[0], xyz0[1], xyz_nei[i][0], xyz_nei[i][1])
        incl=cal_incl(xyz0[0], xyz0[1], xyz0[2], xyz_nei[i][0], xyz_nei[i][1], xyz_nei[i][2])
        _train_x = [_v, dis, azim,incl]
        _train_xs.append(_train_x)

    tr_x = np.array(_train_xs)
    tsvd = TruncatedSVD(n_components=1)  # tsvd
    X_tsvd = tsvd.fit_transform(tr_x)
    X_tsvd = X_tsvd.reshape(-1)
    t1 = X_tsvd.tolist()

    unknownpoints = np.array(t1)

    # make model
    train_x = []
    train_y = []
    for i in range(len(xyz_nei)):
        _train_xs = []
        _train_ys = []
        for j in range(len(xyz_nei)):
            if i != j:
                _v=xyz_nei[j][3]
                dis=distance(xyz_nei[i][0], xyz_nei[i][1], xyz_nei[i][2], xyz_nei[j][0], xyz_nei[j][1], xyz_nei[j][2])
                azim=cal_azim(xyz_nei[i][0], xyz_nei[i][1],xyz_nei[j][0], xyz_nei[j][1])
                incl=cal_incl(xyz_nei[i][0], xyz_nei[i][1], xyz_nei[i][2], xyz_nei[j][0], xyz_nei[j][1], xyz_nei[j][2])
                _train_x = [_v, dis, azim,incl]
                a = _train_xs.append(_train_x)
        _train_ys = xyz_nei[i][3]
        tr_x = np.array(_train_xs)
        tr_x = tr_x[np.argsort(tr_x[:, 1])] # # this variable sort by distance between unknown and known points 
        tsvd = TruncatedSVD(n_components=1)  # tsvd
        X_tsvd = tsvd.fit_transform(tr_x)
        X_tsvd = X_tsvd.reshape(-1)
        t1 = X_tsvd.tolist()
        train_x.append(t1)
        train_y.append(_train_ys)

    # RandomForestRegressor
    model_RFR = ensemble.RandomForestRegressor()
    # starttime=time.clock()

    train_x = np.array(train_x)
    tsvd01 = np.insert(train_x, len(train_x), values=unknownpoints, axis=0)
    

    # fit model
    model_RFR.fit(tsvd01[0:neighbour, :], train_y)
    # endtime=time.clock()
    #print('Train time: %s Seconds'%(endtime-starttime))

   
    y_test = model_RFR.predict(tsvd01[0:neighbour, :])
  
    t1 = tsvd01[neighbour:neighbour+1, :]
    result = model_RFR.predict(t1) #预测
   
    return [xyz0[0], xyz0[1], xyz0[2], result[0]]

# result write to txt
def save_interpolation_result_txt(result_grids, save_txt_path_unblur):
    with open(save_txt_path_unblur, "w") as f:
        for i in range(30):
            for j in range(30):
                for k in range(30):
                    f.write(str(i)+' '+str(j)+' '+str(k)+' ' +
                            str(result_grids[i][j][k])+'\n')

#Read sample points from txt
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
    res=RandomForestInterpolation3D(xyz0,xyz_nei)
    print(res)
   