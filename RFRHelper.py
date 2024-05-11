 
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
from sklearn.model_selection import GridSearchCV



# Calculate distance
def distance(x1, y1, z1, x2, y2, z2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2+(z1-z2)**2)


# Calculate inclination
def cal_incl(x0, y0, z0, x1, y1, z1):
    dx = x0-x1
    dy = y0-y1
    dz = z0-z1
    t = pow(pow(dx, 2)+pow(dy, 2), 0.5)
    incl = 0
    if t == 0:
        incl = 0
    else:
        incl = math.atan(dz/t)
    return incl


# Calculate azimuth
def cal_azim(x1, y1, x2, y2):
    angle = 0.0
    dx = x2 - x1
    dy = y2 - y1
    if x2 == x1:
        angle = math.pi / 2.0
    if y2 == y1:
        angle = 0.0
    elif y2 < y1:
        angle = 3.0 * math.pi / 2.0
    elif x2 > x1 and y2 > y1:
        angle = math.atan(dx / dy)
    elif x2 > x1 and y2 < y1:
        angle = math.pi / 2 + math.atan(-dy / dx)
    elif x2 < x1 and y2 < y1:
        angle = math.pi + math.atan(dx / dy)
    elif x2 < x1 and y2 > y1:
        angle = 3.0 * math.pi / 2.0 + math.atan(dy / -dx)
    return (angle * 180 / math.pi)


# make TSVD-RFR  interpolation model
# "xyz0" is the coordinate of the unknown point
# "xyz_nei" is some known sample points near unknown points
# return is the attribute value of xyz0 and the coordinates of xyz0
# You can find the input format and output format of data in the "__main__" function.
def RandomForestInterpolation3D(unknownpoints,rangeknownpoints,other_factors,n_jobs=30):

    train_x = []
    train_y = []
    unknownpoint_faice=unknownpoints[0][3]
    #假设未知点，建立模型
    for idx,assumingunknownpoint in enumerate(rangeknownpoints): #遍历每个已知点
      
        x = assumingunknownpoint[0]
        y = assumingunknownpoint[1]
        z = assumingunknownpoint[2]
        v = assumingunknownpoint[3]
        #print(x, y, z)
        #  get train dataset as the input of model, the number of variable is n-1
        _train_xs = []
        for i in range(len(rangeknownpoints)):
            if abs(x-rangeknownpoints[i][0])<0.00001 and abs(y-rangeknownpoints[i][1])<0.00001 and abs(z-rangeknownpoints[i][2])<0.00001:
                aaa=1
                # with open("log2.txt","a") as f:
                #     f.writelines(str(idx)+'==================='+str(np.array(assumingunknownpoint)))
                #     f.writelines("\n")
                # print(idx,assumingunknownpoint,rangeknownpoints[i])
            else:
                _v = float(rangeknownpoints[i][3]) #获取孔隙度值
                _face=float(other_factors[i]) #获取沉积相
                dis = distance(x, y, z, rangeknownpoints[i][0], rangeknownpoints[i][1], rangeknownpoints[i][2]) #获取假设未知点到已知点的距离
                azim = cal_azim(x, y, rangeknownpoints[i][0], rangeknownpoints[i][1])  #获取假设未知点与已知点的方位角
                incl = cal_incl(x, y, z, rangeknownpoints[i][0], rangeknownpoints[i][1], rangeknownpoints[i][2]) #获取假设未知点与已知点的倾斜角
                _train_x = [_v, _face,dis, azim, incl]
                _train_xs.append(_train_x) #假设未知点对应的已知点的特征,特征是根据距离远近排序
                if len(rangeknownpoints)-2==len(_train_xs):
                    break
                
        #print(len(_train_xs))
        tr_x = np.array(_train_xs) #假设未知点对应的已知点的特征
        tsvd = TruncatedSVD(n_components=1)  # tsvd
        X_tsvd = tsvd.fit_transform(tr_x)
        X_tsvd = X_tsvd.reshape(-1)
        t1 = X_tsvd.tolist() 
        unknownpoint = np.array(t1) #假设的未知点特征降维
      
        train_x.append(unknownpoint)
        train_y.append(v)#假设的未知点的值

    model_RFR=ensemble.RandomForestRegressor(n_jobs=1)

    #Finding the optimal parameters
    param_grid = [
        {'n_estimators': [ 10, 100], 'max_features': [10, 100]},
        {'bootstrap': [True], 'n_estimators': [10, 100], 'max_features': [ 10, 100]},
    ] 
    grid_search = ensemble.RandomForestRegressor(n_jobs=1)
    cv_num = 5  #Cross validation parameters
    if len(train_y) < 6:
        cv_num = 2
    model_RFR = GridSearchCV(grid_search, param_grid,cv=cv_num, scoring='neg_mean_squared_error',n_jobs=1)
   
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    # fit model
    model_RFR.fit(train_x, train_y)
 
    # model_score = abs(model_RFR.best_score_)#mode score
    # with open("model_score.txt","a") as f:
    #     f.writelines(str(model_score)+"\n")
        
    test_x=[]
    # 真实未知点
    for unknownpoint in unknownpoints:
        x = unknownpoint[0]
        y = unknownpoint[1]
        z = unknownpoint[2]
        
            #  get train dataset as the input of model, the number of variable is n-1
        _test_xs = []
        for i in range(len(rangeknownpoints)-1): #获取未知点附近的n个已知点
            _v = float(rangeknownpoints[i][3]) #获取孔隙度值
            _face=float(other_factors[i]) #获取沉积相
            dis = distance(x, y, z, rangeknownpoints[i][0], rangeknownpoints[i][1], rangeknownpoints[i][2]) #获取假设未知点到已知点的距离
            azim = cal_azim(x, y, rangeknownpoints[i][0], rangeknownpoints[i][1])  #获取假设未知点与已知点的方位角
            incl = cal_incl(x, y, z, rangeknownpoints[i][0], rangeknownpoints[i][1], rangeknownpoints[i][2]) #获取假设未知点与已知点的倾斜角
            _test_x = [_v, _face,dis, azim, incl]
            _test_xs.append(_test_x) #假设未知点对应的已知点的特征
            if len(rangeknownpoints)-2==len(_test_xs):
                break
        tr_x = np.array(_test_xs) #假设未知点对应的已知点的特征
        tsvd = TruncatedSVD(n_components=1)  # tsvd
        X_tsvd = tsvd.fit_transform(tr_x)
        X_tsvd = X_tsvd.reshape(-1)
        t1 = X_tsvd.tolist() 
        unknownpointfeatures = np.array(t1)
        test_x.append(unknownpointfeatures)

    # RandomForestRegressor
    starttime = time.perf_counter()


    test_x = np.array(test_x)
    test_y=model_RFR.predict(test_x)

    endtime = time.perf_counter()
    
    return  np.insert(unknownpoints, unknownpoints.shape[1], test_y, axis=1)
# result write to txt


# Read sample points from txt



if __name__ == '__main__':
    print()
    # xyz0 = [0, 0, 0]  # Coordinates of unknownpoint
    # xyz_nei = [[5, 9, 0, 0.11,0], [5, 9, 1, 0.11,0], [5, 9, 2, 0.1,1], [5, 9, 3, 0.1,2], [
    #     5, 9, 4, 0.1,3], [5, 10, 6, 0.3,3],[4, 8, 0, 0.13,2]]  # Coordinates and values of five known sample points
    # res = RandomForestInterpolation3D(xyz0, xyz_nei)
    # print(res)
