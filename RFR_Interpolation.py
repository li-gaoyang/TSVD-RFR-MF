import numpy as np
 
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

 
import scipy.interpolate as si
import threadpool
import time
start=time.clock()
fp1 = open('grid100_int.txt')
fp2 = open('property100.txt')

#65*60*5的三维网格
xyzs=[]
for i in range(30):
    for j in range(30):
        for k in range(30):
            xyzs.append([i,j,k])
        

grid100=[]
for line1 in fp1:
    t1=line1.replace(' \n','').split(' ')
    _x=int(t1[0])
    _y=int(t1[1])
    _z=int(t1[2])
    tt1=[_x,_y,_z]
  
    # print(tt1)
    grid100.append(tt1)
grid1002=grid100.copy()
property100=[]
for line2 in fp2:
    t2=float(line2)
    property100.append(t2)

print("______")

for idx in range(len(grid100)):
    grid100[idx]=[grid100[idx][0],grid100[idx][1],grid100[idx][2],property100[idx]]




#定义坐标轴

fig = plt.figure()

ax1 = plt.axes(projection='3d')


x=[]
y=[]
z=[]

for i in grid100:
    x.append(i[0])
    y.append(i[1])
    z.append(i[2])


# ax1.scatter3D(x,y,z, cmap='Blues') #绘制散点图



# plt.show()

#计算两点之间的距离
def distance(x1, y1,z1, x2, y2,z2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2+(z1-z2)**2)
#计算两点的角度
def calangle(x0,y0,x1,y1):
    deltaX = x1 - x0+0.000001
    deltaY = y1 - y0+0.000001
    tan = deltaX/deltaY
    formula = math.atan(tan)

    formula = formula ** 2
    formula = math.sqrt(formula)

    formula = math.degrees(formula)
    return formula


            
def get_h_z_anggle(xyz0,neighbour):
    dist=[]
    for i in grid100:#遍历所有的样本点，计算未知点到样本点之间的距离
        _dist=distance(i[0],i[1],i[2],xyz0[0],xyz0[1],xyz0[2])
        dist.append(_dist)
    xyz_nei=[]
    while len(xyz_nei)<neighbour:
        for i in range(0,size(dist)):
            if min(dist)==dist[i]:
                xyz_nei.append(grid100[i])
                dist[i]=999999#移除0，即未知点本身到本身的距离
                break
    train_x=[]
    train_y=[]            
    for i in range(len(xyz_nei)):
        _train_xs=[]
        _train_ys=[]
        for j in range(len(xyz_nei)):
            if i!=j:
                _train_x= [xyz_nei[j][3], distance(xyz_nei[i][0],xyz_nei[i][1],xyz_nei[i][2],xyz_nei[j][0],xyz_nei[j][1],xyz_nei[j][2]),calangle(xyz_nei[i][0],xyz_nei[i][1],xyz_nei[j][0],xyz_nei[j][1])]#
                a=_train_xs.append(_train_x)
        _train_ys=xyz_nei[i][3]
        tr_x=np.array(_train_xs)
        pac = PCA(n_components=1)#主成分分析降维s
        X_pca=pac.fit_transform(tr_x)
        X_pca=X_pca.reshape(-1)
        t1=X_pca.tolist()
        train_x.append(t1)
        train_y.append(_train_ys)

    model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=100,max_depth=20) 
    model_RandomForestRegressor.fit(train_x,train_y)
    y_test=model_RandomForestRegressor.predict(train_x) 
    # print("test_y:"+str(y_test))
    # print("train_y:"+str(train_y))
    _train_xs=[]
    for i in range(len(xyz_nei)-1):
        _train_x= [float(xyz_nei[i][3]), distance(xyz0[0],xyz0[1],xyz0[2],xyz_nei[i][0],xyz_nei[i][1],xyz_nei[i][2]),calangle(xyz0[0],xyz0[1],xyz_nei[i][0],xyz_nei[i][1])]#
        _train_xs.append(_train_x)
         
    tr_x=np.array(_train_xs)
    pac = PCA(n_components=1)#主成分分析降维
    X_pca=pac.fit_transform(tr_x)
    X_pca=X_pca.reshape(-1)
    t1=X_pca.tolist()
    result = model_RandomForestRegressor.predict([t1])
    
    return result[0]

neighbour=4#邻域个数
xyzs2=[]
for _xyz in xyzs:
    
    if _xyz not in grid1002:
        xyzs2.append(_xyz)
    else:
        xyzs2.append(-1)

ress=[]
iii=0
for _xyz in xyzs2:
    if _xyz==-1:
        _xyz=grid100[iii]
        _xyz2=[_xyz[0],_xyz[1],_xyz[2],_xyz[3]]
        iii=iii+1
        # print("---------------------",_xyz[0],_xyz[1],_xyz[2],_xyz[3])
        ress.append(_xyz2)
    else:
        res=get_h_z_anggle(_xyz,neighbour)

        _xyz2=[_xyz[0],_xyz[1],_xyz[2],res]
        ress.append(_xyz2)
    print(_xyz[0],_xyz[1],_xyz[2],res)



     

    


with open("res.txt","w") as f:
    for i in ress:
        f.write(str(i[0])+' '+str(i[1])+' '+str(i[2])+' '+str(i[3])+'\n')

# print("-----over------")
import numpy as np
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
import matplotlib.image as mpimg # mpimg 用于读取图片

import scipy.interpolate as si
import threadpool
import time
import pyclbr
f = open("res.txt")               # 返回一个文件对象   

def means_filter(input_image, filter_size,constant_values):
    '''
    均值滤波器
    :param input_image: 输入图像
    :param filter_size: 滤波器大小
    :return: 输出图像

    注：此实现滤波器大小必须为奇数且 >= 3
    '''
    input_image_cp = np.copy(input_image)  # 输入图像的副本

    filter_template = np.ones((filter_size, filter_size,filter_size))  # 空间滤波器模板

    pad_num = int((filter_size - 1) / 2)  # 输入图像需要填充的尺寸

    input_image_cp = np.pad(input_image_cp, (pad_num,pad_num), mode="constant", constant_values=constant_values)  # 填充输入图像
    
    m, n,kk = input_image_cp.shape  # 获取填充后的输入图像的大小
    for x in range(m):
        for y in range(n):
            for z in range(kk):
             
                    #八个面
                if x!=0 and y!=0 and z==31:#前面
                    input_image_cp[x][y][z]= input_image_cp[x][y][z-1]
                if x==0 and y!=0 and z!=0:#左面
                    input_image_cp[x][y][z]= input_image_cp[x+1][y][z]
                if x!=0 and y==0 and z!=0:#下面
                    input_image_cp[x][y][z]= input_image_cp[x][y+1][z]
                if x!=0 and y==31 and z!=0:#上面
                    input_image_cp[x][y][z]= input_image_cp[x][y-1][z]
                if x!=0  and y!=0 and z==0:#后面
                    input_image_cp[x][y][z]= input_image_cp[x][y][z+1]
                if x==31 and y!=0 and z!=0:#右面
                    input_image_cp[x][y][z]= input_image_cp[x-1][y][z]

                    #十二个棱
                if x!=0 and y==0 and z==31:#前下棱
                    input_image_cp[x][y][z]= input_image_cp[x][1][30]
                if x==0 and y!=0 and z==31:#前左棱
                    input_image_cp[x][y][z]= input_image_cp[1][y][30]
                if x==0 and y==31 and z!=0:#前上棱
                    input_image_cp[x][y][z]= input_image_cp[1][30][z]
                if x==31 and y!=0 and z==31:#前右棱
                    input_image_cp[x][y][z]= input_image_cp[30][y][30]

                if x!=0 and y==0 and z==0:#后下棱
                    input_image_cp[x][y][z]= input_image_cp[x][1][1]
                if x==0 and y!=0 and z==0:#后左棱
                    input_image_cp[x][y][z]= input_image_cp[1][y][1]
                if x!=0 and y==31 and z==0:#后上棱
                    input_image_cp[x][y][z]= input_image_cp[x][30][1]
                if x==31 and y!=0 and z==0:#后右棱
                    input_image_cp[x][y][z]= input_image_cp[30][y][1]

                if x==0 and y==31 and z!=0:#左上棱
                    input_image_cp[x][y][z]= input_image_cp[1][30][z]
                if x==0 and y==0 and z!=0:#左下棱
                    input_image_cp[x][y][z]= input_image_cp[1][1][z]
                if x==31 and y==31 and z!=0:#右上棱
                    input_image_cp[x][y][z]= input_image_cp[30][30][z]
                if x==31 and y==0 and z!=0:#右下棱
                    input_image_cp[x][y][z]= input_image_cp[30][1][z]


                    #八个顶点
                if x==0 and y==31 and z==0:#左上前点
                    input_image_cp[x][y][z]= input_image_cp[1][30][1]
                if x==0 and y==0 and z==31:#左下前点
                    input_image_cp[x][y][z]= input_image_cp[1][1][30]
                if x==31 and y==31 and z==31:#右上前点
                    input_image_cp[x][y][z]= input_image_cp[30][30][30]
                if x==31 and y==0 and z==31:#右下前点
                    input_image_cp[x][y][z]= input_image_cp[30][1][30]
                if x==0 and y==31 and z==0:#左上后点
                    input_image_cp[x][y][z]= input_image_cp[1][30][1]
                if x==31 and y==31 and z==0:#右上后点
                    input_image_cp[x][y][z]= input_image_cp[30][30][1]
                if x==0 and y==0 and z==0:#左下后点
                    input_image_cp[x][y][z]= input_image_cp[1][1][1]
                if x==31 and y==0 and z==0:#右下后点
                    input_image_cp[x][y][z]= input_image_cp[30][1][1]

                
                
             


    output_image = np.copy(input_image_cp)  # 输出图像

    # 空间滤波
    for i in range(pad_num, m - pad_num):
        for j in range(pad_num, n - pad_num):
            for k in range(pad_num, n - pad_num):
              
                ttt1=input_image_cp[i - pad_num:i + pad_num+1,j - pad_num:j + pad_num+1 ,k - pad_num:k + pad_num+1]
                output_image[i, j,k] = np.sum( ttt1) / (filter_size ** 3)
                num=output_image[i, j,k]
              
                  
                
                

    output_image = output_image[pad_num:m - pad_num, pad_num:n - pad_num,pad_num:n - pad_num]  # 裁剪

    return output_image
  
temp1=[]
for i in range(30):
    tempi=[]
    for j in range(30):
        tempj=[]
        for k in range(30):
            tempk=k
            tempj.append(tempk)
        tempi.append(tempj)
    temp1.append(tempi)

values=0
for line1 in f:
    t1=line1.replace(' \n','').split(' ')
    i=int(t1[0])
    j=int(t1[1])
    k=int(t1[2])
    v=float(t1[3])
    values=values+v
    temp1[i][j][k]=v
   
        
    
  
f.close()
 
            
temp2=np.array(temp1)
constant_values=values/size(temp2)
blur=means_filter(temp2,3,constant_values)



ress=[]
for i in range(30):
    for j in range(30):
        for k in range(30):
            _res=[i,j,k,blur[i][j][k]]
            ress.append(_res)




with open("RFR_Interpolation_result.txt","w") as f:
    for i in ress:
        f.write(str(i[0])+' '+str(i[1])+' '+str(i[2])+' '+str(i[3])+'\n')

# cv.imshow("asd",blur)
 
# cv.waitKey(0)

print("-----over------")
            


# end=time.clock()
# import sys
# cost=end-start
# print("%s cost %s second" % (os.path.basename( sys.argv[0]),cost))

