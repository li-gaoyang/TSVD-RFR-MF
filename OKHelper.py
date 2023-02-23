import numpy as np
from math import*
from numpy.linalg import *
import math
from pykrige.ok3d import OrdinaryKriging3D
import os



# new empty 3d gird
def new3Dgrid(size_grid):
    xyzs = []
    for i in range(size_grid):
        for j in range(size_grid):
            for k in range(size_grid):
                xyzs.append([i, j, k])
    return xyzs


def dis(p1,p2):
    a=pow((pow((p1[0]-p2[0]),2)+pow((p1[1]-p2[1]),2)+pow((p1[2]-p2[2]),2)),0.5)
    return a
def rh(z1,z2):
    r=1/2*pow((z1[3]-z2[3]),2)
    return r



def proportional(h,r):
    xx,xy=0,0

    for i in range(len(h)):
        xx+=pow(h[i],2)
        xy+=h[i]*r[i]
    k=xy/xx
    return k
def ave(h_data):
    ttt=0
    for i in h_data:
        ttt=ttt+i[3]
    return ttt/len(h_data)

def OKkriging01(xyz,neighbour,samplepoints):
    tes=[]
    for i in samplepoints:
        tes.append(i[3])
    tes=np.array(tes)

    dis_k_v={}
    for samplepoint in samplepoints:
        dist=dis(xyz,samplepoint)
        if dist>0:
            xx = {dist:samplepoint}
            dis_k_v.update(xx)
    dis_k_v2=sorted(dis_k_v.keys())
    xyz_nei=[] 
    for value in dis_k_v2:
        xyz_nei.append(dis_k_v[value])
        if len(xyz_nei)>neighbour-1:
            break
    temp=[]
    for i in xyz_nei:
        temp.append(i[3])
    temp2=np.array(temp,dtype=float) 
    data=np.array(xyz_nei,dtype=float)
    if len(np.unique(temp2))!=1:
        ok3d = OrdinaryKriging3D(data[:, 0], data[:, 1], data[:, 2], data[:, 3], variogram_model="power")
     
        k3d1, ss3d = ok3d.execute("grid", np.array(xyz[0],dtype=float), np.array(xyz[1],dtype=float), np.array(xyz[2],dtype=float))
        v=k3d1.data[0][0][0]
        if v<tes.min():# The Kriging method has negative values, which are not within the saturation range. Saturation requires positive values. Geological data does not necessarily conform to positive or too distributed, so outliers occur.

            v=np.mean(np.array(tes)) 
        if v>tes.max():
            v=np.mean(np.array(tes)) 
    else:
        v=temp[0]

    #print(str([xyz[0],xyz[1],xyz[2],v]))
    return [xyz[0],xyz[1],xyz[2],v]
     

  

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





# if __name__ == "__main__":
    
#     a=[[69.,76.,60,20.82],[59.,64.,60,10.91],[75.,52. ,60,10.38],[86. , 73.,60, 14.6 ],[69. ,67.,60 , 0.  ]]
