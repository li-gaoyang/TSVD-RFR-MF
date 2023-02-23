import numpy as np
from math import*
from numpy.linalg import *
import math
import os



def dis(p1,p2):
    a=pow((pow((p1[0]-p2[0]),2)+pow((p1[1]-p2[1]),2)+pow((p1[2]-p2[2]),2)),0.5)
    return a



def ave(h_data):
    ttt=0
    for i in h_data:
        ttt=ttt+i[3]
    return ttt/len(h_data)


def IDW(xyz,neighbour,samplepoints):
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

    ds=0 # Sum of distances from all points to unknown points
    for i in xyz_nei:
        dd=1/dis(xyz,i)
        ds=ds+dd
    
    v=0

    for i in xyz_nei:
        d=1/dis(xyz,i)
        w_i=d/ds #weight
        w_iXz=w_i*i[3]
        v=w_iXz+v
  
    return [xyz[0],xyz[1],xyz[2],v]
     


# new empty 3D grid
def new3Dgrid(size_grid):
    xyzs = []
    for i in range(size_grid):
        for j in range(size_grid):
            for k in range(size_grid):
                xyzs.append([i, j, k])
    return xyzs
  

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



if __name__ == "__main__":
    
    a=[[69.,76.,60,20.82],[59.,64.,60,10.91],[75.,52. ,60,10.38],[86. , 73.,60, 14.6 ],[69. ,67.,60 , 0.  ]]
 
    
