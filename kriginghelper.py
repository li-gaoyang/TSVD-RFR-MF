import numpy as np
from math import*
from numpy.linalg import *
import math
from pykrige.ok3d import OrdinaryKriging3D
import os
'''theoretical semivariogram (exponential)  理论变异函数(指数)'''
def semivar_exp2(d):#传入距离 h（d）,块金，基台，变程，带入指数模型进行计算

    t=0+ np.abs(1) * (1.0-np.exp(-d/(np.abs(150))))#指数模型
    #t=np.abs(nug) + np.abs(sill) * (1.0-np.exp(-np.square(d/(np.abs(ran)))))#高斯模型
    # t=np.abs(nug) + np.abs(sill) * (1.5*d/(np.abs(ran)-0.5*np.power(d/(np.abs(ran)),3)))
    # for i in range(0,len(d)):
    #     if d[i]>np.abs(ran):
    #         t[i]=np.abs(nug) + np.abs(sill)

    return t

def dis(p1,p2):
    a=pow((pow((p1[0]-p2[0]),2)+pow((p1[1]-p2[1]),2)+pow((p1[2]-p2[2]),2)),0.5)
    return a
def rh(z1,z2):
    r=1/2*pow((z1[3]-z2[3]),2)
    return r

def linefit(x , y):
    N = float(len(x))
    sx,sy,sxx,syy,sxy=0,0,0,0,0
    for i in range(0,int(N)):
        sx  += x[i]
        sy  += y[i]
        sxx += x[i]*x[i]
        syy += y[i]*y[i]
        sxy += x[i]*y[i]
    a = (sy*sx/N -sxy)/( sx*sx/N -sxx)
    b = (sy - a*sx)/N
    r = abs(sy*sx/N-sxy)/math.sqrt((sxx-sx*sx/N)*(syy-sy*sy/N))
    return a,b,r

#线性半变异函数
def proportional2(h,r):
    a,b,r=linefit(h,r)
    if a<0:
        a=0.01 
    print(a)
    return a

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

def OKkriging(h_data):
    h_data=np.array(h_data)
    r=[]#半变异函数值
    pp=[]
    p=[]#半变异函数距离矩阵
    for i in range(len(h_data)):
        pp.append(h_data[i])
    for i in range(len(pp)):
        for j in range(len(pp)):
            p.append(dis(pp[i],pp[j]))
            r.append(rh(pp[i],pp[j]))
    r=np.array(r).reshape(len(h_data),len(h_data))
    r=np.delete(r,len(h_data)-1,axis =0)
    r=np.delete(r,len(h_data)-1,axis =1)

    h=np.array(p).reshape(len(h_data),len(h_data))
    h=np.delete(h,len(h_data)-1,axis =0)
    oh=h[:,len(h_data)-1]
    h=np.delete(h,len(h_data)-1,axis =1)

    hh=np.triu(h,0)
    rr=np.triu(r,0)
    r0=[]
    h0=[]
    for i in range(len(h_data)-1):
        for j in range(len(h_data)-1):
            if i<j:
                ah=h[i][j]
                h0.append(ah)
                ar=rr[i][j]
                r0.append(ar)
          
    k=proportional(h0,r0)
    hnew=h*k
    a2=np.ones((1,len(h_data)-1))
    a1=np.ones((len(h_data)-1,1))
    a1=np.r_[a1,[[0]]]
    hnew=np.r_[hnew,a2]
    hnew=np.c_[hnew,a1]
    # print('半方差联立矩阵：\n',hnew)
    oh=np.array(k*oh)
    oh=np.r_[oh,[1]]
    
    w=[]
    if np.linalg.det(hnew)!=0:
        w=np.dot(inv(hnew),oh)
        # 
        # ('权阵运算结果：\n',w)
        z0,s2=0,0
        for i in range(len(h_data)-1):
            z0=w[i]*h_data[i][3]+z0
            s2=w[i]*oh[i]+s2
        
        s2=s2+w[len(h_data)-1]

        
        # if z0>23 or z0<7:
        #     z0=h_data[0][3]
         
        return z0 
    else:
        print("矩阵不存在逆矩阵")
        z0=h_data[0][3]
        return z0
 
        

    
    
    print('未知点高程值为：\n',z0)
    # print('半变异值为：\n',pow(s2,0.5))


def OKkriging01(xyz,neighbour,samplepoints):
    tes=[]#pykrige.ok3d库插值会出现异常值，克里金自己的bug？还是克里金代码的bug？
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
        #ok3d = OrdinaryKriging3D(data[:, 0], data[:, 1], data[:, 2], data[:, 3], variogram_model="gaussian")
        k3d1, ss3d = ok3d.execute("grid", np.array(xyz[0],dtype=float), np.array(xyz[1],dtype=float), np.array(xyz[2],dtype=float))
        v=k3d1.data[0][0][0]
        if v<0:#kriging方法会出现负值，负值不在饱和度范围内，饱和度要求必须正值,地区不一定符合正太分布，所以出现异常值
            v=np.mean(np.array(tes)) 
        if v>tes.max():
            v=np.mean(np.array(tes)) 
    else:
        v=temp[0]

    #print(str([xyz[0],xyz[1],xyz[2],v]))
    return [xyz[0],xyz[1],xyz[2],v]
     

  




if __name__ == "__main__":
    
    a=[[69.,76.,60,20.82],[59.,64.,60,10.91],[75.,52. ,60,10.38],[86. , 73.,60, 14.6 ],[69. ,67.,60 , 0.  ]]
 
    
    z0=OKkriging(a)
    print(z0)
