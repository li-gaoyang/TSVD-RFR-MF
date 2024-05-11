from concurrent import futures
import time
import random
from concurrent.futures import ProcessPoolExecutor

import os
import pandas as pd
from decimal import Decimal
import numpy as np
from sklearn.linear_model import LinearRegression
import math
import numpy as np
from math import *
from numpy.linalg import *
import math
import os
# import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time
from RFRHelper import RandomForestInterpolation3D
from concurrent.futures import ProcessPoolExecutor, wait, as_completed
import time
from datetime import datetime
from tqdm import tqdm


def main(knownpoints, unknowngrids,range=650):
    outrangegrid=[]
    knownpoints2 = knownpoints.copy()
    unknownpoints=[]
    rangeknownpointss=[]
    for idx, unknownpoint in enumerate(unknowngrids):
        # if idx>100:
        #     break
        unknownpointx = unknownpoint[0]
        unknownpointy = unknownpoint[1]
        unknownpointz = unknownpoint[2]
        knownpoints2[:, 0] = knownpoints[:, 0]-unknownpointx
        knownpoints2[:, 1] = knownpoints[:, 1]-unknownpointy
        knownpoints2[:, 2] = knownpoints[:, 2]-unknownpointz
        dis = np.sqrt(knownpoints2[:, 0]**2+knownpoints2[:, 1]**2+knownpoints2[:, 2]**2) #计算距离，
        knownpoints3 = np.insert(knownpoints2, 5, dis, axis=1) #把距离插入最后一列 
        rangeknownpoints = knownpoints[knownpoints3[:, 5] < range, :] #筛选最后一列距离在变程范围内的数据
        #rangeknownpoints = np.delete(rangeknownpoints, 4, 1) #筛选完成后，删除最后一列
        
        if len(rangeknownpoints) == 0:
            res = unknownpoint.tolist()
            res.pop()
            res.append(0) #没有在变程范围内，所以不需要计算，直接给0
            res.append(0) #没有计算，误差为0
            outrangegrid.append(np.array([unknownpointx,unknownpointy,unknownpointz,0,0]))
        elif len(rangeknownpoints)==1:
            # unknownpoints.append(np.array([unknownpointx,unknownpointy,unknownpointz]))#待求的未知点
            # rangeknownpoints=np.insert(rangeknownpoints, 0, values=rangeknownpoints, axis=0)
            # rangeknownpointss.append(rangeknownpoints)
            outrangegrid.append(np.array([unknownpointx,unknownpointy,unknownpointz,rangeknownpoints[0][3],0]))
        else:
            unknownpoints.append(unknownpoint)
            rangeknownpointss.append(rangeknownpoints)# xyz_near_zip变程范围内的临点个数
            # grid_v2.append([xyz, xyz_near])
        # print(len(rangeknownpointss))
    unknownpoints_rangeknownpoints = list(zip(unknownpoints,rangeknownpointss))
    return outrangegrid,unknownpoints_rangeknownpoints #grid_v无效值，邻域点个数为0，xyz_zip要求的位置点, xyz_near_zip未知点附近的n个已知样本点

def call_func_job(args):
    """
    参数解包
    @param args:
    @return:
    """
    unknownpoints=np.array([args[0]])
    rangeknownpoints=np.empty(shape=(args[1].shape[0],4))
    rangeknownpoints[:,0]=args[1][:,0]
    rangeknownpoints[:,1]=args[1][:,1]
    rangeknownpoints[:,2]=args[1][:,2]
    rangeknownpoints[:,3]=args[1][:,3]
    other_factors=args[1][:,4] #face
    # print(unknownpoints)
    res=RandomForestInterpolation3D(unknownpoints,rangeknownpoints,other_factors)
    
    return res


def run(f, this_iter,max_workers=32):
    """
    主函数，运行多进程程序并显示进度条。
    @param f:
    @param this_iter:
    @return:
    """
    process_pool = ProcessPoolExecutor(max_workers)
    results = list(tqdm(process_pool.map(f, this_iter), total=len(this_iter)))
    process_pool.shutdown()
    return results



if __name__ == '__main__':
    zone='2'  # 2_400  # 5_460
    rangesize=584
    knownpoints_path='./train/well_por_zone_'+zone+'_train.txt'
    # knownpoints_path="well_por_zones/well_por_zone_5.txt"
    unknowngrids_path='./face_zones/zone_'+zone+'.txt'
    save_res_path='./result/res_RFR_zones'+zone+'.txt'
    # save_txt_path = "RFR_result_segments_S3D12_face0.txt"  # save res
    knownpoints = np.loadtxt(knownpoints_path, dtype=np.float32)
    unknowngrids = np.loadtxt(unknowngrids_path, dtype=np.float32)
    
    # unknowngrids=unknowngrids[0:2,:]
    # knownpoints=knownpoints[:100,:]
 
    #grid_v无效值，邻域点个数为0，xyz_zip要求的位置点, xyz_near_zip未知点附近的n个已知样本点
    outrangegrid,unknownpoints_rangeknownpoints_zipiter=main(knownpoints, unknowngrids,range=rangesize)
    print(len(unknownpoints_rangeknownpoints_zipiter))  
    start2 =time.perf_counter() 
    jobs = run(call_func_job, unknownpoints_rangeknownpoints_zipiter,max_workers=30)

    unknowngrids_res=outrangegrid
 
    for job in jobs:

        unknowngrids_res.extend(job.tolist())
       
        # unknowngrids_res.append(job)
    with open(save_res_path, "w") as f:
        for idx,i in enumerate(unknowngrids_res):
            f.write(str(i[0])+' '+str(i[1])+' ' +str(i[2])+' '+str(i[4])+'\n')  
    end2=time.perf_counter()
    print('Running time all: %s Seconds'%(end2-start2))

   