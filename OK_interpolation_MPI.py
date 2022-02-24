from numpy.core.shape_base import block
import blurhelper
import numpy as np
import math
import time
import numpy as np
import RFRHelper
from multiprocessing import Pool
import kriginghelper
import os                              
import txtToVTK
import mpi4py.MPI as MPI
 
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

def main(size_grid,samplepoints,xyz_points,neighbour):
    
    start =time.clock()
    result_grids=[]
     
     
    res = []
    for grid in size_grid:
        if grid not in xyz_points:
            result_grids.append(kriginghelper.OKkriging01(grid,neighbour,samplepoints)) 
            # if len(result_grids)>5:
            #     break   
        else:
            for samplepoint in samplepoints:
                if samplepoint[0]==grid[0] and samplepoint[1]==grid[1] and samplepoint[2]==grid[2]:
                    result_grids.append(samplepoint)
                    break
        
                          
  
    end=time.clock()
    print('Running time: %s Seconds'%(end-start))
    
    return result_grids




def list_split(lines, comm_size):
    num=int(len(lines)/comm_size)
    res=[]
    for idx in range(0,comm_size):
        if idx==comm_size-1:
            res.append(lines[idx*num:len(lines)])
        else:
            res.append(lines[idx*num:idx*num+num])
    return res

if __name__ == '__main__':
    neighbour =15 # 邻域个数
    samplepoints,xyz_points=RFRHelper.readsamplepoints('9well.txt',5,5,1)#读取样本数据
    save_txt_path="OK_Interpolation_result15_9well_linear.txt"
    save_vtk_path="OK_Interpolation_result15_9well_linear.vtk"
    if  comm_rank==0:
        start2 =time.clock()
        size_grid=RFRHelper.new3Dgrid(30)#新建网格30*30*30的
        size_grid_block=list_split(size_grid,comm_size=comm_size)
    else:
        size_grid_block = None
        
    _samplepoints=comm.bcast(samplepoints if comm_rank == 0 else None, root=0) 
    _xyz_points=comm.bcast(xyz_points if comm_rank == 0 else None, root=0) 
    _neighbour=comm.bcast(neighbour if comm_rank == 0 else None, root=0) 
    _size_grid_block= comm.scatter(size_grid_block, root=0)  #把数据分发

    pd=main(_size_grid_block,_samplepoints,_xyz_points,_neighbour)#计算插值结果
    strs="comm_rank over"+str(comm_rank)
    print(strs)
    overflag=comm.gather(pd)
    if overflag!=None:
        print(len(overflag))
        with open(save_txt_path, "w") as f:
            for i in overflag:
                for j in i:
                    f.write(str(j[0])+' '+str(j[1])+' '+str(j[2])+' '+str(j[3])+'\n')
        txtToVTK.TxtToVTK(save_txt_path,save_vtk_path)
        end2=time.clock()
        
       
        print('Running time   all: %s Seconds'%(end2-start2))
    

    
     
    #RFRHelper.save_interpolation_result_txt(result_grids,save_txt_path)
    #txtToVTK.TxtToVTK(save_txt_path,save_vtk_path)
    
