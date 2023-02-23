from numpy.core.shape_base import block
import Blurhelper
import numpy as np
import math
import time
import numpy as np
import IDWhelper
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
            result_grids.append(IDWhelper.IDW(grid,neighbour,samplepoints)) 
     
            print("comm_rank:"+str(comm_rank)+"----finish:"+str(round(len(result_grids)/len(size_grid)*100,2))+"%")
  
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
    area="Sulige"
    neighbour =40 # the number of known points near unknownpoint
    samplepoints,xyz_points=IDWhelper.readsamplepoints('S14_8_train.txt',1,1,1)# read knownpoints
    save_txt_path="IDW_Interpolation_result"+str(neighbour)+"_"+area+".txt"
    save_vtk_path="IDW_Interpolation_result"+str(neighbour)+"_"+area+".vtk"
    if  comm_rank==0:
        start2 =time.clock()
        size_grid=IDWhelper.new3Dgrid(30) # new empty 3D grid 30*30*30
        size_grid_block=list_split(size_grid,comm_size=comm_size)
    else:
        size_grid_block = None
        
    _samplepoints=comm.bcast(samplepoints if comm_rank == 0 else None, root=0) 
    _xyz_points=comm.bcast(xyz_points if comm_rank == 0 else None, root=0) 
    _neighbour=comm.bcast(neighbour if comm_rank == 0 else None, root=0) 
    _size_grid_block= comm.scatter(size_grid_block, root=0)  #Distribute data to computing node(process)

    pd=main(_size_grid_block,_samplepoints,_xyz_points,_neighbour)##compute value of unknown points
    strs="comm_rank over"+str(comm_rank)
    print(strs)
    overflag=comm.gather(pd) # gather all result
    if overflag!=None:
        print(len(overflag))
        with open(save_txt_path, "w") as f:
            for i in overflag:
                for j in i:
                    f.write(str(j[0])+' '+str(j[1])+' '+str(j[2])+' '+str(j[3])+'\n')
        Blurhelper.blur(save_txt_path,save_txt_path,1)
        
        txtToVTK.TxtToVTK(save_txt_path,save_vtk_path)
        end2=time.clock()
        
       
        print('Running time   all: %s Seconds'%(end2-start2))
    
        with open("Runing_time.txt", "a") as f:
            strs=save_txt_path+'-----Running time all: %s Seconds'%(end2-start2)+"\n"
            f.write(strs)
    
     
 
    
