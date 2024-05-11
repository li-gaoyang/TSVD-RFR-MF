import numpy as np
import vtk
import os
from numpy.core.fromnumeric import size
import math
import time
import mpi4py.MPI as MPI
import pointsToVTKpoint
import pointsToVTKsurface
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()


def blur(_points_block, _points_all):
    points_blur = []
    for p1 in _points_block:
        m = []
        for p2 in _points_all:
            if abs(p2[0]-p1[0]) < 12 and abs(p2[1]-p1[1]) < 12:
                m.append(p2[3])
        points_blur.append([p1[0], p1[1], p1[2], sum(m)/len(m)])
        # print('index-------------', len(_points_block) - len(points_blur))
        print("comm_rank:"+str(comm_rank)+"----finish:" +
              str(round(len(points_blur)/len(_points_block)*100, 2))+"%", "-------", len(_points_block)-len(points_blur))
        # if len(points_blur) > 20:
        #     break
    return points_blur


def list_split(lines, comm_size):
    num = int(len(lines)/comm_size)
    res = []
    for idx in range(0, comm_size):
        if idx == comm_size-1:
            res.append(lines[idx*num:len(lines)])
        else:
            res.append(lines[idx*num:idx*num+num])
    return res


if __name__ == '__main__':
    paths="./result/res_IDW_zones2"
    unblur_path = paths+'.txt'
    blur_path = paths+'_blur.txt'
    vtk_path =paths+'_blur_surface.vtk'
    if comm_rank == 0:
        f = open(unblur_path)
        points = []
        for line1 in f:
            t1 = line1.replace('\n', '').split(' ')
            i = float(t1[0])
            j = float(t1[1])
            k = float(t1[2])
            v = float(t1[3])
            points.append([i, j, k, v])
        f.close()
        start2 = time.perf_counter()
        size_grid_block = list_split(points, comm_size=comm_size)
    else:
        size_grid_block = None

    _points_all = comm.bcast(points if comm_rank == 0 else None, root=0)

    _points_block = comm.scatter(size_grid_block, root=0)  # 把数据分发

    pd = blur(_points_block, _points_all)  # 计算插值结果
    strs = "comm_rank over"+str(comm_rank)
    print(strs)
    overflag = comm.gather(pd)
    if overflag != None:
        print(len(overflag))
        with open(blur_path, "w") as f:
            for i in overflag:
                for j in i:
                    f.write(str(j[0])+' '+str(j[1])+' ' +
                            str(j[2])+' '+str(j[3])+'\n')

        end2 = time.perf_counter()
        pointsToVTKsurface.txt_to_vtk(blur_path,vtk_path)
        print('Blur  running time all: %s Seconds' % (end2-start2))
