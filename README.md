# TSVD-RFR-MF
9well.txt为插值样本点，
3well.txt为测试样本
12well.txt为总体样本的井轨迹（点集的形式体现）
welltrace.py能够把txt转换成井轨迹点集

run_mpi.py是MPI启动程序，支持windows单机并行测试，需要修改“python_file=相应的MPI脚本”

OK_interpolation_MPI为OK插值方法，
RFR_Interpolation_MPI为TSVD-RFR-MF插值方法，

所有的数据最终都生成VTK格式的三维数据集，可以用paraview查看

errorhelper.py是计算最终插值结果与验证测试数据的MAE和RMSE误差


核心算法封装在RFRHelper.py文件中。
类似于pykrige包中的OrdinaryKriging3D函数，本文把插值算法封装成RandomForestInterpolation3D函数。

RandomForestInterpolation3D函数中的：xyz0表示未知点的坐标，只支持一个未知点坐标传入。
xyz_nei表示附近的n个样本点（需要包含坐标点，以及坐标点的属性值）.

RFRHelper.py中的__main__函数中是测试样本。本文方法只支持三维空间插值。
