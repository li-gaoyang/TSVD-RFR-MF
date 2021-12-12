# TSVD-RFR-MF
xyzv231.txt为插值样本点，

run_mpi.py是MPI启动程序，支持windows单机并行测试，需要修改“python_file=相应的MPI脚本”

OK_interpolation_MPI为OK插值方法，
RFR_Interpolation_MPI为TSVD-RFR-MF插值方法，

所有的数据最终都生成VTK格式的三维数据集，可以用paraview查看

xyzv231testdata.txt为验证测试数据。
errorhelper.py是计算最终插值结果与验证测试数据的MAE和RMSE误差
