# A Novel 3D Interpolation Based on TSVD and RFR
  
## Description
This is an automatic 3D interpolation project with anisotropy. We designed it for use in 3D geological modeling. We use random forest regression and truncated singular value decomposition to build a three-dimensional interpolation model. The inclination and azimuth in 3D geological modeling are considered in this model. 


##  Requirements
This demo environment:
mpi4py == 3.1.4
numpy == 1.21.6
opencv_python == 4.7.0.68
PyKrige == 1.7.0
scikit_learn == 1.2.1
scipy == 1.7.3
vtk == 9.2.6
python == 3.7.16

## Data
"zone_2.txt" is the unknowngrid.  
"well_por_zone_2_test.txt" is test data.
"well_por_zone_2_train.txt" is interpolation data.
In the dataset, each row represents each point. In each row, the first three numbers represent the coordinates of the point, and the fourth number represents the attribute value of the point. The fifth number is the sedimentary facies on the well.

Each point is the porosity value of 3D geological space. These values are calculated by well logs.


## Testing
Because the interpolation operation of 3D geological body is large, we implemented a test demo with MPI.    So that you can run it in a computing cluster.    Of course, if your computer has an MPI environment, it can also run on a standalone computer.

You can run "run_mpi.py" to run test core and get the results of a 3D geological body.
The "RFR_main_zones.py" is the TSVD-RFR Interpolation test in this paper.
We also compared other interpolation schemes, which you can find in "run_mpi.py".

Because some data are confidential, we have hidden the coordinate information of the well and used the porosity data of a geological horizon.

In addition, we have encapsulated the TSVD-RFR Interpolation core in "RFRHelper.    py".

You can find test samples in the "__ main __" function in "RFRHelper.    py".    The specific use method is similar to the OrderyKriging3D in pykrige(https://github.com/GeoStat-Framework).

We encapsulate the interface as "RandomForestInterpolation3D" function.    So that everyone can use it conveniently.

##  3D visualization
We use VTK to render the calculated results into 3D and save them as ". vtk" files. You can use paraview open source software to view the calculation results.
Paraview(https://www.paraview.org/)
![image](3D_result.png)


##  Documentation

You can get the principle of the algorithm from DOI:
The html documentation is available in: https://


