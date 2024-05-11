import vtk
import numpy as np
def txt_to_vtk(txt_path,vtk_path):



    well_por_zone=np.loadtxt(txt_path,dtype=np.float32)
    vtkpoints=vtk.vtkPoints()
    scalar = vtk.vtkFloatArray() #实例化多个scalars
    scalar.SetName('porosity')#给标量命名

    for idx,xyz in enumerate(well_por_zone):

        vtkpoints.InsertNextPoint([xyz[0],xyz[1],xyz[2]])
        scalar.InsertNextTuple1(float(xyz[3]))

    inputPolyData=vtk.vtkPolyData()
    inputPolyData.SetPoints(vtkpoints)
    inputPolyData.GetPointData().AddArray(scalar)#标量添加到
    inputPolyData.GetPointData().SetActiveScalars("porosity")#激活标量
        
    # 三角网
    delaunay =vtk.vtkDelaunay2D()
    delaunay.SetInputData(inputPolyData);#点集放到狄洛尼三角网中
    delaunay.Update()
    # 拉普拉斯光滑 
    smoothFilter =vtk.vtkSmoothPolyDataFilter()
    smoothFilter.SetInputConnection(delaunay.GetOutputPort())
    smoothFilter.SetNumberOfIterations(5)
    smoothFilter.SetRelaxationFactor(0.1)
    smoothFilter.FeatureEdgeSmoothingOff()
    smoothFilter.BoundarySmoothingOn()
    smoothFilter.Update()
    # 拟合曲面 
    normalGenerator = vtk.vtkPolyDataNormals()
    normalGenerator.SetInputConnection(delaunay.GetOutputPort()) 
    normalGenerator.ComputePointNormalsOn() 
    normalGenerator.ComputeCellNormalsOn() 
    normalGenerator.Update() 
    surfaceWriter=vtk.vtkPolyDataWriter()
    surfaceWriter.SetFileName(vtk_path)
    surfaceWriter.SetInputData(delaunay.GetOutput())
    surfaceWriter.Write()



 
if __name__ == '__main__':
    
    # strs="./face_zones/zone_"
    # strs="./well_por_zones/well_por_zone_"
    # for zone in range(1,14):
    #     txt_path,vtk_path=strs+str(zone)+".txt",strs+str(zone)+".vtk"
    #     txt_to_vtk(txt_path,vtk_path)
    #     print(txt_path)
    # zone=2
    # tp='test'
    txt_to_vtk("res_SK_zones2.txt","res_SK_zones2_surface.vtk")
    print()
