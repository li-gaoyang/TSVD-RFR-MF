import vtk



fp1 = open("11well.txt")


points=vtk.vtkPoints()
scalars=vtk.vtkFloatArray()

for line1 in fp1:
    t1=line1.replace(' \n','').split(' ')
    x=int(float(t1[0]))
    y=int(float(t1[1]))
    z=int(float(t1[2]))
    v=float(t1[3])/100
    points.InsertNextPoint(x,y,z)
    scalars.InsertNextTuple1(v)

pd=vtk.vtkPolyData()
pd.SetPoints(points)
scalars.SetName("property")
pd.GetPointData().AddArray(scalars)
pd.GetPointData().SetActiveScalars("property")
logWriter=vtk.vtkPolyDataWriter()
file_name_log = "11well.vtk"
logWriter.SetFileName(file_name_log)
logWriter.SetInputData(pd)
logWriter.Write()