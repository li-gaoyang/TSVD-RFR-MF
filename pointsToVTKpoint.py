import vtk

def txt_to_vtk(txt_path,vtk_path):
    fp1 = open(txt_path)
    points = vtk.vtkPoints()
    scalars = vtk.vtkFloatArray()

    for line1 in fp1:
        t1 = line1.replace(' \n', '').split(' ')
        x = float(t1[0])
        y = float(t1[1])
        z = float(t1[2])
        v = float(t1[3])
        # v = float(t1[4])
        # if v==2:
        #     v=1

        points.InsertNextPoint(x, y, z)
        scalars.InsertNextTuple1(v)

    pd = vtk.vtkPolyData()
    pd.SetPoints(points)
    scalars.SetName("porosity")
    pd.GetPointData().AddArray(scalars)
    pd.GetPointData().SetActiveScalars("porosity")

    logWriter = vtk.vtkPolyDataWriter()
    file_name_log = vtk_path
    logWriter.SetFileName(file_name_log)
    logWriter.SetInputData(pd)
    logWriter.Write()


if __name__ == '__main__':
    # strs="./face_zones/zone_"
    # strs="./well_por_zones/well_por_zone_"
    # for zone in range(1,14):
    #     txt_path,vtk_path=strs+str(zone)+".txt",strs+str(zone)+".vtk"
    #     txt_to_vtk(txt_path,vtk_path)
    #     print(txt_path)
    # zone=2
    # tp='test'
    #txt_to_vtk("./test/well_por_zone_2_test.txt","./test/well_por_zone_2_test.vtk")
    txt_to_vtk("well_por_zone_2.txt","well_por_zone_2.vtk")
