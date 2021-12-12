import vtk


def TxtToVTK(txt_path,vtk_path):


    points=vtk.vtkPoints()
    # points.SetNumberOfPoints(8) 
    unstructuredGrid = vtk.vtkUnstructuredGrid()
    hexahedronActiveScalars = vtk.vtkFloatArray()
 
    seq = 0

    fp1 = open(txt_path)
    for line1 in fp1:
        t1=line1.replace(' \n','').split(' ')
        x=int(float(t1[0]))
        y=int(float(t1[1]))
        z=int(float(t1[2]))
        v=float(t1[3])


        hexahedron = vtk.vtkHexahedron()
 
        flag =5
        points.InsertNextPoint(0 + x * flag, 0 + y* flag, 0 + z)
        points.InsertNextPoint(1 * flag + x * flag, 0 + y * flag, 0 + z)
        points.InsertNextPoint(1 * flag + x * flag, 1 * flag + y * flag, 0 + z)
    
        points.InsertNextPoint(0 + x * flag, 1* flag + y * flag, 0 + z)
        points.InsertNextPoint(0 + x * flag, 0 + y * flag, 1 + z)
        points.InsertNextPoint(1 * flag + x * flag, 0 + y * flag, 1 + z)
        points.InsertNextPoint(1 * flag + x * flag, 1 * flag + y * flag, 1 + z)
        points.InsertNextPoint(0 + x * flag, 1 * flag + y * flag, 1 + z)
    
  
  
        for i in range(8):
            hexahedron.GetPointIds().SetId(i, seq)
            seq=seq+1
        unstructuredGrid.InsertNextCell(hexahedron.GetCellType(), hexahedron.GetPointIds()) 
	 
        hexahedronActiveScalars.InsertNextTuple1(v)
        #print("txt_to_vtk"+str(t1))
 
    unstructuredGrid.SetPoints(points)#非结构化网格中插入点





    hexahedronActiveScalars.SetName("property")#给标量数组命名
    unstructuredGrid.GetCellData().SetScalars(hexahedronActiveScalars)

    writer =vtk.vtkUnstructuredGridWriter()

    writer.SetFileName(vtk_path)

    writer.SetInputData(unstructuredGrid)
    writer.Write()


if __name__=='__main__':
    TxtToVTK('rfr_Interpolation_result10_231_unblur.txt','rfr_Interpolation_result10_231_unblur.vtk')
     