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
        # take the transform coordinate, 
        if 'Liaohe' in txt_path:
            x=x+4043  
            y=y+7961
            z=z+1640
        if 'Sulige' in txt_path:
            x=x+622
            y=y+481
            z=z+186

        v=float(t1[3])


        hexahedron = vtk.vtkHexahedron()
        flag =10
        flagz=-1
        if 'Liaohe' in txt_path:
            flag =10
            flagz=-1
        if 'Sulige' in txt_path:
            flag =100
            flagz=-10
   
        points.InsertNextPoint(0 + x * flag, 0 + y* flag, 0 + z*flagz)
        points.InsertNextPoint(1 * flag + x * flag, 0 + y * flag, 0 + z*flagz)
        points.InsertNextPoint(1 * flag + x * flag, 1 * flag + y * flag, 0 + z*flagz)
    
        points.InsertNextPoint(0 + x * flag, 1* flag + y * flag, 0 + z*flagz)
        points.InsertNextPoint(0 + x * flag, 0 + y * flag, 1 + z*flagz)
        points.InsertNextPoint(1 * flag + x * flag, 0 + y * flag, 1 + z*flagz)
        points.InsertNextPoint(1 * flag + x * flag, 1 * flag + y * flag, 1 + z*flagz)
        points.InsertNextPoint(0 + x * flag, 1 * flag + y * flag, 1 + z*flagz)
    
  
  
        for i in range(8):
            hexahedron.GetPointIds().SetId(i, seq)
            seq=seq+1
        unstructuredGrid.InsertNextCell(hexahedron.GetCellType(), hexahedron.GetPointIds()) 
	 
        hexahedronActiveScalars.InsertNextTuple1(v)
        #print("txt_to_vtk"+str(t1))
 
    unstructuredGrid.SetPoints(points)# take value in unstructuredgrid 





    hexahedronActiveScalars.SetName("property")# make scale name
    unstructuredGrid.GetCellData().SetScalars(hexahedronActiveScalars)

    writer =vtk.vtkUnstructuredGridWriter()

    writer.SetFileName(vtk_path)

    writer.SetInputData(unstructuredGrid)
    writer.Write()


if __name__=='__main__':
    TxtToVTK('RFR_Interpolation_result15_Liaohe.txt','RFR_Interpolation_result15_Liaohe.vtk')
     