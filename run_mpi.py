import os


python_path = "python"

#python_file = "IDW_Interpolation_MPI.py"
#python_file = "OK_Interpolation_MPI.py"
#ppython_file = "RFSI_Interpolation_MPI.py"
ython_file = "RFR_Interpolation_MPI.py" #TSVD-RFR Interpolation
total_process = "-np {}".format(8)  
 
#machinefile = "-machinefile /home/xxxxxx/myhosts"

machinefile = ""

command_str = 'mpiexec {0} {1} {2} {3}'.format(total_process, machinefile, python_path, python_file)
print(command_str)


os.system(command_str) 

