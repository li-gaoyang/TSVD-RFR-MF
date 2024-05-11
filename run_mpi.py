import os


python_path = r"C:\Anaconda3\envs\py38\python"


python_file = "Blurhelper_mpi.py"  # TSVD-RFR Interpolation
total_process = " -n {}".format(28)

# machinefile = "-machinefile /home/xxxxxx/myhosts"

machinefile = ""

command_str = 'mpiexec {0} {1} {2} {3}'.format(
    total_process, machinefile, python_path, python_file)

print(command_str)


os.system(command_str)




