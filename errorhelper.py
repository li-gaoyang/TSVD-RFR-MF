
import math
import RFRHelper
import numpy as np
def MAE(real,test):
    mae=0
    for i in range(len(test)):
        mae=abs(real[i]-test[i])+mae
    mae=mae/len(test)
    return abs(mae)

def RMSE(real,test):
    rms=0
    for i in range(len(test)):
        rms=rms+(real[i]-test[i])*(real[i]-test[i])
    rms=rms/len(test)
    return math.sqrt(rms)

test_samplepoints,p2=RFRHelper.readsamplepoints("xyzv231testdata.txt",5,5,1)

rfrpoints=RFRHelper.read_txtpoints("RFR_Interpolation_result_231_blur.txt")
okpoints=RFRHelper.read_txtpoints("OK_Interpolation_result5_231.txt")

rfrpoints_array=np.ones((30,30,30))
okpoints_array=np.ones((30,30,30))

for i in range(30*30*30):
    rfrpoint=rfrpoints[i]
    rfrpoints_array[rfrpoint[0]][rfrpoint[1]][rfrpoint[2]]=rfrpoint[3]
    okpoint=okpoints[i]
    okpoints_array[okpoint[0]][okpoint[1]][okpoint[2]]=okpoint[3]

real=[]
rfr=[]
ok=[]
for test_point in test_samplepoints:
    real.append(test_point[3])
    rfr.append(rfrpoints_array[test_point[0]][test_point[1]][test_point[2]])
    ok.append(okpoints_array[test_point[0]][test_point[1]][test_point[2]])

sss=abs(np.array(real)-np.array(rfr))

sss2=abs(np.array(real)-np.array(ok))
print(sss)
print(sss2)
# print(real)
# print(rfr)
# print(ok)
np.array(ok)

rfr_RMSE=RMSE(real,rfr)
ok_RMSE=RMSE(real,ok)


rfr_mae=MAE(real,rfr)
ok_mae=MAE(real,ok)



print("rfr_RMSE:",str(rfr_RMSE))
print("ok_RMSE:",str(ok_RMSE))
print("rfr_mae:",str(rfr_mae))
print("ok_mae:",str(ok_mae))






