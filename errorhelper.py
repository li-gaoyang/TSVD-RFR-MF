
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

test_samplepoints,p2=RFRHelper.readsamplepoints("3well_test.txt",5,5,1)

rfrpoints=RFRHelper.read_txtpoints("RFR_Interpolation_result15_9well_blur.txt")
okpoints=RFRHelper.read_txtpoints("OK_Interpolation_result15_9well_linear.txt")

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

# print(rfr)
# print(ok)
rfr_error=abs(np.array(real)-np.array(rfr))

ok_error=abs(np.array(real)-np.array(ok))
# print(rfr_error)
# print(ok_error)

nptxt=[]
for idx,i in enumerate(test_samplepoints):
    i.append(rfr[idx])
    i.append(ok[idx])
    i.append(rfr_error[idx])
    i.append(ok_error[idx])
    nptxt.append(i)
# print(real)
# print(rfr)
# print(ok)
nptxt=np.array(nptxt)
np.savetxt("nptxt.txt",nptxt)

rfr_RMSE=RMSE(real,rfr)
ok_RMSE=RMSE(real,ok)

rfr_mae=MAE(real,rfr)
ok_mae=MAE(real,ok)

print("RFR_RMSE:",str(rfr_RMSE))
print("OK_RMSE:",str(ok_RMSE))
print("RFR_MAE:",str(rfr_mae))
print("OK_MAE:",str(ok_mae))

