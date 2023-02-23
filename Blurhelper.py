import numpy as np
import vtk
import os
from numpy.core.fromnumeric import size
import math


import random
###########4.预设回归方法##########
####随机森林回归####



#写入txt
def save_interpolation_result_txt(result_grids, save_txt_path_unblur):
    with open(save_txt_path_unblur, "w") as f:
        for i in range(30):
            for j in range(30):
                for k in range(30):
                    f.write(str(i)+' '+str(j)+' '+str(k)+' ' +
                            str(result_grids[i][j][k])+'\n')


           # 返回一个文件对象   

def means_filter(input_image, filter_size,constant_values):
    '''
    均值滤波器
    :param input_image: 输入图像
    :param filter_size: 滤波器大小
    :return: 输出图像

     '''
    input_image_cp = np.copy(input_image)  # 输入图像的副本

    filter_template = np.ones((filter_size, filter_size,filter_size))  # 空间滤波器模板

    pad_num = int((filter_size - 1) / 2)  # 输入图像需要填充的尺寸

    input_image_cp = np.pad(input_image_cp, (pad_num,pad_num), mode="constant", constant_values=constant_values)  # 填充输入图像
    
    m, n,kk = input_image_cp.shape  # 获取填充后的输入图像的大小
    for x in range(m):
        for y in range(n):
            for z in range(kk):
             
                    #八个面
                if x!=0 and y!=0 and z==31:#前面
                    input_image_cp[x][y][z]= input_image_cp[x][y][z-1]
                if x==0 and y!=0 and z!=0:#左面
                    input_image_cp[x][y][z]= input_image_cp[x+1][y][z]
                if x!=0 and y==0 and z!=0:#下面
                    input_image_cp[x][y][z]= input_image_cp[x][y+1][z]
                if x!=0 and y==31 and z!=0:#上面
                    input_image_cp[x][y][z]= input_image_cp[x][y-1][z]
                if x!=0  and y!=0 and z==0:#后面
                    input_image_cp[x][y][z]= input_image_cp[x][y][z+1]
                if x==31 and y!=0 and z!=0:#右面
                    input_image_cp[x][y][z]= input_image_cp[x-1][y][z]

                    #十二个棱
                if x!=0 and y==0 and z==31:#前下棱
                    input_image_cp[x][y][z]= input_image_cp[x][1][z-1]
                if x==0 and y!=0 and z==31:#前左棱
                    input_image_cp[x][y][z]= input_image_cp[1][y][z-1]
                if x==0 and y==31 and z!=0:#前上棱
                    input_image_cp[x][y][z]= input_image_cp[1][y-1][z]
                if x==31 and y!=0 and z==31:#前右棱
                    input_image_cp[x][y][z]= input_image_cp[x-1][y][z-1]

                if x!=0 and y==0 and z==0:#后下棱
                    input_image_cp[x][y][z]= input_image_cp[x][1][1]
                if x==0 and y!=0 and z==0:#后左棱
                    input_image_cp[x][y][z]= input_image_cp[1][y][1]
                if x!=0 and y==31 and z==0:#后上棱
                    input_image_cp[x][y][z]= input_image_cp[x][y-1][1]
                if x==31 and y!=0 and z==0:#后右棱
                    input_image_cp[x][y][z]= input_image_cp[x-1][y][1]

                if x==0 and y==31 and z!=0:#左上棱
                    input_image_cp[x][y][z]= input_image_cp[1][y-1][z]
                if x==0 and y==0 and z!=0:#左下棱
                    input_image_cp[x][y][z]= input_image_cp[1][1][z]
                if x==31 and y==31 and z!=0:#右上棱
                    input_image_cp[x][y][z]= input_image_cp[x-1][y-1][z]
                if x==31 and y==0 and z!=0:#右下棱
                    input_image_cp[x][y][z]= input_image_cp[x-1][1][z]


                    #八个顶点
                if x==0 and y==31 and z==0:#左上前点
                    input_image_cp[x][y][z]= input_image_cp[1][y-1][1]
                if x==0 and y==0 and z==31:#左下前点
                    input_image_cp[x][y][z]= input_image_cp[1][1][z-1]
                if x==31 and y==31 and z==31:#右上前点
                    input_image_cp[x][y][z]= input_image_cp[x-1][y-1][z-1]
                if x==31 and y==0 and z==31:#右下前点
                    input_image_cp[x][y][z]= input_image_cp[x-1][1][z-1]
                if x==0 and y==31 and z==0:#左上后点
                    input_image_cp[x][y][z]= input_image_cp[1][y-1][1]
                if x==31 and y==31 and z==0:#右上后点
                    input_image_cp[x][y][z]= input_image_cp[x-1][y-1][1]
                if x==0 and y==0 and z==0:#左下后点
                    input_image_cp[x][y][z]= input_image_cp[1][1][1]
                if x==31 and y==0 and z==0:#右下后点
                    input_image_cp[x][y][z]= input_image_cp[x-1][1][1]
 
                
                
             


    output_image = np.copy(input_image_cp)  # 输出图像

    # 空间滤波
    for i in range(pad_num, m - pad_num):
        for j in range(pad_num, n - pad_num):
            for k in range(pad_num, n - pad_num):
              
                ttt1=input_image_cp[i - pad_num:i + pad_num+1,j - pad_num:j + pad_num+1 ,k - pad_num:k + pad_num+1]
                output_image[i, j,k] = np.sum( ttt1) / (filter_size ** 3)
                num=output_image[i, j,k]
              
                  
                
                

    output_image = output_image[pad_num:m - pad_num, pad_num:n - pad_num,pad_num:n - pad_num]  # 裁剪

    return output_image
  
def blur(unblur_path,blur_path,times):
    
    f = open(unblur_path)

    temp1 = np.ones((30,30,30))
    

    values = 0
    for line1 in f:
        t1 = line1.replace(' \n', '').split(' ')
        i = int(t1[0])
        j = int(t1[1])
        k = int(t1[2])
        v = float(t1[3])
        values = values+v
        temp1[i][j][k] = v


    f.close()


    temp2 = np.array(temp1)
    constant_values = values/size(temp2)
    blur=[]
    for  i in range(0,times):
        blur = means_filter(temp2, 3, constant_values)
    
 
    
    save_interpolation_result_txt(blur,blur_path)
    print("blur-------over")
    


if __name__=='__main__':
    blur('RFR_Interpolation_result10_231_unblur.txt','RFR_Interpolation_result10_231_blur.txt',1)
     
 
            