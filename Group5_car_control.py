#!/usr/bin/env python
# _*_ coding: utf-8 _*_
file_name='ReverseParking_JetRacer_SSD300_20230604.pth' # 副檔名通常以.pt或.pth儲存，建議使用.pth
import torch
device=torch.device('cuda') # 'cuda'/'cpu'，import torch
num_classes=2 # 物件類別數+1(背景)
batch_size=1 # 必為1
top_k=1 # 依scores挑出最大前top_k個後代入NMS，參考值=200
NMS_threshold=0.5 # 將同類別且IoU小於等於NMS_threshold的物件視為不同物件。此值愈小邊界框會愈少，參考值=0.5

# 取得網路
from torch import nn
class SSD(nn.Module):
    def __init__(self):
        super(SSD,self).__init__()

        # block_1：Conv1_1~Conv4_3+ReLU
        self.block_1=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1), # [batch_size,64,300,300]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1), # [batch_size,64,300,300]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2), # [batch_size,64,150,150]
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1), # [batch_size,128,150,150]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1), # [batch_size,128,150,150]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2), # [batch_size,128,75,75]
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1), # [batch_size,256,75,75]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1), # [batch_size,256,75,75]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1), # [batch_size,256,75,75]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True), # [batch_size,256,38,38]
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1), # [batch_size,512,38,38]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1), # [batch_size,512,38,38] 
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1), # [batch_size,512,38,38]
            nn.ReLU(inplace=True),
        )
        
        # Layer learns to scale the L2 normalized features from conv4_3
        self.l2norm=L2Norm(512,20) # 512為輸入的特徵圖個數，20為scale
         
        # block_2：Pool4~Conv7+ReLU
        self.block_2=nn.Sequential(
            nn.MaxPool2d(kernel_size=2,stride=2), # [batch_size,512,19,19]
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1), # [batch_size,512,19,19]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1), # [batch_size,512,19,19]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1), # [batch_size,512,19,19]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1), # [batch_size,512,19,19]
            nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3,stride=1,padding=6,dilation=6), # [batch_size,1024,19,19]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=1,stride=1), # [batch_size,1024,19,19]
            nn.ReLU(inplace=True),
        )

        # block_3：Conv8_1~Conv8_2+ReLU
        self.block_3=nn.Sequential(
            nn.Conv2d(in_channels=1024,out_channels=256,kernel_size=1), # [batch_size,256,19,19]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=2,padding=1), # [batch_size,512,10,10]
            nn.ReLU(inplace=True),
        )

        # block_4：Conv9_1~Conv9_2+ReLU
        self.block_4=nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=128,kernel_size=1), # [batch_size,128,10,10]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=2,padding=1), # [batch_size,256,5,5]
            nn.ReLU(inplace=True),
        )

        # block_5：Conv10_1~Conv10_2+ReLU
        self.block_5=nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=128,kernel_size=1), # [batch_size,128,5,5]
            nn.ReLU(inplace=True),                            
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3), # [batch_size,256,3,3]
            nn.ReLU(inplace=True),
        )

        # block_6：Conv11_1~Conv11_2+ReLU
        self.block_6=nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=128,kernel_size=1), # [batch_size,128,3,3]                            
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3), # [batch_size,256,1,1]
            nn.ReLU(inplace=True),
        )

        # loc_1
        self.loc_1=nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=4*8,kernel_size=3,stride=1,padding=1), # [batch_size,32,38,38]
        )
        # conf_1
        self.conf_1=nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=4*num_classes,kernel_size=3,stride=1,padding=1), # [batch_size,(4*num_classes),38,38]
        )
        # loc_2
        self.loc_2=nn.Sequential(
            nn.Conv2d(in_channels=1024,out_channels=6*8,kernel_size=3,stride=1,padding=1), # [batch_size,48,19,19]
        )
        # conf_2
        self.conf_2=nn.Sequential(
            nn.Conv2d(in_channels=1024,out_channels=6*num_classes,kernel_size=3,stride=1,padding=1), # [batch_size,(6*num_classes),19,19]
        ) 
        # loc_3
        self.loc_3=nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=6*8,kernel_size=3,stride=1,padding=1), # [batch_size,48,10,10]
        )
        # conf_3
        self.conf_3=nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=6*num_classes,kernel_size=3,stride=1,padding=1), # [batch_size,(6*num_classes),10,10]
        ) 
        # loc_4
        self.loc_4=nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=6*8,kernel_size=3,stride=1,padding=1), # [batch_size,48,5,5]
        )
        # conf_4
        self.conf_4=nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=6*num_classes,kernel_size=3,stride=1,padding=1), # [batch_size,(6*num_classes),5,5]
        )       
        # loc_5
        self.loc_5=nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=4*8,kernel_size=3,stride=1,padding=1), # [batch_size,32,3,3]
        )
        # conf_5
        self.conf_5=nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=4*num_classes,kernel_size=3,stride=1,padding=1), # [batch_size,(4*num_classes),3,3]
        )   
        # loc_6
        self.loc_6=nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=4*8,kernel_size=3,stride=1,padding=1), # [batch_size,32,1,1]
        )
        # conf_6
        self.conf_6=nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=4*num_classes,kernel_size=3,stride=1,padding=1), # [batch_size,(4*num_classes),1,1]
        )   

    def forward(self,x):
        x=self.block_1(x) # [batch_size,512,38,38] (Conv4_3+ReLU輸出)
        n=self.l2norm(x)
        loc1=self.loc_1(n).permute(0,2,3,1).contiguous().view(batch_size,-1).view(batch_size,-1,8)
        conf1=self.conf_1(n).permute(0,2,3,1).contiguous().view(batch_size,-1).view(batch_size,-1,num_classes)
        x=self.block_2(x) # [batch_size,1024,19,19] (Conv7+ReLU輸出)
        loc2=self.loc_2(x).permute(0,2,3,1).contiguous().view(batch_size,-1).view(batch_size,-1,8)
        conf2=self.conf_2(x).permute(0,2,3,1).contiguous().view(batch_size,-1).view(batch_size,-1,num_classes)
        x=self.block_3(x) # [batch_size,512,10,10] (Conv8_2+ReLU輸出)
        loc3=self.loc_3(x).permute(0,2,3,1).contiguous().view(batch_size,-1).view(batch_size,-1,8)
        conf3=self.conf_3(x).permute(0,2,3,1).contiguous().view(batch_size,-1).view(batch_size,-1,num_classes)
        x=self.block_4(x) # [batch_size,256,5,5] (Conv9_2+ReLU輸出)
        loc4=self.loc_4(x).permute(0,2,3,1).contiguous().view(batch_size,-1).view(batch_size,-1,8)
        conf4=self.conf_4(x).permute(0,2,3,1).contiguous().view(batch_size,-1).view(batch_size,-1,num_classes)
        x=self.block_5(x) # [batch_size,256,3,3] (Conv10_2+ReLU輸出)
        loc5=self.loc_5(x).permute(0,2,3,1).contiguous().view(batch_size,-1).view(batch_size,-1,8)
        conf5=self.conf_5(x).permute(0,2,3,1).contiguous().view(batch_size,-1).view(batch_size,-1,num_classes)
        x=self.block_6(x) # [batch_size,256,1,1] (Conv11_2+ReLU輸出)
        loc6=self.loc_6(x).permute(0,2,3,1).contiguous().view(batch_size,-1).view(batch_size,-1,8)
        conf6=self.conf_6(x).permute(0,2,3,1).contiguous().view(batch_size,-1).view(batch_size,-1,num_classes)
        loc=torch.cat((loc1,loc2,loc3,loc4,loc5,loc6),1) # [batch_size,8732,4]，import torch
        conf=torch.cat((conf1,conf2,conf3,conf4,conf5,conf6),1) # [batch_size,8732,num_classes]，import torch
        return loc,conf

class L2Norm(nn.Module):
    def __init__(self,in_channels,scale):
        super(L2Norm,self).__init__()
        self.in_channels=in_channels
        self.gamma=scale or None
        self.eps=1e-10
        self.weight=nn.Parameter(torch.Tensor(self.in_channels)) # from torch import nn，import torch
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.constant_(self.weight,self.gamma) # from torch import nn 
    def forward(self,x):
        norm=x.pow(2).sum(dim=1,keepdim=True).sqrt()+self.eps
        x=torch.div(x,norm) # import torch
        out=self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)*x
        return out

detector=SSD().to(device)
detector.load_state_dict(torch.load(file_name)) # import torch
detector.eval()

# 建立錨框
feature_scale=[38,19,10,5,3,1] # 預測用的特徵圖尺寸(以像素為單位)
sk=[0.07,0.15,0.33,0.51,0.69,0.87,1.05] # 各預測特徵圖的默認框尺度(相對於輸入影像的比例)，比預測特徵圖的個數多1
aspect_ratio=[[1,2,1/2],[1,2,3,1/2,1/3],[1,2,3,1/2,1/3],[1,2,3,1/2,1/3],[1,2,1/2],[1,2,1/2]] # 各預測特徵圖的縱橫比(須檢查loc、conf的濾波器個數)
abox=[]
import itertools
import math
for i,j in enumerate(feature_scale):
    for m,n in itertools.product(range(j),repeat=2):
        cx=(n+0.5)/j # 等同於cx相對於輸入影像的比例位置(乘以輸入影像尺寸即為cx在輸入影像的像素位置)
        cy=(m+0.5)/j # 等同於cy相對於輸入影像的比例位置(乘以輸入影像尺寸即為cy在輸入影像的像素位置)
        for ar in aspect_ratio[i]:
            abox+=[cx-sk[i]*math.sqrt(ar)/2,cy-sk[i]/math.sqrt(ar)/2,cx+sk[i]*math.sqrt(ar)/2,cy-sk[i]/math.sqrt(ar)/2,cx+sk[i]*math.sqrt(ar)/2,cy+sk[i]/math.sqrt(ar)/2,cx-sk[i]*math.sqrt(ar)/2,cy+sk[i]/math.sqrt(ar)/2] # [x0 y0 x1=x2 y1=y0 x2 y2 x3=x0 y3=y2]
        abox+=[cx-math.sqrt(sk[i]*sk[i+1])/2,cy-math.sqrt(sk[i]*sk[i+1])/2,cx+math.sqrt(sk[i]*sk[i+1])/2,cy-math.sqrt(sk[i]*sk[i+1])/2,cx+math.sqrt(sk[i]*sk[i+1])/2,cy+math.sqrt(sk[i]*sk[i+1])/2,cx-math.sqrt(sk[i]*sk[i+1])/2,cy+math.sqrt(sk[i]*sk[i+1])/2] # [x0 y0 x1=x2 y1=y0 x2 y2 x3=x0 y3=y2]
anchor=torch.Tensor(abox).view(-1,8).to(device) # [8732,8] (所有錨框的[xmin ymin xmax ymax]，皆相對於輸入影像的比例位置，乘以輸入影像尺寸即為在輸入影像的像素位置)，import torch
anchor.clamp_(max=1, min=0) # 限定最大值為1、最小值0
anchor=anchor*300 # 轉換成輸入影像尺寸

import numpy
import skfuzzy.control
def Fuzzy_Controller(V1,V2,V3): # V1：x座標，V2：phi,V3: y
    # 定義linquist variable(delta x、delta phi、theta)的universe of discourse
    universe_delta_x=numpy.arange(-1632,1632,0.1) # delta x
    universe_delta_phi=numpy.arange(-90,90,0.1) # delta phi
    universe_delta_y=numpy.arange(0,3264,0.1) # delta y
    universe_theta=numpy.arange(-30,30,0.1) # theta

    # 定義linquist variable (delta x、delta phi、theta)
    delta_x=skfuzzy.control.Antecedent(universe_delta_x,'delta_x')
    delta_phi=skfuzzy.control.Antecedent(universe_delta_phi,'delta_phi')
    delta_y=skfuzzy.control.Antecedent(universe_delta_y,'delta_y')
    theta=skfuzzy.control.Consequent(universe_theta,'theta')

    # 定義delta x的linquist value及其membership function
    delta_x['A11']=skfuzzy.trapmf(universe_delta_x,[-1632,-1632,-204,-102]) # 縮小6倍
    delta_x['A12']=skfuzzy.trimf(universe_delta_x,[-204,-102,0])
    delta_x['A13']=skfuzzy.trimf(universe_delta_x,[-102,0,102])
    delta_x['A14']=skfuzzy.trimf(universe_delta_x,[0,102,204])
    delta_x['A15']=skfuzzy.trapmf(universe_delta_x,[102,204,1632,1632])
    #delta_x['A16']=skfuzzy.trimf(universe_delta_x,[68,136,204])
    #delta_x['A17']=skfuzzy.trapmf(universe_delta_x,[136,204,1632,1632])

    # 定義delta phi的linquist value及其membership function
    delta_phi['A21']=skfuzzy.trapmf(universe_delta_phi,[-90,-90,-30,-15]) # 90度看起來像30度，故縮小3倍
    delta_phi['A22']=skfuzzy.trimf(universe_delta_phi,[-30,-15,0])
    delta_phi['A23']=skfuzzy.trimf(universe_delta_phi,[-15,0,15])
    delta_phi['A24']=skfuzzy.trimf(universe_delta_phi,[0,15,30])
    delta_phi['A25']=skfuzzy.trapmf(universe_delta_phi,[15,30,90,90])
    #delta_phi['A26']=skfuzzy.trimf(universe_delta_phi,[7.5,15,22.5])
    #delta_phi['A27']=skfuzzy.trapmf(universe_delta_phi,[15,22.5,90,90])


    # 定義delta y的linquist value及其membership function
    delta_y['A35']=skfuzzy.trapmf(universe_delta_y,[0,0,544,1088])
    delta_y['A34']=skfuzzy.trimf(universe_delta_y,[544,1088,1632])
    delta_y['A33']=skfuzzy.trimf(universe_delta_y,[1088,1632,2176])
    delta_y['A32']=skfuzzy.trimf(universe_delta_y,[1632,2176,2720])
    delta_y['A31']=skfuzzy.trapmf(universe_delta_y,[2176,2720,3264,3264])
    #delta_y['A36']=skfuzzy.trimf(universe_delta_y,[68,136,204])
    #delta_y['A37']=skfuzzy.trapmf(universe_delta_y,[136,204,1632,1632])
    
    # 定義theta的linquist value及其membership function
    theta['Y1']=skfuzzy.trimf(universe_theta,[-36,-24,-12])#-30,-20,-10
    theta['Y2']=skfuzzy.trimf(universe_theta,[-24,-12,0])#-20,-10,0
    theta['Y3']=skfuzzy.trimf(universe_theta,[-12,0,12])#-10,0,10
    theta['Y4']=skfuzzy.trimf(universe_theta,[0,12,24])#0,10,20
    theta['Y5']=skfuzzy.trimf(universe_theta,[12,24,36])#10,20.30
    #theta['Y6']=skfuzzy.trimf(universe_theta,[10,20,30])
    #theta['Y7']=skfuzzy.trimf(universe_theta,[20,30,40])



    # 解模糊化方法
    theta.defuzzify_method='centroid' # 重心法

    # rule base
    rope1=(delta_x['A11']&delta_phi['A21'])
    rope2=(delta_x['A12']&delta_phi['A21'])|(delta_x['A11']&delta_phi['A22'])
    rope3=(delta_x['A13']&delta_phi['A21'])|(delta_x['A12']&delta_phi['A22'])|(delta_x['A11']&delta_phi['A24'])
    rope4=(delta_x['A14']&delta_phi['A21'])|(delta_x['A13']&delta_phi['A22'])|(delta_x['A12']&delta_phi['A23'])|(delta_x['A11']&delta_phi['A24'])
    rope5=(delta_x['A15']&delta_phi['A21'])|(delta_x['A14']&delta_phi['A22'])|(delta_x['A13']&delta_phi['A23'])|(delta_x['A12']&delta_phi['A24'])|(delta_x['A11']&delta_phi['A25'])
    rope6=(delta_x['A15']&delta_phi['A22'])|(delta_x['A14']&delta_phi['A23'])|(delta_x['A13']&delta_phi['A24'])|(delta_x['A12']&delta_phi['A25'])
    rope7=(delta_x['A15']&delta_phi['A23'])|(delta_x['A14']&delta_phi['A24'])|(delta_x['A13']&delta_phi['A25'])
    rope8=(delta_x['A15']&delta_phi['A24'])|(delta_x['A14']&delta_phi['A25'])
    rope9=(delta_x['A15']&delta_phi['A25'])
    
    #D_SN(A31)
    A31case5=(delta_y['A31']&(rope1|rope2|rope3))
    A31case4=(delta_y['A31']&rope4)
    A31case3=(delta_y['A31']&rope5)
    A31case2=(delta_y['A31']&rope6)
    A31case1=(delta_y['A31']&(rope7|rope8|rope9))
    #D_N(A32)
    A32case5=(delta_y['A32']&(rope1|rope2|rope3))
    A32case4=(delta_y['A32']&rope4)
    A32case3=(delta_y['A32']&rope5)
    A32case2=(delta_y['A32']&rope6)
    A32case1=(delta_y['A32']&(rope7|rope8|rope9))
    #D_M(A33)
    A33case5=(delta_y['A33']&(rope1|rope2))
    A33case4=(delta_y['A33']&(rope3|rope4))
    A33case3=(delta_y['A32']&rope5)
    A33case2=(delta_y['A33']&(rope6|rope7))
    A33case1=(delta_y['A33']&(rope8|rope9))
    #D_F(A34)
    A34case5=(delta_y['A32']&rope1)
    A34case4=(delta_y['A32']&(rope2|rope3|rope4))
    A34case3=(delta_y['A32']&rope5)
    A34case2=(delta_y['A32']&(rope6|rope7|rope8))
    A34case1=(delta_y['A32']&rope9)
    #D_SF(A35)
    A35case5=(delta_y['A31']&rope1)
    A35case4=(delta_y['A31']&(rope2|rope3))
    A35case3=(delta_y['A31']&(rope4|rope5|rope6))
    A35case2=(delta_y['A31']&(rope7|rope8))
    A35case1=(delta_y['A31']&rope9)
    
    all_5=A31case5|A32case5|A33case5|A34case5|A35case5
    all_4=A31case4|A32case4|A33case4|A34case4|A35case4
    all_3=A31case3|A32case3|A33case3|A34case3|A35case3
    all_2=A31case2|A32case2|A33case2|A34case2|A35case2
    all_1=A31case1|A32case1|A33case1|A34case1|A35case1


    rule1=skfuzzy.control.Rule(antecedent=(all_1),consequent=theta['Y1'],label='Y1')
    rule2=skfuzzy.control.Rule(antecedent=(all_2),consequent=theta['Y2'],label='Y2')
    rule3=skfuzzy.control.Rule(antecedent=(all_3),consequent=theta['Y3'],label='Y3')
    rule4=skfuzzy.control.Rule(antecedent=(all_4),consequent=theta['Y4'],label='Y4')
    rule5=skfuzzy.control.Rule(antecedent=(all_5),consequent=theta['Y5'],label='Y5')
   
    system=skfuzzy.control.ControlSystem(rules=[rule1,rule2,rule3,rule4,rule5])
    sim=skfuzzy.control.ControlSystemSimulation(system)
    sim.input['delta_x']=V1
    sim.input['delta_phi']=V2
    sim.input['delta_y']=V3
    sim.compute()
    return sim.output['theta']

# 接收攝影機串流影像，採用多執行緒的方式，降低緩衝區堆疊圖幀的問題
# 設定攝影機
def gstreamer_pipeline(
    capture_width=3264,
    capture_height=2464,
    display_width=3264,
    display_height=2464,
    framerate=21,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )
import cv2 # 匯入cv2套件
import threading
class capCapture:
    def __init__(self,cap):
        self.Frame=[]
        self.status=False
        self.isstop=False
        self.capture=cv2.VideoCapture(cap) # 攝影機連接
    def start(self):
        threading.Thread(target=self.queryframe,daemon=True,args=()).start() # 把程式放進子執行緒，daemon=True表示該執行緒會隨著主執行緒關閉而關閉
    def stop(self):
        self.isstop=True # 設計停止無限迴圈的關閉
    def getframe(self):
        return self.Frame.copy() # 當有需要影像時，再回傳最新的影像
    def queryframe(self):
        while (not self.isstop):
            self.status,self.Frame=self.capture.read()
        self.capture.release()
csi_camera=capCapture(gstreamer_pipeline(flip_method=0)) # 連接攝影機
csi_camera.start() # 啟動子執行緒
import time
time.sleep(1) # 暫停1秒，確保影像已經充填

from PIL import Image
from torchvision import transforms
transforms=transforms.Compose([transforms.Resize((300,300)),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]) # ToTensor將影像像素歸一化至0~1(直接除以255)，from torchvision import transforms
from jetracer.nvidia_racecar import NvidiaRacecar
car=NvidiaRacecar()
car.throttle_gain=0.5
car.throttle=0.2
#while True:
for iteration in range(0,50,1):
    img_cv=csi_camera.getframe() # 取得最新的影像
    I=Image.fromarray(cv2.cvtColor(img_cv,cv2.COLOR_BGR2RGB)) # from PIL import Image，opencv轉PIL.Image
    img=transforms(I) # [3,300,300]，from torchvision import transforms
    img=img.unsqueeze(0) # [1,3,300,300]
    img=img.to(device)
    
    # 預測結果
    pred_loc,pred_conf=detector(img) # pred_loc：[batch_size,8732,8]，pred_conf：[batch_size,8732,num_classes]
    pred_conf=pred_conf.view(batch_size,pred_conf.shape[1],num_classes).transpose(2,1) # [batch_size,num_classes,8732]
    max_pred_conf,_=torch.max(pred_conf,dim=1) # 每個錨框最大預測置信值，[batch_size,8732]，import torch
    for i in range(batch_size):
        
        # 預測x0、y0、...、x3、y3
        pred_bbox_list=list()
        for j in range(8):
            pred_bbox_list.append(anchor[:,j]+pred_loc[i][:,j])
        pred_bbox=torch.stack((pred_bbox_list),dim=1) # 預測的邊界框，[8732,8]，[x0 y0 x1 y1 x2 y2 x3 y3]，import torch
        for j in range(1,num_classes): # 1~(num_classes-1)
            c_mask=torch.ge(pred_conf[i][j],max_pred_conf[i]) # 每個錨框對第j(1~(num_classes-1))個類別的預測置信值是否在所有類別(包含背景)中為最大，True/False，[8732]，import torch
            scores=pred_conf[i][j][c_mask] # 針對第i個batch的第j個類別，若其預測置信值在所有類別(包含背景)中為最大，則取出其預測置信值，並命其為scores，[如533]
            if scores.size(0)==0:
                car.steering=0
                continue
            l_mask=c_mask.unsqueeze(1).expand_as(pred_bbox)
            boxes=pred_bbox[l_mask].view(-1,8) # 針對第i個batch的第j個物件，找出scores對應的預測邊界框，[如533,8]

            _,idx=scores.sort(0) # idx：scores由小而大排列並取得位置編號，[如533]
            idx=idx[-top_k:] # 挑出scores最大前top_k個的位置編號，[top_k]
            b=idx[-1] # 目前scores中最大值的位置編號
            x0=int(boxes[b,0]/300*img_cv.shape[1])
            y0=int(boxes[b,1]/300*img_cv.shape[0])
            x1=int(boxes[b,2]/300*img_cv.shape[1])
            y1=int(boxes[b,3]/300*img_cv.shape[0])
            x2=int(boxes[b,4]/300*img_cv.shape[1])
            y2=int(boxes[b,5]/300*img_cv.shape[0])
            x3=int(boxes[b,6]/300*img_cv.shape[1])
            y3=int(boxes[b,7]/300*img_cv.shape[0])
            points=numpy.array([[x0,y0],[x1,y1],[x2,y2],[x3,y3]],numpy.int32)
            cv2.polylines(img_cv,pts=[points],isClosed=False,color=(255,0,0),thickness=3) # color：BGR

            # 計算x與phi
            delta_x=(x0+x3)/2-img_cv.shape[1]/2
            point1=numpy.array([x3,y3]) # 定義組成一個角的三個點，其中point2為中點
            point2=numpy.array([x0,y0])
            point3=numpy.array([x0,y0-10])
            vector1=point1-point2
            vector2=point3-point2
            cos_phi=numpy.dot(vector1,vector2)/(((vector1[0]**2+vector1[1]**2)**0.5)*((vector2[0]**2+vector2[1]**2)**0.5))
            phi=numpy.arccos(cos_phi) # 弧度
            delta_phi=math.degrees(phi)-90 # 角度(180度制)
            y=(x3-x0)
            # 利用Fuzzy計算steer角度(-30~30)
            theta=Fuzzy_Controller(delta_x,delta_phi,y)
            #steer=str(int(theta))
            car.steering=theta/30
    img_small=cv2.resize(img_cv,(816,616)) # 改變尺寸
    cv2.imshow('Frame',img_small) # 顯示新圖
    k=cv2.waitKey(1)
car.throttle=0

