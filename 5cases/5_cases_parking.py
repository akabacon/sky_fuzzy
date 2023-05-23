################################################################################
# x：-120~120
# y：0~120
# phi：angle of the car (360度制)，0度~180度，車尾朝正右為0度，車尾朝正左為180度
# theta: steering angle (360度制)，-1~1，正使車順時針轉、負使車逆時針轉
# 控制目標：x=0、phi=90度 
################################################################################

import math
def Car_Kinematics_Model(x,y,phi,theta):
    phi=phi/180*math.pi
    theta=theta/180*math.pi
    x=x+math.cos(phi+theta)+math.sin(phi)*math.sin(theta)
    y=y+math.sin(phi+theta)-math.sin(phi)*math.sin(theta)
    phi=phi-math.asin(2*math.sin(theta)/4) # 4為車長
    phi=phi/math.pi*180
    return x,y,phi

import numpy
import skfuzzy.control
import matplotlib.pyplot as plt
def Fuzzy_Controller(V1,V2,V3): # V1：delta_x，V2：delta_phi ，delta_y
    # 定義linquist variable(delta x、delta phi、theta)的universe of discourse
    universe_delta_x=numpy.arange(-120,120.1,0.1) # delta x #[start, ]stop, [step, ]
    universe_delta_phi=numpy.arange(-90,90.1,0.1) # delta phi
    universe_delta_y=numpy.arange(0,120.1,0.1) # delta y

    universe_theta=numpy.arange(-90,90.1,0.1) # theta

    # 定義linquist variable (delta x、delta phi、theta)
    delta_x=skfuzzy.control.Antecedent(universe_delta_x,'delta_x')
    delta_y=skfuzzy.control.Antecedent(universe_delta_y,'delta_y')
    delta_phi=skfuzzy.control.Antecedent(universe_delta_phi,'delta_phi')
    theta=skfuzzy.control.Consequent(universe_theta,'theta')

    # 定義delta x的linquist value及其membership function
    delta_x['C_SL']=skfuzzy.trapmf(universe_delta_x,[-120,-120,-80,-40])
    delta_x['C_L']=skfuzzy.trimf(universe_delta_x,[-80,-40,0])
    delta_x['C_M']=skfuzzy.trimf(universe_delta_x,[-40,0,40])
    delta_x['C_R']=skfuzzy.trimf(universe_delta_x,[0,40,80])
    delta_x['C_SR']=skfuzzy.trapmf(universe_delta_x,[40,80,120,120])
    
    # 定義delta y的linquist value及其membership function
    delta_y['D_SN']=skfuzzy.trapmf(universe_delta_y,[0,0,20,40]) #0~90
    delta_y['D_N']=skfuzzy.trimf(universe_delta_y,[20,40,60])
    delta_y['D_M']=skfuzzy.trimf(universe_delta_y,[40,60,80])
    delta_y['D_F']=skfuzzy.trimf(universe_delta_y,[60,80,100]) #0~90
    delta_y['D_SF']=skfuzzy.trapmf(universe_delta_y,[80,100,120,120]) #0~90
    
    # 定義delta phi的linquist value及其membership function
    delta_phi['A_SC']=skfuzzy.trapmf(universe_delta_phi,[-90,-90,-60,-30])
    delta_phi['A_C']=skfuzzy.trimf(universe_delta_phi,[-60,-30,0])
    delta_phi['A_M']=skfuzzy.trimf(universe_delta_phi,[-30,0,30])
    delta_phi['A_A']=skfuzzy.trimf(universe_delta_phi,[0,30,60])
    delta_phi['A_SA']=skfuzzy.trapmf(universe_delta_phi,[30,60,90,90])

    # 定義theta的linquist value及其membership function
    theta['T_SA']=skfuzzy.trimf(universe_theta,[-90,-60,-30])
    theta['T_A']=skfuzzy.trimf(universe_theta,[-60,-30,0])
    theta['T_M']=skfuzzy.trimf(universe_theta,[-30,0,30])
    theta['T_C']=skfuzzy.trimf(universe_theta,[0,30,60])
    theta['T_SC']=skfuzzy.trimf(universe_theta,[30,60,90])
    # 解模糊化方法
    theta.defuzzify_method='centroid' # 重心法

    # 繪membership funcion圖
    #delta_x.view()
    #delta_phi.view()
    #delta_y.view()
    #theta.view()
    #plt.show()

    rope1=(delta_x['C_SL']&delta_phi['A_SC'])
    rope2=(delta_x['C_L']&delta_phi['A_SC'])|(delta_x['C_SL']&delta_phi['A_C'])
    rope3=(delta_x['C_M']&delta_phi['A_SC'])|(delta_x['C_L']&delta_phi['A_C'])|(delta_x['C_SL']&delta_phi['A_A'])
    rope4=(delta_x['C_R']&delta_phi['A_SC'])|(delta_x['C_M']&delta_phi['A_C'])|(delta_x['C_L']&delta_phi['A_M'])|(delta_x['C_SL']&delta_phi['A_A'])
    rope5=(delta_x['C_SR']&delta_phi['A_SC'])|(delta_x['C_R']&delta_phi['A_C'])|(delta_x['C_M']&delta_phi['A_M'])|(delta_x['C_L']&delta_phi['A_A'])|(delta_x['C_SL']&delta_phi['A_SA'])
    rope6=(delta_x['C_SR']&delta_phi['A_C'])|(delta_x['C_R']&delta_phi['A_M'])|(delta_x['C_M']&delta_phi['A_A'])|(delta_x['C_L']&delta_phi['A_SA'])
    rope7=(delta_x['C_SR']&delta_phi['A_M'])|(delta_x['C_R']&delta_phi['A_A'])|(delta_x['C_M']&delta_phi['A_SA'])
    rope8=(delta_x['C_SR']&delta_phi['A_A'])|(delta_x['C_R']&delta_phi['A_SA'])
    rope9=(delta_x['C_SR']&delta_phi['A_SA'])
    
    #D_SN(A31)
    A31case5=(delta_y['D_SN']&(rope1|rope2|rope3))
    A31case4=(delta_y['D_SN']&rope4)
    A31case3=(delta_y['D_SN']&rope5)
    A31case2=(delta_y['D_SN']&rope6)
    A31case1=(delta_y['D_SN']&(rope7|rope8|rope9))
    #D_N(A32)
    A32case5=(delta_y['D_N']&(rope1|rope2|rope3))
    A32case4=(delta_y['D_N']&rope4)
    A32case3=(delta_y['D_N']&rope5)
    A32case2=(delta_y['D_N']&rope6)
    A32case1=(delta_y['D_N']&(rope7|rope8|rope9))
    #D_M(A33)
    A33case5=(delta_y['D_M']&(rope1|rope2))
    A33case4=(delta_y['D_M']&(rope3|rope4))
    A33case3=(delta_y['D_N']&rope5)
    A33case2=(delta_y['D_M']&(rope6|rope7))
    A33case1=(delta_y['D_M']&(rope8|rope9))
    #D_F(A34)
    A34case5=(delta_y['D_F']&rope1)
    A34case4=(delta_y['D_F']&(rope2|rope3|rope4))
    A34case3=(delta_y['D_F']&rope5)
    A34case2=(delta_y['D_F']&(rope6|rope7|rope8))
    A34case1=(delta_y['D_F']&rope9)
    #D_SF(A35)
    A35case5=(delta_y['D_SF']&rope1)
    A35case4=(delta_y['D_SF']&(rope2|rope3))
    A35case3=(delta_y['D_SF']&(rope4|rope5|rope6))
    A35case2=(delta_y['D_SF']&(rope7|rope8))
    A35case1=(delta_y['D_SF']&rope9)
    
    all_5=A31case5|A32case5|A33case5|A34case5|A35case5
    all_4=A31case4|A32case4|A33case4|A34case4|A35case4
    all_3=A31case3|A32case3|A33case3|A34case3|A35case3
    all_2=A31case2|A32case2|A33case2|A34case2|A35case2
    all_1=A31case1|A32case1|A33case1|A34case1|A35case1

    # rule base
    rule1=skfuzzy.control.Rule(antecedent=(all_1),consequent=theta['T_SA'],label='Turn to super anticlockwsie')
    rule2=skfuzzy.control.Rule(antecedent=(all_2),consequent=theta['T_A'],label='Turn to Anticlockwise')
    rule3=skfuzzy.control.Rule(antecedent=(all_3),consequent=theta['T_M'],label='Turn to middle')
    rule4=skfuzzy.control.Rule(antecedent=(all_4),consequent=theta['T_C'],label='Turn to clockwise')
    rule5=skfuzzy.control.Rule(antecedent=(all_5),consequent=theta['T_SC'],label='Turn to super clockwise')
    system=skfuzzy.control.ControlSystem(rules=[rule1,rule2,rule3])
    sim=skfuzzy.control.ControlSystemSimulation(system)
    sim.input['delta_x']=V1
    sim.input['delta_phi']=V2
    sim.input['delta_y']=V3

    sim.compute()
    # theta.view(sim=sim)
    # plt.show()
    return sim.output['theta']

x=-40 # x初始位置
y=50 # y初始位置
phi=90 # phi初始位置

step=40 #步數
x_pos=[0]*step
y_pos=[0]*step
for t in range(0,step,1):
    theta=Fuzzy_Controller(x-0,phi-90,y-0) # park target(0,0)
    x,y,phi=Car_Kinematics_Model(x,y,phi,theta)
    x_pos[t]=x
    y_pos[t]=y

plt.plot(x_pos,y_pos)
plt.show()


