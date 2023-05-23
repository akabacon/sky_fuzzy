#import numpy as np

num=3
tri_n=3
trap_n=4
fuzzy_in="delta_x"
fuzzy_out=""
np1={'C_L','C_M','C_R'}
xyco1=np.array([[-10,0],[-10,1],[-5,1],[0,0]])
xyco2=np.array([[-5,0],[0,1],[5,0]])
xyco3=np.array([[0,0],[5,1],[10,1],[10,0]])

print(np1.size)
print(fuzzy_in,end="")
if(xyco1.shape[0]==3):
    print("['{}'] ==skfuzzy.trimf(universe_{},[".format(np1[0],fuzzy_in))
    #for i in range ()
    #delta_x['L']=skfuzzy.trimf(universe_delta_x,[-10,-10,-5,0])