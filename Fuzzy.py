import numpy
# 定義linquist variable(品質、服務、小費)的universe of discourse
universe_quality=numpy.arange(0,11,0.1) # 品質
universe_service=numpy.arange(0,11,0.1) # 服務
universe_tip=numpy.arange(0,26,0.1) # 小費

import skfuzzy.control
# 定義linquist variable (品質、服務、小費)
quality=skfuzzy.control.Antecedent(universe_quality,'quality') # 'quality'顯示於圖形下方
service=skfuzzy.control.Antecedent(universe_service,'service') # 'service'顯示於圖形下方
tip=skfuzzy.control.Consequent(universe_tip,'tip') # 'tip'顯示於圖形下方

# 定義品質的linquist value及其membership function 
quality['L']=skfuzzy.trimf(universe_quality,[0,0,5])
quality['M']=skfuzzy.trimf(universe_quality,[0,5,10])
quality['H']=skfuzzy.trimf(universe_quality,[5,10,10])

# 定義服務的linquist value及其membership function
service['L']=skfuzzy.trimf(universe_service,[0,0,5])
service['M']=skfuzzy.trimf(universe_service,[0,5,10])
service['H']=skfuzzy.trimf(universe_service,[5,10,10])

# 定義小費的linquist value及其membership function
tip['L']=skfuzzy.trimf(universe_tip,[0,0,13])
tip['M']=skfuzzy.trimf(universe_tip,[0,13,25])
tip['H']=skfuzzy.trimf(universe_tip,[13,25,25])

# 解模糊化方法
tip.defuzzify_method='centroid' # 重心法

import matplotlib.pyplot as plt
# 繪membership funcion圖
#quality.view()
#service.view()
#tip.view()
#plt.show()

# rule base
rule1=skfuzzy.control.Rule(antecedent=((quality['L']&service['L'])|(quality['L']&service['M'])|(quality['M']&service['L'])),consequent=tip['L'],label='Low')
rule2=skfuzzy.control.Rule(antecedent=((quality['M']&service['M'])|(quality['L']&service['H'])|(quality['H']&service['L'])),consequent=tip['M'],label='Medium')
rule3=skfuzzy.control.Rule(antecedent=((quality['M']&service['H'])|(quality['H']&service['M'])|(quality['H']&service['H'])),consequent=tip['H'],label='Hight')

system=skfuzzy.control.ControlSystem(rules=[rule1,rule2,rule3])
sim=skfuzzy.control.ControlSystemSimulation(system)
sim.input['quality']=6.5
sim.input['service']=9.8
sim.compute()   
print(sim.output['tip'])
tip.view(sim=sim)
plt.show()

