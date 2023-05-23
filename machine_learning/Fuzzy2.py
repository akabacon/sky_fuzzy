import numpy
# 定義linquist variable(品質、服務、小費)的universe of discourse
universe_temperature=numpy.arange(-5,35,0.1) # 品質
universe_wind_speed=numpy.arange(0,30,0.1) # 服務
universe_wind_chill=numpy.arange(-40,50,0.1) # 小費

import skfuzzy.control
# 定義linquist variable (品質、服務、小費)
temperature=skfuzzy.control.Antecedent(universe_temperature,'temperature') # 'temperature'顯示於圖形下方
wind_speed=skfuzzy.control.Antecedent(universe_wind_speed,'wind_speed') # 'wind_speed'顯示於圖形下方
wind_chill=skfuzzy.control.Consequent(universe_wind_chill,'wind_chill') # 'wind_chill'顯示於圖形下方

# 定義品質的linquist value及其membership function 
temperature['COLD']=skfuzzy.trapmf(universe_temperature,[-5,-5,0,10])
temperature['COOL']=skfuzzy.trimf(universe_temperature,[0,10,20])
temperature['WARM']=skfuzzy.trimf(universe_temperature,[10,20,30])
temperature['HOT']=skfuzzy.trapmf(universe_temperature,[20,30,35,35])

# 定義服務的linquist value及其membership function
wind_speed['L']=skfuzzy.trapmf(universe_wind_speed,[0,0,5,17.5])
wind_speed['G']=skfuzzy.trimf(universe_wind_speed,[2.5,15,27.5])
wind_speed['H']=skfuzzy.trapmf(universe_wind_speed,[12.5,25,30,30])

# 定義小費的linquist value及其membership function
wind_chill['SEVERE']=skfuzzy.trimf(universe_wind_chill,[-40,-30,-10])
wind_chill['BAD']=skfuzzy.trimf(universe_wind_chill,[-25,-10,5])
wind_chill['BEARABLE']=skfuzzy.trimf(universe_wind_chill,[-10,5,20])
wind_chill['MILD']=skfuzzy.trimf(universe_wind_chill,[5,20,35])
wind_chill['UNNOTICEABLE']=skfuzzy.trimf(universe_wind_chill,[20,35,50])

# 解模糊化方法
wind_chill.defuzzify_method='centroid' # 重心法

import matplotlib.pyplot as plt
# 繪membership funcion圖
#temperature.view()
#wind_speed.view()
#wind_chill.view()
#plt.show()

# rule base
rule1=skfuzzy.control.Rule(antecedent=((temperature['COLD']&wind_speed['H'])),consequent=wind_chill['SEVERE'],label='SEVERE')
rule2=skfuzzy.control.Rule(antecedent=((temperature['COLD']&wind_speed['G'])|(temperature['COOL']&wind_speed['H'])),consequent=wind_chill['BAD'],label='BAD')
rule3=skfuzzy.control.Rule(antecedent=((temperature['COLD']&wind_speed['L'])|(temperature['COOL']&wind_speed['G'])|(temperature['WARM']&wind_speed['H'])),consequent=wind_chill['BEARABLE'],label='BEARABLE')
rule4=skfuzzy.control.Rule(antecedent=((temperature['COOL']&wind_speed['L'])|(temperature['WARM']&wind_speed['G'])|(temperature['HOT']&wind_speed['H'])),consequent=wind_chill['MILD'],label='MILD')
rule5=skfuzzy.control.Rule(antecedent=((temperature['WARM']&wind_speed['L'])|(temperature['HOT']&wind_speed['L'])|(temperature['WARM']&wind_speed['G'])),consequent=wind_chill['UNNOTICEABLE'],label='UNNOTICEABLE')


system=skfuzzy.control.ControlSystem(rules=[rule1,rule2,rule3,rule4,rule5])
sim=skfuzzy.control.ControlSystemSimulation(system)
sim.input['temperature']=23
sim.input['wind_speed']=6
sim.compute()   
print(sim.output['wind_chill'])
wind_chill.view(sim=sim)
plt.show()




