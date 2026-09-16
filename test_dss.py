import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from model import load_model, solve_allocation, best_enablement, compare_strategies, pooling_sensitivity, goal_programming, minimax_pooling, simulate
p=Path(__file__).with_name('Style_Sphere_Model.xlsx')
s,r,c,f,risk=load_model(p)
assert len(s)==5 and len(r)==4 and len(f)==7
assert set(s.Store)=={'S1','S2','S3','S4','S5'}
assert set(r.Region)=={'North','West','South','East'}
base=best_enablement(s,r,c,f,.30,1,1,.96,False,False,capacity_loss=risk['capacity_loss'])
assert base['enabled']==[]
assert abs(base['result']['fill_rate']-1)<1e-9 and base['result']['lost'].sum()==0
high=best_enablement(s,r,c,f,.30,1.4,1.4,.96,False,False,capacity_loss=risk['capacity_loss'])
assert high['result']['total_cost']>base['result']['total_cost']
stress=best_enablement(s,r,c,f,.30,1.4,1.4,.96,True,True,capacity_loss=risk['capacity_loss'])
assert stress['result']['fill_rate']>=.96-1e-9
assert stress['result']['lost'].sum()>0
assert set(stress['enabled']).issubset(set(s.Store))
assert stress['result']['total_cost']>high['result']['total_cost']

# Exhaustive regression over all 32 store-selection combinations.
import itertools
for combo in itertools.chain.from_iterable(itertools.combinations(s.Store, k) for k in range(len(s.Store)+1)):
    z=solve_allocation(s,r,c,f,list(combo),.30,1,1,.96,False,False,capacity_loss=risk['capacity_loss'])
    assert np.isfinite(z['total_cost']) and 0 <= z['fill_rate'] <= 1.0000001
    assert (z['flow'].to_numpy() >= -1e-7).all() and (z['lost'].to_numpy() >= -1e-7).all()

# Direct scenario integration: same controls must alter demand/capacity outcomes.
z0=solve_allocation(s,r,c,f,[],.30,1,1,.96,False,False,capacity_loss=risk['capacity_loss'])
z1=solve_allocation(s,r,c,f,[],.30,1.4,1.4,.96,True,True,capacity_loss=risk['capacity_loss'])
assert np.isclose(z1['demand'].sum(), z0['demand'].sum()*1.4*1.25)
assert z1['fill_rate']<z0['fill_rate']
# Pooling sensitivity re-optimises store selection for each policy.
ps=pooling_sensitivity(s,r,c,f,1,1,.96,False,False,capacity_loss=risk['capacity_loss'])
assert list(ps['Pooling'])==[.30,.40,.50]
assert len(ps)==3
# Strategy page recalculates under stress.
strat,opt=compare_strategies(s,r,c,f,.30,1.4,1.4,.96,True,True,capacity_loss=risk['capacity_loss'])
assert len(strat)==3 and opt['result']['fill_rate']<1
# Simulation determinism + forced stress effect.
sim0=simulate(s,r,c,f,risk,stress['enabled'],.30,1.4,1.4,risk['promo_prob'],risk['disruption_prob'],250,42,False,False)
sim1=simulate(s,r,c,f,risk,stress['enabled'],.30,1.4,1.4,risk['promo_prob'],risk['disruption_prob'],250,42,True,True)
assert sim0.equals(simulate(s,r,c,f,risk,stress['enabled'],.30,1.4,1.4,risk['promo_prob'],risk['disruption_prob'],250,42,False,False))
assert sim1['Fill Rate'].mean() <= sim0['Fill Rate'].mean()+1e-9
print('ALL MODEL TESTS PASSED')
print('Base:', base['enabled'], base['result']['fill_rate'], base['result']['lost'].sum(), base['result']['total_cost'])
print('High demand:', high['enabled'], high['result']['fill_rate'], high['result']['lost'].sum(), high['result']['total_cost'])
print('Stress:', stress['enabled'], stress['result']['fill_rate'], stress['result']['lost'].sum(), stress['result']['total_cost'])

# Analytical sensitivity checks: every exposed control must propagate to model outputs.
# Service target: higher targets tighten the service constraint and should increase the
# required service and/or cost when the target is binding.
target_results=[]
for t in [.90,.94,.96,.98,.99]:
    o=best_enablement(s,r,c,f,.30,1.4,1.4,t,False,True,capacity_loss=risk['capacity_loss'])
    target_results.append(o['result'])
assert all(x['target_feasible'] for x in target_results)
assert all(target_results[i]['fill_rate'] <= target_results[i+1]['fill_rate']+1e-9 for i in range(len(target_results)-1))
assert all(target_results[i]['total_cost'] <= target_results[i+1]['total_cost']+1e-7 for i in range(len(target_results)-1))
assert target_results[0]['fill_rate'] >= .90-1e-9 and target_results[-1]['fill_rate'] >= .99-1e-9
# A target below achieved fill remains non-binding and therefore need not change the base allocation.
base90=best_enablement(s,r,c,f,.30,1,1,.90,False,False,capacity_loss=risk['capacity_loss'])
base99=best_enablement(s,r,c,f,.30,1,1,.99,False,False,capacity_loss=risk['capacity_loss'])
assert base90['enabled']==base99['enabled']==[]
assert base90['result']['fill_rate']==base99['result']['fill_rate']==1.0
# Pooling changes enabled-store online capacity when the pool cap is binding.
z30=solve_allocation(s,r,c,f,['S2','S4'],.30,1,1,0,False,False,capacity_loss=risk['capacity_loss'])
z50=solve_allocation(s,r,c,f,['S2','S4'],.50,1,1,0,False,False,capacity_loss=risk['capacity_loss'])
assert z50['flow'].loc['S2'].sum() >= z30['flow'].loc['S2'].sum()-1e-9
assert z50['flow'].loc['S4'].sum() >= z30['flow'].loc['S4'].sum()-1e-9
# Promotion and disruption propagate separately.
normal=solve_allocation(s,r,c,f,[],.30,1,1,0,False,False,capacity_loss=risk['capacity_loss'])
promo=solve_allocation(s,r,c,f,[],.30,1,1,0,False,True,capacity_loss=risk['capacity_loss'])
disrupt=solve_allocation(s,r,c,f,[],.30,1,1,0,True,False,capacity_loss=risk['capacity_loss'])
assert np.isclose(promo['demand'].sum(),normal['demand'].sum()*1.25)
assert disrupt['flow'].loc[['W1','W2']].sum().sum() <= normal['flow'].loc[['W1','W2']].sum().sum()+1e-9
# Stockout penalty and enablement cost are genuine decision inputs.
for sc in [200,420,700]:
    o=best_enablement(s,r,c,f,.30,1.4,1.4,.96,False,True,capacity_loss=risk['capacity_loss'],stockout_cost=sc)
    assert np.isfinite(o['result']['total_cost'])
sc_low=best_enablement(s,r,c,f,.30,1.4,1.4,.96,False,True,capacity_loss=risk['capacity_loss'],stockout_cost=200)['result']
sc_high=best_enablement(s,r,c,f,.30,1.4,1.4,.96,False,True,capacity_loss=risk['capacity_loss'],stockout_cost=700)['result']
assert sc_low['total_cost'] <= sc_high['total_cost']+1e-7
em_low=best_enablement(s,r,c,f,.30,1.4,1,.96,False,False,capacity_loss=risk['capacity_loss'],enablement_cost_mult=.5)
em_high=best_enablement(s,r,c,f,.30,1.4,1,.96,False,False,capacity_loss=risk['capacity_loss'],enablement_cost_mult=2.0)
assert em_low['result']['total_cost'] <= em_high['result']['total_cost']+1e-7
# Simulation event controls and risk inputs propagate and random event reporting is correct.
sim_p=simulate(s,r,c,f,risk,['S2','S4'],.30,1,1,1.0,0.0,1000,7,False,False)
sim_d=simulate(s,r,c,f,risk,['S2','S4'],.30,1,1,0.0,1.0,1000,7,False,False)
assert sim_p['Promotion'].mean()==1.0 and sim_d['Disrupted'].mean()==1.0
assert sim_p['Disrupted'].mean()==0.0 and sim_d['Promotion'].mean()==0.0
sim_base_no_events=simulate(s,r,c,f,risk,['S2','S4'],.30,1,1,0.0,0.0,1000,7,False,False)
assert sim_p['Total Cost'].mean() >= sim_base_no_events['Total Cost'].mean()-1e-9
for col in ['Transfer Cost','Excess Holding Cost','Markdown Cost','Total Cost','Return Rate']:
    assert col in sim_p.columns
print('ANALYTICAL SENSITIVITY TESTS PASSED')

# Literal Goal Programming and MINIMAX model checks.
gp=goal_programming(s,r,c,f,[],.30,1,1,.96,False,False,capacity_loss=risk['capacity_loss'])
assert abs(gp['fill_rate']-.96)<1e-9
assert gp['deviations']['service_shortfall'] < 1e-9
assert np.isfinite(gp['objective'])
mm=minimax_pooling(s,r,c,f,1,1,.96,False,False,capacity_loss=risk['capacity_loss'])
assert list(mm['Pooling'])==[.30,.40,.50]
assert mm['Q'].ge(0).all() and mm['Q'].le(1).all()
assert mm['Fill Rate'].between(0,1.0000001).all()
mm_stress=minimax_pooling(s,r,c,f,1.4,1.4,.96,True,True,capacity_loss=risk['capacity_loss'])
assert mm_stress['Fill Rate'].between(0,1.0000001).all()
print('GOAL PROGRAMMING + MINIMAX TESTS PASSED')

