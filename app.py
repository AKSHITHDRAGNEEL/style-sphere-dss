import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from model import load_model, solve_allocation, best_enablement, store_economics, compare_strategies, pooling_sensitivity, goal_programming, minimax_pooling, simulate

st.set_page_config(page_title='Style Sphere DSS', page_icon='◈', layout='wide', initial_sidebar_state='expanded')

st.markdown('''
<style>
.block-container {padding-top:1.35rem; padding-bottom:2rem; max-width:1450px;}
.metric-card {border:1px solid rgba(148,163,184,.25);border-radius:14px;padding:15px 17px;background:rgba(255,255,255,.025);min-height:104px;}
.metric-label{font-size:.80rem;color:#9ca3af;margin-bottom:7px}.metric-value{font-size:1.55rem;font-weight:700;line-height:1.1}.metric-sub{font-size:.73rem;color:#9ca3af;margin-top:7px}
.reco{border-left:4px solid #22c55e;background:rgba(34,197,94,.10);padding:14px 17px;border-radius:9px;margin-bottom:8px}
.scenario{border:1px solid rgba(148,163,184,.20);border-radius:10px;padding:9px 12px;background:rgba(255,255,255,.025);font-size:.82rem}
</style>
''', unsafe_allow_html=True)

DATA = Path(__file__).with_name('Style_Sphere_Model.xlsx')
stores, regions, costs, facilities, risk = load_model(DATA)
STORES = stores['Store'].tolist(); REGIONS = regions['Region'].tolist()

with st.sidebar:
    st.header('Scenario controls')
    st.caption('Current-scenario controls change the live network decision. Risk probabilities are used by the simulation.')
    pooling = st.select_slider('Pooling limit', options=[.30,.40,.50], value=float(risk['pooling']), format_func=lambda x:f'{x:.0%}')
    online_pct = st.slider('Online demand (% of base)', 70, 140, 100, 5)
    store_pct = st.slider('Store demand (% of base)', 70, 140, 100, 5)
    service_pct = st.slider('Online fill-rate goal (%)', 90, 99, 96, 1)
    enablement_mult = st.slider('Enablement cost multiplier', 50, 200, 100, 10) / 100
    stockout_penalty = st.slider('Store stockout penalty (₹/unit)', 200, 700, int(risk['SC']), 20)
    online_mult, store_mult, service = online_pct/100, store_pct/100, service_pct/100

    st.divider(); st.subheader('Current operating scenario')
    promo_active = st.checkbox('Promotion active now', value=False)
    warehouse_disruption = st.checkbox('Warehouse disruption active now', value=False)

    st.divider(); st.subheader('Store policy')
    selection_mode = st.radio('Store selection', ['Model recommended','Manual'], index=0)
    if selection_mode == 'Manual':
        enabled = st.multiselect('Enabled stores', STORES, default=[])
    else:
        optimal_preview = best_enablement(stores, regions, costs, facilities, pooling, online_mult, store_mult, service,
                                          warehouse_disruption, promo_active, capacity_loss=risk['capacity_loss'], stockout_cost=stockout_penalty, enablement_cost_mult=enablement_mult)
        enabled = optimal_preview['enabled']
        st.info('Recommended: ' + (', '.join(enabled) if enabled else 'No stores'))
        st.caption('Store enablement is selected by minimum total monthly cost subject to the service target; one-time enablement cost is amortised over 36 months for comparison.')

    st.divider(); st.subheader('Risk assumptions')
    st.caption('These probabilities affect the Risk & What-if simulation. They do not mean the event is happening now.')
    promo_pct = st.slider('Promotion probability (%)', 0, 40, int(round(risk['promo_prob']*100)), 5)
    disruption_pct = st.slider('Warehouse disruption probability (%)', 0, 20, int(round(risk['disruption_prob']*100)), 2)

promo_prob, disruption_prob = promo_pct/100, disruption_pct/100
result = solve_allocation(stores, regions, costs, facilities, enabled, pooling, online_mult, store_mult, service,
                          warehouse_disruption, promo_active, capacity_loss=risk['capacity_loss'], stockout_cost=stockout_penalty, enablement_cost_mult=enablement_mult)
strategy_df, optimal = compare_strategies(stores, regions, costs, facilities, pooling, online_mult, store_mult, service,
                                          warehouse_disruption, promo_active, capacity_loss=risk['capacity_loss'], stockout_cost=stockout_penalty, enablement_cost_mult=enablement_mult)

st.title('Style Sphere | Omnichannel Fulfilment DSS')
st.caption('Model-driven management dashboard • change assumptions to see the decision change')

scenario_bits=[]
if online_pct!=100: scenario_bits.append(f'Online demand {online_pct}%')
if store_pct!=100: scenario_bits.append(f'Store demand {store_pct}%')
if promo_active: scenario_bits.append('Promotion active')
if warehouse_disruption: scenario_bits.append('Warehouse disruption active')
scenario_label=' • '.join(scenario_bits) if scenario_bits else 'Base operating conditions'

kpi=st.columns(4)
vals=[('Online fill rate',f"{result['fill_rate']:.1%}",f'Goal: {service:.0%}'),('Monthly total cost',f"₹{result['total_cost']:,.0f}",'Fulfilment + lost sales + enablement'),('Lost sales',f"{result['lost'].sum():,.0f}",'Unfulfilled online orders'),('Stores enabled',f"{len(enabled)}",', '.join(enabled) if enabled else 'Warehouse-led network')]
for col,(lab,val,sub) in zip(kpi,vals):
    with col: st.markdown(f'<div class="metric-card"><div class="metric-label">{lab}</div><div class="metric-value">{val}</div><div class="metric-sub">{sub}</div></div>',unsafe_allow_html=True)

if selection_mode=='Model recommended':
    if result['target_feasible'] and not enabled:
        rec=(f"Warehouse-led fulfilment is sufficient under the current scenario. No stores are required to meet the {service:.0%} service target at the minimum-cost configuration; pooling remains available as a policy lever.")
    elif result['target_feasible'] and enabled:
        rec=(f"Selective store fulfilment is recommended: use {pooling:.0%} pooling with {', '.join(enabled)}. This configuration meets the {service:.0%} service target at the minimum-cost feasible network.")
    else:
        rec=(f"The {service:.0%} service target is not feasible under the current scenario even after optimisation. Review capacity, pooling and store participation; the model is showing the lowest-cost feasible allocation.")
else:
    if result['target_feasible']:
        rec=(f"Manual policy: {pooling:.0%} pooling with {', '.join(enabled) if enabled else 'no stores enabled'}. The configuration meets the {service:.0%} service target under the current scenario.")
    else:
        rec=(f"Manual policy: {pooling:.0%} pooling with {', '.join(enabled) if enabled else 'no stores enabled'}. The {service:.0%} service target is not feasible under this configuration.")
st.markdown(f'<div class="reco"><strong>Management recommendation</strong><br>{rec}</div>',unsafe_allow_html=True)
st.markdown(f'<div class="scenario"><strong>Current scenario:</strong> {scenario_label}</div>',unsafe_allow_html=True)
st.caption('Interpretation: “service target met” means the optimised allocation can fulfil at least the requested share of online demand. A scenario input does not have to change the recommendation if the current network still has sufficient capacity.')
target_gap=result['fill_rate']-service
if result['target_feasible']:
    st.success(f'Service target status: MET • actual fill rate {result["fill_rate"]:.1%} vs target {service:.0%}. The target is {'non-binding' if target_gap > 1e-9 else 'binding'} in this allocation.')
else:
    st.error(f'Service target status: NOT FEASIBLE • best achievable fill rate is {result["fill_rate"]:.1%} vs target {service:.0%}.')

st.write('')
t1,t2,t3,t4,t5=st.tabs(['Executive','Network','Store decisions','Strategy & pooling','Risk & what-if'])

with t1:
    a,b=st.columns([1.1,1])
    with a:
        st.subheader('Current decision')
        d=pd.DataFrame({'Metric':['Fulfilment cost','Lost-sale cost','Enablement cost','Total monthly cost','Online fill rate'],
                        'Value':[f"₹{result['fulfil_cost']:,.0f}",f"₹{result['lost_cost']:,.0f}",f"₹{result['enable_cost']:,.0f}",f"₹{result['total_cost']:,.0f}",f"{result['fill_rate']:.1%}"]})
        st.dataframe(d,hide_index=True,use_container_width=True)
    with b:
        st.subheader('Online demand by region')
        dd=pd.DataFrame({'Region':REGIONS,'Online demand':result['demand'].values})
        fig=px.bar(dd,x='Region',y='Online demand',text_auto='.0f'); fig.update_layout(height=310,margin=dict(l=5,r=5,t=15,b=5),showlegend=False)
        st.plotly_chart(fig,use_container_width=True)

with t2:
    st.subheader('Where online orders are fulfilled from')
    flow=result['flow'].clip(lower=0)
    display=flow.mask(flow.abs()<0.5,0).round(0)
    st.dataframe(display.style.format('{:,.0f}'),use_container_width=True)
    active=flow.stack().reset_index(); active.columns=['Facility','Region','Orders']; active=active[active['Orders']>0.5]
    if not active.empty:
        fig=px.bar(active,x='Region',y='Orders',color='Facility',barmode='stack',text_auto='.0f'); fig.update_layout(height=350,margin=dict(l=5,r=5,t=15,b=5),yaxis_title='Orders fulfilled')
        st.plotly_chart(fig,use_container_width=True)
    util=[]
    for f in facilities.index:
        cap=float(facilities.loc[f,'Capacity']); used=float(flow.loc[f].sum()); effective=cap*(1-risk['capacity_loss']) if warehouse_disruption and facilities.loc[f,'Type']=='Warehouse' else cap
        util.append({'Facility':f,'Utilisation':used/effective if effective else 0})
    udf=pd.DataFrame(util)
    st.subheader('Capacity utilisation')
    fig=px.bar(udf,x='Utilisation',y='Facility',orientation='h',text=udf['Utilisation'].map(lambda x:f'{x:.0%}')); fig.update_xaxes(range=[0,1.05],tickformat='.0%'); fig.update_layout(height=300,margin=dict(l=5,r=5,t=15,b=5),showlegend=False)
    st.plotly_chart(fig,use_container_width=True)

with t3:
    st.subheader('Store enablement economics')
    et=store_economics(stores,regions,costs,facilities,pooling,online_mult,store_mult,36,warehouse_disruption,promo_active,capacity_loss=risk['capacity_loss'], stockout_cost=stockout_penalty, enablement_cost_mult=enablement_mult)
    et['Decision']=et['Store'].isin(enabled).map({True:'Enabled',False:'Not enabled'})
    et['Payback (months)']=et['Payback (months)'].replace([float('inf')],pd.NA)
    fmt={'Local Demand':'{:,.0f}','Capacity':'{:,.0f}','Enablement Cost (₹ lakh)':'₹{:,.1f}','Monthly Saving':'₹{:,.0f}','Payback (months)':'{:,.1f}'}
    st.dataframe(et.style.format(fmt,na_rep='—'),hide_index=True,use_container_width=True)
    st.caption('Payback = one-time enablement cost ÷ estimated monthly fulfilment-cost saving. The 36-month hurdle is a planning assumption used for comparison.')
    st.markdown('**Current model-selected stores:** ' + (', '.join(enabled) if enabled else 'None') if selection_mode=='Model recommended' else '**Manual store selection:** ' + (', '.join(enabled) if enabled else 'None'))

with t4:
    st.subheader('Strategy comparison')
    st.caption('Each policy is recalculated under the current scenario.')
    st.dataframe(strategy_df.rename(columns={'Total Cost':'Monthly total cost','Fill Rate':'Online fill rate','Lost Sales':'Lost sales'}).style.format({'Monthly total cost':'₹{:,.0f}','Online fill rate':'{:.1%}','Lost sales':'{:,.0f}'}),hide_index=True,use_container_width=True)
    a,b=st.columns(2)
    with a:
        fig=px.bar(strategy_df,x='Strategy',y='Total Cost',text_auto='.3s'); fig.update_layout(height=330,margin=dict(l=5,r=5,t=15,b=5),yaxis_title='Monthly total cost (₹)'); st.plotly_chart(fig,use_container_width=True)
    with b:
        fig=px.bar(strategy_df,x='Strategy',y='Fill Rate',text_auto='.1%'); fig.add_hline(y=service,line_dash='dash',annotation_text=f'Goal {service:.0%}'); fig.update_yaxes(range=[max(.75,float(strategy_df['Fill Rate'].min())-.03),1.01],tickformat='.0%'); fig.update_layout(height=330,margin=dict(l=5,r=5,t=15,b=5),yaxis_title='Online fill rate'); st.plotly_chart(fig,use_container_width=True)
    st.subheader('Pooling sensitivity')
    pdf=pooling_sensitivity(stores,regions,costs,facilities,online_mult,store_mult,service,warehouse_disruption,promo_active,capacity_loss=risk['capacity_loss'], stockout_cost=stockout_penalty, enablement_cost_mult=enablement_mult)
    st.dataframe(pdf.style.format({'Pooling':'{:.0%}','Total Cost':'₹{:,.0f}','Fill Rate':'{:.1%}','Lost Sales':'{:,.0f}'}),hide_index=True,use_container_width=True)
    fig=px.line(pdf,x='Pooling',y='Total Cost',markers=True,text=pdf['Total Cost'].map(lambda x:f'₹{x/1e6:.2f}M')); fig.update_xaxes(tickvals=[.3,.4,.5],tickformat='.0%'); fig.update_layout(height=300,margin=dict(l=5,r=5,t=15,b=5),yaxis_title='Monthly total cost (₹)'); st.plotly_chart(fig,use_container_width=True)

    st.subheader('Multi-objective decision models')
    st.caption('Goal Programming balances service, store protection, cost and excess-inventory goals. MINIMAX balances normalised economic-cost and service regret across the three pooling policies.')
    gp = goal_programming(stores, regions, costs, facilities, enabled, pooling, online_mult, store_mult, service,
                          warehouse_disruption, promo_active, capacity_loss=risk['capacity_loss'], lost_sale_cost=risk['LC'],
                          stockout_cost=stockout_penalty, enablement_cost_mult=enablement_mult)
    gpd = pd.DataFrame({'Goal / deviation':['Service shortfall','Store over-diversion','Cost overrun','Excess / markdown deviation'],
                        'Value':[gp['deviations']['service_shortfall'],gp['deviations']['store_overdiversion'],gp['deviations']['cost_overrun'],gp['deviations']['markdown_excess']]})
    st.dataframe(gpd.style.format({'Value':'{:,.2f}'}),hide_index=True,use_container_width=True)
    mm = minimax_pooling(stores, regions, costs, facilities, online_mult, store_mult, service,
                         warehouse_disruption, promo_active, capacity_loss=risk['capacity_loss'], lost_sale_cost=risk['LC'],
                         stockout_cost=stockout_penalty, enablement_cost_mult=enablement_mult)
    st.dataframe(mm[['Pooling','Stores','Q','Total Cost','Fill Rate','Cost Regret','Service Regret']].style.format({
        'Pooling':'{:.0%}','Q':'{:.3f}','Total Cost':'₹{:,.0f}','Fill Rate':'{:.1%}','Cost Regret':'{:.1%}','Service Regret':'{:.1%}'
    }),hide_index=True,use_container_width=True)

with t5:
    st.subheader('What-if & risk view')
    a,b,c=st.columns(3)
    with a: trials=st.select_slider('Simulation trials',options=[1000,2500,5000],value=1000)
    with b: seed=st.number_input('Simulation seed',min_value=0,max_value=999999,value=42,step=1)
    with c:
        force_promo=st.checkbox('Force promotion in simulation',value=False)
        force_disruption=st.checkbox('Force warehouse disruption in simulation',value=False)
    sim=simulate(stores,regions,costs,facilities,risk,enabled,pooling,online_mult,store_mult,promo_prob,disruption_prob,trials,seed,force_promo,force_disruption)
    sc=st.columns(4)
    sc[0].metric('Mean fill rate',f"{sim['Fill Rate'].mean():.1%}")
    sc[1].metric('Mean lost sales',f"{sim['Lost Sales'].mean():,.0f}")
    sc[2].metric('90th pct. lost sales',f"{sim['Lost Sales'].quantile(.90):,.0f}")
    sc[3].metric('90th pct. fulfilment cost',f"₹{sim['Fulfilment Cost'].quantile(.90):,.0f}")
    st.subheader('Risk distribution')
    fig=px.histogram(sim,x='Fill Rate',nbins=25); fig.add_vline(x=service,line_dash='dash',annotation_text=f'Goal {service:.0%}'); fig.update_xaxes(tickformat='.0%'); fig.update_layout(height=360,margin=dict(l=5,r=5,t=15,b=5)); st.plotly_chart(fig,use_container_width=True)
    rs=pd.DataFrame({'Risk metric':['Store stockout units','Excess inventory','Markdown units'],'Mean':[sim['Store Stockout Units'].mean(),sim['Excess Inventory'].mean(),sim['Markdown Units'].mean()],'90th percentile':[sim['Store Stockout Units'].quantile(.9),sim['Excess Inventory'].quantile(.9),sim['Markdown Units'].quantile(.9)]})
    st.dataframe(rs.style.format({'Mean':'{:,.0f}','90th percentile':'{:,.0f}'}),hide_index=True,use_container_width=True)
    st.caption('The simulation uses the uncertainty assumptions in the Style Sphere workbook. “Force” controls are explicit stress-test overrides for the current simulation only.')
