from pathlib import Path
import itertools
import numpy as np
import pandas as pd
from scipy.optimize import linprog


def _num(v):
    if pd.isna(v):
        return np.nan
    s = str(v).replace(',', '').replace('₹', '').replace('%', '').strip()
    x = pd.to_numeric(s, errors='coerce')
    return float(x) if pd.notna(x) else np.nan


def _find(raw, label):
    m = raw.index[raw.iloc[:, 0].astype(str).str.strip().eq(label)]
    if len(m) == 0:
        raise ValueError(f"Missing '{label}' in Data sheet")
    return int(m[0])


def _require_numeric(df, cols, name):
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if df[col].isna().any():
            raise ValueError(f"Non-numeric or missing values in {name}: {col}")


def load_model(path):
    raw = pd.read_excel(path, sheet_name='Data', header=None)

    h = _find(raw, 'Location')
    stores, warehouses = [], []
    r = h + 1
    while r < len(raw):
        loc = str(raw.iat[r, 0]).strip()
        typ = str(raw.iat[r, 1]).strip()
        if loc in ('', 'nan') or typ in ('', 'nan'):
            break
        facility = loc.split()[0] if typ == 'Warehouse' else loc
        rec = {'Facility': facility, 'Capacity': _num(raw.iat[r, 3]), 'Type': typ}
        if typ == 'Store':
            rec.update({'Store': loc, 'Local Demand': _num(raw.iat[r, 2]), 'Enablement Lakh': _num(raw.iat[r, 4])})
            stores.append(rec)
        elif typ == 'Warehouse':
            warehouses.append(rec)
        r += 1

    stores = pd.DataFrame(stores)
    wh = pd.DataFrame(warehouses)
    _require_numeric(stores, ['Capacity', 'Local Demand', 'Enablement Lakh'], 'stores')
    _require_numeric(wh, ['Capacity'], 'warehouses')
    facilities = pd.concat([wh[['Facility', 'Capacity', 'Type']], stores[['Facility', 'Capacity', 'Type']]], ignore_index=True).set_index('Facility')

    h = _find(raw, 'Region')
    regions = []
    r = h + 1
    while r < len(raw):
        reg = str(raw.iat[r, 0]).strip()
        if reg in ('', 'nan'):
            break
        regions.append({'Region': reg, 'Mean Demand': _num(raw.iat[r, 1]), 'Std Dev': _num(raw.iat[r, 2]), 'Target Share': _num(raw.iat[r, 3])})
        r += 1
    regions = pd.DataFrame(regions)
    _require_numeric(regions, ['Mean Demand', 'Std Dev', 'Target Share'], 'regions')

    h = _find(raw, 'Facility')
    regs = regions['Region'].tolist()
    rows = []
    r = h + 1
    while r < len(raw):
        f = str(raw.iat[r, 0]).strip()
        if f in ('', 'nan'):
            break
        if f in facilities.index:
            rows.append({'Facility': f, **{reg: _num(raw.iat[r, j + 1]) for j, reg in enumerate(regs)}})
        r += 1
    costs = pd.DataFrame(rows).set_index('Facility')
    _require_numeric(costs.reset_index(), regs, 'cost matrix')
    costs = costs.astype(float)

    risk = {
        'online_sd': _num(raw.iat[_find(raw, 'Online regional demand std dev'), 1]),
        'store_sd': _num(raw.iat[_find(raw, 'Store demand std dev'), 1]),
        'promo_prob': _num(raw.iat[_find(raw, 'Promotional surge probability (x1.25 online demand)'), 1]),
        'local_spike_prob': _num(raw.iat[_find(raw, 'Local demand spike probability'), 1]),
        'local_spike_size': _num(raw.iat[_find(raw, 'Local demand spike size'), 1]),
        'disruption_prob': _num(raw.iat[_find(raw, 'Labor disruption probability'), 1]),
        'capacity_loss': _num(raw.iat[_find(raw, 'Warehouse capacity loss (labor disruption)'), 1]),
        'resaleable': _num(raw.iat[_find(raw, 'Returned inventory immediately resaleable share'), 1]),
        'pooling': _num(raw.iat[_find(raw, 'Pooling limit p (policy lever - change this!)'), 1]),
        'return_probs': np.array([_num(raw.iat[_find(raw, x), 1]) for x in ['Low', 'Normal', 'High']], dtype=float),
        'return_rates': np.array([_num(raw.iat[_find(raw, x), 2]) for x in ['Low', 'Normal', 'High']], dtype=float),
        'LC': _num(raw.iat[_find(raw, 'Online lost-sale contribution loss (/order)'), 1]),
        'SC': _num(raw.iat[_find(raw, 'Store stockout contribution loss (/unit)'), 1]),
        'HC': _num(raw.iat[_find(raw, 'End-of-month excess inventory holding (/unit)'), 1]),
        'MC': _num(raw.iat[_find(raw, 'Markdown on aged fashion inventory (/unit)'), 1]),
        'TC': _num(raw.iat[_find(raw, 'Store-to-store transfer (/unit)'), 1]),
    }
    for k, v in risk.items():
        if isinstance(v, (float, int)) and not np.isfinite(v):
            raise ValueError(f"Invalid risk parameter: {k}")
    if not np.isclose(risk['return_probs'].sum(), 1.0):
        risk['return_probs'] = risk['return_probs'] / risk['return_probs'].sum()
    return stores, regions, costs, facilities, risk


def scenario_demand(regions, online_mult=1.0, promo_active=False, promo_multiplier=1.25):
    demand = regions['Mean Demand'].to_numpy(dtype=float) * float(online_mult)
    if promo_active:
        demand *= float(promo_multiplier)
    return demand


def solve_allocation(stores, regions, costs, facilities, enabled, pooling, online_mult=1.0, store_mult=1.0,
                     service_target=0.96, warehouse_disruption=False, promo_active=False,
                     promo_multiplier=1.25, capacity_loss=0.15, lost_sale_cost=360.0, stockout_cost=420.0, enablement_cost_mult=1.0):
    regs = regions['Region'].tolist()
    sts = stores['Store'].tolist()
    fac = facilities.index.tolist()
    n_f, n_r = len(fac), len(regs)
    n = n_f * n_r + n_r + len(sts)
    demand = scenario_demand(regions, online_mult, promo_active, promo_multiplier)
    local = stores['Local Demand'].to_numpy(dtype=float) * float(store_mult)

    def ix(i, j):
        return i * n_r + j

    c = np.zeros(n, dtype=float)
    for i, f in enumerate(fac):
        for j, reg in enumerate(regs):
            c[ix(i, j)] = float(costs.loc[f, reg])
    c[n_f * n_r:n_f * n_r + n_r] = float(lost_sale_cost)
    c[n_f * n_r + n_r:] = float(stockout_cost)

    A, b = [], []
    for i, f in enumerate(fac):
        cap = float(facilities.loc[f, 'Capacity'])
        if f in sts:
            if f not in enabled:
                cap = 0.0
            else:
                # Online diversion is limited by pooling; local-demand protection
                # is handled as a soft stockout penalty so the stockout-cost
                # control has a genuine analytical effect.
                cap = min(float(pooling) * cap, cap)
        elif warehouse_disruption:
            cap *= (1.0 - float(capacity_loss))
        row = np.zeros(n); row[[ix(i, j) for j in range(n_r)]] = 1.0
        A.append(row); b.append(max(0.0, cap))

    # Store capacity balance: online diversion + local demand - stockout <= capacity.
    # This lets the stockout penalty govern the trade-off when online service is
    # more/less valuable than protecting walk-in inventory.
    for k, f in enumerate(sts):
        i = fac.index(f)
        row = np.zeros(n)
        row[[ix(i, j) for j in range(n_r)]] = 1.0
        row[n_f * n_r + n_r + k] = -1.0
        A.append(row)
        b.append(float(facilities.loc[f, 'Capacity']) - float(local[k]))

    # Aggregate service constraint. If it is infeasible, the function below
    # intentionally falls back to the minimum-cost capacity allocation.
    row = np.zeros(n); row[n_f * n_r:n_f * n_r + n_r] = 1.0
    A.append(row); b.append(max(0.0, (1.0 - float(service_target)) * demand.sum()))

    E, eb = [], []
    for j in range(n_r):
        row = np.zeros(n)
        row[[ix(i, j) for i in range(n_f)]] = 1.0
        row[n_f * n_r + j] = 1.0
        E.append(row); eb.append(float(demand[j]))

    kwargs = dict(A_ub=np.asarray(A), b_ub=np.asarray(b), A_eq=np.asarray(E), b_eq=np.asarray(eb), bounds=[(0, None)] * n, method='highs')
    res = linprog(c, **kwargs)
    target_feasible = bool(res.success)
    if not res.success:
        kwargs['A_ub'] = np.asarray(A[:-1]); kwargs['b_ub'] = np.asarray(b[:-1])
        res = linprog(c, **kwargs)
    if not res.success:
        raise RuntimeError('No feasible allocation could be calculated for this scenario.')

    # Preserve whether the ORIGINAL requested service target was feasible.
    # The fallback solution below is capacity-feasible, but it must not be
    # reported as target-feasible merely because the fallback removed the
    # service constraint.
    requested_target_feasible = target_feasible

    x = np.maximum(res.x[:n_f * n_r].reshape(n_f, n_r), 0.0)
    lost = np.maximum(res.x[n_f * n_r:n_f * n_r + n_r], 0.0)
    stockout = np.maximum(res.x[n_f * n_r + n_r:], 0.0)
    flow = pd.DataFrame(x, index=fac, columns=regs)
    fulfil_cost = float(np.sum(x * np.array([[costs.loc[f, r] for r in regs] for f in fac], dtype=float)))
    lost_cost = float(lost.sum() * lost_sale_cost)
    enable_cost = float(sum(stores.loc[stores['Store'].isin(enabled), 'Enablement Lakh']) * 100000.0 / 36.0 * float(enablement_cost_mult))
    stockout_cost_total = float(stockout.sum() * stockout_cost)
    total_cost = fulfil_cost + lost_cost + stockout_cost_total + enable_cost
    fill_rate = float(1.0 - lost.sum() / demand.sum()) if demand.sum() else 1.0
    return {'flow': flow, 'lost': pd.Series(lost, index=regs), 'demand': pd.Series(demand, index=regs),
            'fulfil_cost': fulfil_cost, 'lost_cost': lost_cost, 'enable_cost': enable_cost,
            'total_cost': total_cost, 'stockout': pd.Series(stockout, index=sts), 'stockout_cost': stockout_cost_total, 'fill_rate': fill_rate, 'target_feasible': requested_target_feasible}


def best_enablement(stores, regions, costs, facilities, pooling, online_mult=1.0, store_mult=1.0,
                    service_target=0.96, warehouse_disruption=False, promo_active=False,
                    capacity_loss=0.15, lost_sale_cost=360.0, stockout_cost=420.0, enablement_cost_mult=1.0):
    names = stores['Store'].tolist()
    best = None
    # First choose the lowest total cost among configurations that meet the target.
    for k in range(len(names) + 1):
        for combo in itertools.combinations(names, k):
            z = solve_allocation(stores, regions, costs, facilities, list(combo), pooling, online_mult, store_mult,
                                 service_target, warehouse_disruption, promo_active, capacity_loss=capacity_loss,
                                 lost_sale_cost=lost_sale_cost, stockout_cost=stockout_cost, enablement_cost_mult=enablement_cost_mult)
            if z['target_feasible']:
                score = z['total_cost']
                if best is None or score < best['score'] - 1e-7:
                    best = {'score': score, 'enabled': list(combo), 'result': z}
    if best is None:
        # If the service target cannot be met, select the lowest-cost capacity-feasible network.
        for k in range(len(names) + 1):
            for combo in itertools.combinations(names, k):
                z = solve_allocation(stores, regions, costs, facilities, list(combo), pooling, online_mult, store_mult,
                                     0.0, warehouse_disruption, promo_active, capacity_loss=capacity_loss,
                                     lost_sale_cost=lost_sale_cost, stockout_cost=stockout_cost, enablement_cost_mult=enablement_cost_mult)
                if best is None or z['total_cost'] < best['score'] - 1e-7:
                    # This branch is used only when the requested service target
                    # cannot be met by any store configuration. Preserve that
                    # status explicitly instead of inheriting the fallback
                    # solve's relaxed (target=0) feasibility flag.
                    z['target_feasible'] = False
                    best = {'score': z['total_cost'], 'enabled': list(combo), 'result': z}
    return best



def goal_programming(stores, regions, costs, facilities, enabled, pooling, online_mult=1.0, store_mult=1.0,
                     service_target=0.96, warehouse_disruption=False, promo_active=False,
                     promo_multiplier=1.25, capacity_loss=0.15, lost_sale_cost=360.0, stockout_cost=420.0,
                     enablement_cost_mult=1.0, store_availability_target=0.0, cost_budget=None,
                     excess_target=0.0, weights=(1.0, 1.0, 1.0, 1.0)):
    """Weighted Goal Programming version of the Style Sphere model.

    G1: online fulfilled units target (service_target * demand).
    G2: store online diversion target (store_availability_target * capacity).
    G3: fulfilment-cost budget target (cost_budget).
    G4: excess-inventory target (excess_target).

    The four goals use positive/negative deviation variables. Capacity and demand
    balance remain hard constraints. Defaults are deliberately conservative:
    protect all walk-in store demand (G2 target = 0), use the minimum-cost
    deterministic allocation as the G3 reference budget, and set excess target
    to zero because no starting inventory balance is supplied in the case.
    """
    regs = regions['Region'].tolist(); sts = stores['Store'].tolist(); fac = facilities.index.tolist()
    n_f, n_r, n_s = len(fac), len(regs), len(sts)
    demand = scenario_demand(regions, online_mult, promo_active, promo_multiplier)
    local = stores['Local Demand'].to_numpy(float) * float(store_mult)
    flow_n = n_f*n_r
    lost0 = flow_n; stock0 = lost0+n_r
    # Deviation blocks: d1-,d1+, then d2-/d2+ per store, d3-,d3+, d4-,d4+
    d1m,d1p = stock0+n_s, stock0+n_s+1
    d2m0 = d1p+1; d2p0 = d2m0+n_s
    d3m,d3p = d2p0+n_s, d2p0+n_s+1
    d4m,d4p = d3p+1,d3p+2
    n= d4p+1
    c=np.zeros(n)
    # Goal-programming objective: normalized weighted deviations only.
    total_d=float(demand.sum())
    budget = cost_budget
    if budget is None:
        ref=solve_allocation(stores,regions,costs,facilities,enabled,pooling,online_mult,store_mult,0.0,
                             warehouse_disruption,promo_active,promo_multiplier,capacity_loss,lost_sale_cost,stockout_cost,enablement_cost_mult)
        budget=float(ref['fulfil_cost'])
    # Avoid zero denominators.
    scales=[max(float(service_target)*total_d,1.0), max(float(sum(facilities.loc[x,'Capacity'] for x in sts)),1.0), max(abs(budget),1.0), max(abs(excess_target),1.0)]
    w1,w2,w3,w4=[float(x) for x in weights]
    c[d1m]=w1/scales[0]
    for k in range(n_s): c[d2p0+k]=w2/scales[1]
    c[d3p]=w3/scales[2]
    c[d4p]=w4/scales[3]

    A=[]; b=[]
    # Facility capacities.
    for i,f in enumerate(fac):
        cap=float(facilities.loc[f,'Capacity'])
        if f in sts:
            if f not in enabled: cap=0.0
            else: cap=min(float(pooling)*cap, cap)
        elif warehouse_disruption: cap*=1.0-float(capacity_loss)
        row=np.zeros(n); row[[i*n_r+j for j in range(n_r)]]=1.0
        A.append(row); b.append(max(0.0,cap))
    # Hard local stock protection: online diversion + local demand - stockout <= capacity.
    for k,f in enumerate(sts):
        i=fac.index(f); row=np.zeros(n)
        row[[i*n_r+j for j in range(n_r)]]=1.0; row[stock0+k]=-1.0
        A.append(row); b.append(float(facilities.loc[f,'Capacity'])-float(local[k]))

    E=[]; eb=[]
    for j in range(n_r):
        row=np.zeros(n); row[[i*n_r+j for i in range(n_f)]]=1.0; row[lost0+j]=1.0
        E.append(row); eb.append(float(demand[j]))

    # G1: fulfilled + d1- - d1+ = target*D.
    row=np.zeros(n); row[:flow_n]=1.0; row[d1m]=1.0; row[d1p]=-1.0
    E.append(row); eb.append(float(service_target)*total_d)
    # G2: online diversion - d2+ + d2- = r*Cap. Each store.
    for k,f in enumerate(sts):
        i=fac.index(f); row=np.zeros(n); row[[i*n_r+j for j in range(n_r)]]=1.0
        row[d2m0+k]=1.0; row[d2p0+k]=-1.0
        E.append(row); eb.append(float(store_availability_target)*float(facilities.loc[f,'Capacity']))
    # G3 fulfilment cost + d3- - d3+ = budget.
    row=np.zeros(n)
    for i,f in enumerate(fac):
        for j,r in enumerate(regs): row[i*n_r+j]=float(costs.loc[f,r])
    row[d3m]=1.0; row[d3p]=-1.0; E.append(row); eb.append(float(budget))
    # G4 excess inventory + d4- - d4+ = target. No inventory state is supplied,
    # so the deterministic model's excess proxy is zero; simulation carries the
    # full excess-inventory calculation.
    row=np.zeros(n); row[d4m]=1.0; row[d4p]=-1.0; E.append(row); eb.append(float(excess_target))

    res=linprog(c,A_ub=np.asarray(A),b_ub=np.asarray(b),A_eq=np.asarray(E),b_eq=np.asarray(eb),
                bounds=[(0,None)]*n,method='highs')
    if not res.success: raise RuntimeError('Goal Programming model is infeasible for this scenario.')
    x=np.maximum(res.x[:flow_n].reshape(n_f,n_r),0.0); lost=np.maximum(res.x[lost0:lost0+n_r],0.0)
    stock=np.maximum(res.x[stock0:stock0+n_s],0.0)
    fulfil_cost=float(np.sum(x*np.array([[costs.loc[f,r] for r in regs] for f in fac],float)))
    enable_cost=float(sum(stores.loc[stores['Store'].isin(enabled),'Enablement Lakh'])*100000/36*float(enablement_cost_mult))
    lost_cost=float(lost.sum()*lost_sale_cost); stock_cost=float(stock.sum()*stockout_cost)
    total_cost=fulfil_cost+enable_cost+lost_cost+stock_cost
    return {
        'flow':pd.DataFrame(x,index=fac,columns=regs),'lost':pd.Series(lost,index=regs),
        'stockout':pd.Series(stock,index=sts),'demand':pd.Series(demand,index=regs),
        'fill_rate':float(1-lost.sum()/total_d) if total_d else 1.0,
        'fulfil_cost':fulfil_cost,'enable_cost':enable_cost,'lost_cost':lost_cost,
        'stockout_cost':stock_cost,'total_cost':total_cost,
        'objective':float(res.fun),'budget':budget,
        'deviations':{'service_shortfall':float(res.x[d1m]),'service_excess':float(res.x[d1p]),
                      'store_overdiversion':float(np.sum(res.x[d2p0:d2p0+n_s])),
                      'cost_overrun':float(res.x[d3p]),'markdown_excess':float(res.x[d4p])}
    }


def minimax_pooling(stores, regions, costs, facilities, online_mult=1.0, store_mult=1.0,
                    service_target=0.96, warehouse_disruption=False, promo_active=False,
                    capacity_loss=0.15, lost_sale_cost=360.0, stockout_cost=420.0,
                    enablement_cost_mult=1.0):
    """True MINIMAX cost-vs-service model, enumerating the 32 store policies.

    For each pooling limit p, first obtain the ideal minimum fulfilment cost and
    ideal maximum service. Then minimise the maximum of the two normalised regret
    terms over all store policies. Enumeration keeps the binary enablement decision
    exact while each inner problem remains a linear program.
    """
    regs=regions['Region'].tolist(); sts=stores['Store'].tolist(); fac=facilities.index.tolist(); n_f=len(fac); n_r=len(regs)
    rows=[]
    for p in [.30,.40,.50]:
        policies=[]
        for k in range(len(sts)+1):
            for combo in itertools.combinations(sts,k):
                # MINIMAX treats service as an objective, not as a hard target.
                # Use the full Model-1 economic cost (including enablement, lost-sale
                # and store-stockout costs) so the cost-vs-service frontier is managerial.
                z=solve_allocation(stores,regions,costs,facilities,list(combo),p,online_mult,store_mult,0.0,
                                   warehouse_disruption,promo_active,capacity_loss=capacity_loss,lost_sale_cost=lost_sale_cost,
                                   stockout_cost=stockout_cost,enablement_cost_mult=enablement_cost_mult)
                policies.append((list(combo),z))
        # Ideal values over the capacity-feasible policy set.
        min_cost=min(z['total_cost'] for _,z in policies)
        max_service=max(z['fill_rate'] for _,z in policies)
        cost_scale=max(abs(min_cost),1.0); service_scale=max(abs(max_service),1e-9)
        best=None
        for combo,z in policies:
            cost_reg=max(0.0,(z['total_cost']-min_cost)/cost_scale)
            service_reg=max(0.0,(max_service-z['fill_rate'])/service_scale)
            q=max(cost_reg,service_reg)
            if best is None or q < best['Q']-1e-12 or (abs(q-best['Q'])<1e-12 and z['total_cost']<best['result']['total_cost']):
                best={'Q':q,'enabled':combo,'result':z,'cost_regret':cost_reg,'service_regret':service_reg,
                      'ideal_fulfil_cost':min_cost,'ideal_fill_rate':max_service}
        rows.append({'Pooling':p,'Q':best['Q'],'Stores':', '.join(best['enabled']) if best['enabled'] else 'None',
                     'Fulfilment Cost':best['result']['fulfil_cost'],'Total Cost':best['result']['total_cost'],
                     'Fill Rate':best['result']['fill_rate'],'Cost Regret':best['cost_regret'],'Service Regret':best['service_regret'],
                     'Ideal Economic Cost':min_cost,'Ideal Fill Rate':max_service})
    return pd.DataFrame(rows)

def store_economics(stores, regions, costs, facilities, pooling, online_mult=1.0, store_mult=1.0,
                    hurdle=36, warehouse_disruption=False, promo_active=False, capacity_loss=0.15, stockout_cost=420.0, enablement_cost_mult=1.0):
    base = solve_allocation(stores, regions, costs, facilities, [], pooling, online_mult, store_mult, 0.0,
                            warehouse_disruption, promo_active, capacity_loss=capacity_loss, stockout_cost=stockout_cost, enablement_cost_mult=enablement_cost_mult)
    rows = []
    for _, s in stores.iterrows():
        z = solve_allocation(stores, regions, costs, facilities, [s['Store']], pooling, online_mult, store_mult, 0.0,
                             warehouse_disruption, promo_active, capacity_loss=capacity_loss, stockout_cost=stockout_cost, enablement_cost_mult=enablement_cost_mult)
        saving = max(0.0, base['fulfil_cost'] - z['fulfil_cost'])
        one_time = float(s['Enablement Lakh']) * 100000.0 * float(enablement_cost_mult)
        payback = one_time / saving if saving > 0 else np.inf
        rows.append({'Store': s['Store'], 'Local Demand': s['Local Demand'] * store_mult, 'Capacity': s['Capacity'],
                     'Enablement Cost (₹ lakh)': s['Enablement Lakh'], 'Monthly Saving': saving,
                     'Payback (months)': payback, 'Within hurdle': payback <= hurdle})
    return pd.DataFrame(rows)


def compare_strategies(stores, regions, costs, facilities, pooling, online_mult, store_mult, service_target,
                       warehouse_disruption=False, promo_active=False, capacity_loss=0.15, stockout_cost=420.0, enablement_cost_mult=1.0):
    optimal = best_enablement(stores, regions, costs, facilities, pooling, online_mult, store_mult, service_target,
                              warehouse_disruption, promo_active, capacity_loss=capacity_loss, stockout_cost=stockout_cost, enablement_cost_mult=enablement_cost_mult)
    all_stores = stores['Store'].tolist()
    configs = [('Centralised', []), ('Selective / model', optimal['enabled']), ('Fully pooled', all_stores)]
    rows = []
    for name, enabled in configs:
        z = solve_allocation(stores, regions, costs, facilities, enabled, pooling, online_mult, store_mult,
                             service_target, warehouse_disruption, promo_active, capacity_loss=capacity_loss, stockout_cost=stockout_cost, enablement_cost_mult=enablement_cost_mult)
        rows.append({'Strategy': name, 'Stores': ', '.join(enabled) if enabled else 'None',
                     'Total Cost': z['total_cost'], 'Fill Rate': z['fill_rate'], 'Lost Sales': z['lost'].sum()})
    return pd.DataFrame(rows), optimal


def pooling_sensitivity(stores, regions, costs, facilities, online_mult, store_mult, service_target,
                        warehouse_disruption=False, promo_active=False, capacity_loss=0.15, stockout_cost=420.0, enablement_cost_mult=1.0):
    rows = []
    for p in [0.30, 0.40, 0.50]:
        opt = best_enablement(stores, regions, costs, facilities, p, online_mult, store_mult, service_target,
                              warehouse_disruption, promo_active, capacity_loss=capacity_loss, stockout_cost=stockout_cost, enablement_cost_mult=enablement_cost_mult)
        z = opt['result']
        rows.append({'Pooling': p, 'Total Cost': z['total_cost'], 'Fill Rate': z['fill_rate'],
                     'Lost Sales': z['lost'].sum(), 'Stores': ', '.join(opt['enabled']) if opt['enabled'] else 'None'})
    return pd.DataFrame(rows)


def _solve_realized(stores, regions, costs, facilities, enabled, pooling, online_demand, local_demand,
                    warehouse_disruption=False, capacity_loss=0.15, lost_sale_cost=360.0):
    """Cost-minimising allocation for one realised Monte-Carlo trial with a fixed store policy."""
    regs = regions['Region'].tolist(); sts = stores['Store'].tolist(); fac = facilities.index.tolist()
    n_f, n_r = len(fac), len(regs); n = n_f * n_r + n_r
    def ix(i,j): return i*n_r+j
    c = np.zeros(n, dtype=float)
    for i,f in enumerate(fac):
        for j,reg in enumerate(regs): c[ix(i,j)] = float(costs.loc[f,reg])
    c[n_f*n_r:] = float(lost_sale_cost)
    A=[]; b=[]
    for i,f in enumerate(fac):
        cap=float(facilities.loc[f,'Capacity'])
        if f in sts:
            if f not in enabled: cap=0.0
            else:
                local_i=float(local_demand[sts.index(f)])
                cap=max(0.0,min(float(pooling)*cap, cap-local_i))
        elif warehouse_disruption:
            cap*=1.0-float(capacity_loss)
        row=np.zeros(n); row[[ix(i,j) for j in range(n_r)]]=1.0
        A.append(row); b.append(max(0.0,cap))
    E=[]; eb=[]
    for j in range(n_r):
        row=np.zeros(n); row[[ix(i,j) for i in range(n_f)]]=1.0; row[n_f*n_r+j]=1.0
        E.append(row); eb.append(float(online_demand[j]))
    res=linprog(c,A_ub=np.asarray(A),b_ub=np.asarray(b),A_eq=np.asarray(E),b_eq=np.asarray(eb),
                bounds=[(0,None)]*n,method='highs')
    if not res.success: raise RuntimeError('Realised Monte-Carlo allocation is infeasible.')
    x=np.maximum(res.x[:n_f*n_r].reshape(n_f,n_r),0.0); lost=np.maximum(res.x[n_f*n_r:],0.0)
    flow=pd.DataFrame(x,index=fac,columns=regs)
    return flow,pd.Series(lost,index=regs)

def simulate(stores, regions, costs, facilities, risk, enabled, pooling, online_mult=1.0, store_mult=1.0,
             promo_prob=None, disruption_prob=None, trials=1000, seed=42, force_promo=False,
             force_disruption=False):
    rng = np.random.default_rng(int(seed))
    regs = regions['Region'].tolist(); sts = stores['Store'].tolist()
    promo_prob = risk['promo_prob'] if promo_prob is None else float(promo_prob)
    disruption_prob = risk['disruption_prob'] if disruption_prob is None else float(disruption_prob)
    base_online = regions['Mean Demand'].to_numpy(float) * float(online_mult)
    base_local = stores['Local Demand'].to_numpy(float) * float(store_mult)
    out=[]
    for _ in range(int(trials)):
        od=np.maximum(rng.normal(base_online,base_online*risk['online_sd']),0.0)
        promo_event=bool(force_promo or rng.random()<promo_prob)
        if promo_event: od*=1.25
        ld=np.maximum(rng.normal(base_local,base_local*risk['store_sd']),0.0)
        local_spike_event=bool(rng.random()<risk['local_spike_prob'])
        if local_spike_event: ld[rng.integers(0,len(sts))]*=1.0+risk['local_spike_size']
        disrupted=bool(force_disruption or rng.random()<disruption_prob)
        flow,lost=_solve_realized(stores,regions,costs,facilities,enabled,pooling,od,ld,
                                  disrupted,risk['capacity_loss'],risk['LC'])
        fulfilled=float(od.sum()-lost.sum())
        fc=float(np.sum(flow.to_numpy()*np.array([[costs.loc[f,r] for r in regs] for f in flow.index],dtype=float)))
        return_rate=float(rng.choice(risk['return_rates'],p=risk['return_probs']))
        returned=fulfilled*return_rate
        resaleable=returned*risk['resaleable']
        markdown_units=max(0.0,returned-resaleable)
        excess=max(0.0,resaleable-lost.sum())
        store_stockout_units=0.0
        for i,f in enumerate(sts):
            cap=float(facilities.loc[f,'Capacity'])
            if f in enabled:
                online_used=float(flow.loc[f].sum())
                store_stockout_units += max(0.0,ld[i]+online_used-cap)
            else:
                store_stockout_units += max(0.0,ld[i]-cap)
        lost_cost=float(lost.sum()*risk['LC'])
        transfer_cost=float(returned*risk['TC'])
        excess_cost=float(excess*risk['HC'])
        markdown_cost=float(markdown_units*risk['MC'])
        total_cost=fc+lost_cost+transfer_cost+excess_cost+markdown_cost
        fill=1.0-lost.sum()/od.sum() if od.sum() else 1.0
        out.append([lost.sum(),fc,transfer_cost,excess,excess_cost,markdown_units,markdown_cost,
                    total_cost,fill,store_stockout_units,int(disrupted),int(promo_event),return_rate])
    return pd.DataFrame(out,columns=['Lost Sales','Fulfilment Cost','Transfer Cost','Excess Inventory',
        'Excess Holding Cost','Markdown Units','Markdown Cost','Total Cost','Fill Rate',
        'Store Stockout Units','Disrupted','Promotion','Return Rate'])

