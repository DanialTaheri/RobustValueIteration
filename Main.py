from typing import Tuple, NamedTuple, Set, Mapping, Sequence
from itertools import chain, product, groupby
import numpy as np
from numpy.core.multiarray import ndarray
from scipy.stats import lognorm
from processes.mdp_refined import MDPRefined
from func_approx.dnn_spec import DNNSpec
from func_approx.func_approx_base import FuncApproxBase
from func_approx_spec import FuncApproxSpec
from copy import deepcopy
from operator import itemgetter
from processes.det_policy import DetPolicy
from dp.dp_analytic import DPAnalytic
from dp.dp_numeric import DPNumeric
import json
import os
import FigureFunc

StateType = Tuple[int, ...]
# DP algorithm

class InvEnv(NamedTuple):
    L_max: float
    L_min: float
    mu_inflow: Sequence[float]
    sigma_inflow: Sequence[float]
    mu_price: Sequence[float]
    sigma_price: Sequence[float]
    epoch_disc_factor: float
    failure_cost: float
    condition_ini: float
    av_det: float
    adjust_rate: float
    prod_cap: float
    len_month: int
    gran: float

    def validate_spec(self) -> bool:
        b1 = self.L_max > 0.
        b2 = self.L_min >= 0
        #b3 = self.mu_inflow > 0.
        #b4 = self.sigma_inflow >= 0.
        b5 = 0. <= self.epoch_disc_factor <= 1.
        #b6 = self.mu_price > 0.
        #b7 = self.sigma_price > 0.
        b8 = self.failure_cost > 0.
        return all([b1, b2, b5, b8])

    def get_all_states(self) -> Set[StateType]:
        # reservior level
        on_hand_range = list(np.arange(int(self.L_min), int(self.L_max + 1), self.gran))
        if self.av_det == 0.:
            current_cond = list([0.])
        else:
            current_cond = list(np.arange(self.condition_ini, 1.41, self.av_det))
        month = range(self.len_month)
        return set(product(
            *chain([on_hand_range], [current_cond], [month])
        ))

    # Order of operations in an epoch are:
    # 1) Order Placement (Action)
    # 2) Receipt
    # 3) Throwout Space-Limited-Excess Inventory
    # 4) Demand
    # 5) Adjust (Negative) Inventory to not fall below stockout limit

    # In the following func, the input "state" is represented by
    # the on-hand and on-order right before an order is placed (the very
    # first event in the epoch) and the "state"s in the output are represented
    # by the  on-hand and on-order just before the next order is placed (in the
    # next epoch).  Both the input and output "state"s are arrays of length (L+1).

    def get_next_states_probs_rewards(
            self,
            state: StateType,
            action: int,
            inflow_probs:  Sequence[float],
            price_probs: Sequence[float], 
            model
    ) -> Mapping[StateType, Tuple[float, float]]:
        next_state_arr: ndarray = np.array(state)
        #print("next_state_arr, action", next_state_arr, action)
        # The next line represents state change due to Action and Receipt[1 if next_state_arr[2] < 11 else -11]
        next_state_arr += np.append([-action *self.adjust_rate, self.av_det if action > 0. else 0.],
                                    [1 if next_state_arr[2] < self.len_month-1 else -(self.len_month-1)])
        #next_state_arr += np.append(-action *self.adjust_rate, self.av_det if action > 0 else 0.)

        # The next line represents state change due to demand
        temp_list = []

        for price, prob_pr in price_probs:
            #there is no need to create a cost for the failure since we can ensure that the plant never fails by considering 
            # the initial condition to be less than a threshold
            reward = action * price if next_state_arr[1] < 1. else 0.
            #- (self.failure_cost if next_state_arr[1] > 1. else 0.)  
            
            for inflow, prob_in in inflow_probs:
                #print("inflow : {}, next_state_arr[0] : {}".format(inflow, next_state_arr[0]))
                next_state_arr[0] = min(inflow+next_state_arr[0], self.L_max)
                ns = deepcopy(next_state_arr)
                inv = ns[0]
                cond = ns[1]
                onhand = inv
                ns_tup = tuple(x for x in ns)
                temp_list.append((ns_tup, prob_pr, prob_in, reward))

        ret = {}
        crit = itemgetter(0)
        for s, v in groupby(sorted(temp_list, key=crit), key=crit):
            tl = [(p1, p2, r) for _, p1, p2, r in v]
            sum_p1 = sum(p1/len(inflow_probs) for p1, _, _ in tl)
            
            sum_p2 = sum(p2/len(price_probs) for _, p2, _ in tl)
            #print(s, sum_p2)
            if model == "nominal":
                avg_r = sum((p1/len(inflow_probs) * r 
                            for p1, _, r in tl))/(sum_p1) if sum_p1 != 0. else 0.
                #print("sum of probabilities",sum((p2/len(price_probs) for _, p2, _ in tl))/(sum_p2) if sum_p2 != 0. else 0.)
                #print("action, avg_r", action, avg_r)
                ret[s] = (sum_p2, avg_r)
                
#                 print("state:", s)
#                 print("action: {}, reward : {}".format(action, ret[s][1]))
            elif model == "robust":
                r_list = [r for p1, _, r in tl if p1 != 0]
#                 print(s)
#                 print(action, r_list)
                if r_list:
                    worst_r = min(r_list)
                    worst_r = sum((p1/len(inflow_probs) * worst_r 
                            for p1, _, _ in tl))/(sum_p1) if sum_p1 != 0. else 0.
                    
                else:
                    worst_r = 0.

                ret[s] = (sum_p2, worst_r)
            else:
                raise ValueError("Model is not selected appropriately")
                
        return ret

    def get_mdp_refined_dict(self, model) \
            -> Mapping[StateType,
                       Mapping[int,
                               Mapping[StateType,
                                       Tuple[float, float]]]]:
        ret_price, ret_inflow = self.get_exogenous_state()
        #print("pp_inflow", pp_inflow)
        return {s: {a: self.get_next_states_probs_rewards(s, a, ret_inflow[s[-1]], ret_price[s[-1]], model)
                    for a in np.arange(0, self.get_all_actions(s), self.gran)}
                for s in self.get_all_states()}

    def get_exogenous_state(self):
        # self.mu_price is a list
        rv_price = [0] * self.len_month
        rv_inflow = [0] * self.len_month
        raw_price_probs = [0] * self.len_month
        raw_inflow_probs = [0] * self.len_month
        pp_price = [0] * self.len_month
        pp_inflow = [0] * self.len_month
        ret_inflow = [[] for _ in range(self.len_month)]
        ret_price = [[] for _ in range(self.len_month)]
        for month in range(self.len_month):
            rv_price[month] = lognorm(s= self.sigma_price[month], scale = self.mu_price[month])
            rv_inflow[month] = lognorm(s = self.sigma_inflow[month], scale = self.mu_inflow[month])
            raw_price_probs[month] = [rv_price[month].cdf(i) for i in np.arange(0, int(rv_price[month].ppf(0.999)), self.gran)]
            raw_inflow_probs[month] = [rv_inflow[month].cdf(i) for i in np.arange(0, int(rv_inflow[month].ppf(0.999)), self.gran)]
            pp_price[month] = [p / sum(raw_price_probs[month]) for p in raw_price_probs[month]]
            pp_inflow[month] = [p / sum(raw_inflow_probs[month]) for p in raw_inflow_probs[month]]
            for i, num in enumerate(np.arange(0, int(rv_price[month].ppf(0.999)), self.gran)):
                ret_price[month].append([num, pp_price[month][i]])
            for i, num in enumerate(np.arange(0, int(rv_inflow[month].ppf(0.999)), self.gran)):
                ret_inflow[month].append([num, pp_inflow[month][i]])
            #print(ret_inflow[month])
                
        #print({month: ret_inflow[month] for month in range(self.len_month)})
        return {month: ret_price[month] for month in range(self.len_month)}, {month: ret_inflow[month] for month in range(self.len_month)}
        
            
            

    # Actions given the inventory 
    def get_all_actions(self, state):
        if state[1] >= 1.:
            cap = self.gran
        else:
            cap = max(min(self.prod_cap, (state[0] - self.L_min)* (1/self.adjust_rate)), self.gran)
#             if state[0] > 90:
#                 print("cap", cap)
        return cap
        
        
        
    def get_mdp_refined(self, model) -> MDPRefined:
        return MDPRefined(self.get_mdp_refined_dict(model), self.epoch_disc_factor)

    def get_optimal_policy(self) -> DetPolicy:
        return self.get_mdp_refined().get_optimal_policy()

    def get_ips_orders_dict(self) -> Mapping[int, Sequence[int]]:
        sa_pairs = self.get_optimal_policy().get_state_to_action_map().items()

        def crit(x: Tuple[Tuple[int, ...], int]) -> int:
            return sum(x[0])

        return {ip: [y for _, y in v] for ip, v in
                groupby(sorted(sa_pairs, key=crit), key=crit)}


if __name__ == '__main__':
    import time
    start = time.time()
        #initial parameters
    kwargs = {"L_max":500, "L_min":1. ,"mu_inflow":[10., 10., 10., 15., 15., 15., 20., 20., 20., 15., 15, 15.]  , "sigma_inflow":[3.] * 12,
           "mu_price": [30., 30., 30., 20., 20., 20., 10., 10., 10., 20., 20., 20.] * 12,"sigma_price": [4.] *12, "epoch_disc_factor":0.95, "gran": 0.5,
           "failure_cost":10., "condition_ini":0., "av_det": 0.05,"adjust_rate": 1, "prod_cap": 20., "len_month":12}

#    kwargs = {"L_max":50., "L_min":40. ,"mu_inflow":[1., 1., 1., 1.5, 1.5, 1.5, 2., 2., 2., 1.5, 1.5, 1.5]  , "sigma_inflow":[0.5] * 12,
#          "mu_price": [3., 3., 3., 2., 2., 2., 1., 1., 1., 2., 2., 2.] * 12,"sigma_price": [0.5] *12, "epoch_disc_factor":0.95, "gran": 0.5,
#          "failure_cost":10., "condition_ini":0., "av_det": 0.,"adjust_rate": 1, "prod_cap": 20., "len_month":12}



    # month and reservior level
#     kwargs = {"L_max":200., "L_min":40. ,"mu_inflow":[1., 1., 1., 1.5, 1.5, 1.5, 2., 2., 2., 1.5, 1.5, 1.5]  , "sigma_inflow":[0.5] * 12,
#           "mu_price": [3., 3., 3., 2., 2., 2., 1., 1., 1., 2., 2., 2.] * 12,"sigma_price": [0.5] *12, "epoch_disc_factor":0.95, "gran": 0.5,
#           "failure_cost":10., "condition_ini":0., "av_det": 0.,"adjust_rate": 1, "prod_cap": 20., "len_month":12}

    # reservior level and condition of plant
#     kwargs = {"L_max":20., "L_min":1. ,"mu_inflow":[3.2, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2, 5., 5., 5.]  , "sigma_inflow":[1.] * 12,
#           "mu_price": [2.] * 12,"sigma_price": [0.5] *12, "epoch_disc_factor":0.96, 
#           "failure_cost":10., "condition_ini":0., "av_det": 0.2,"adjust_rate": 1, "prod_cap": 5., "len_month":1}



#     resservior level
#     kwargs = {"L_max":100., "L_min":1. ,"mu_inflow":[1., 1., 1., 2., 2., 2., 3., 3., 3., 2., 2., 2.]  , "sigma_inflow":[0.5] * 12,
#            "mu_price": [2.] * 12,"sigma_price": [0.5] *12, "epoch_disc_factor":0.95, "gran": 1.,
#           "failure_cost":10., "condition_ini":0., "av_det": 0.,"adjust_rate":1,"prod_cap": 10.,
#           "len_month":1}

    inv = InvEnv(**kwargs)
    States = inv.get_all_states()
    
    if not inv.validate_spec():
        raise ValueError
    mdp_ref_obj_nominal = inv.get_mdp_refined(model = "nominal")
    #mdp_ref_obj_robust = inv.get_mdp_refined(model = "robust")
    this_tolerance = 1e-2
    dp_obj_nominal = DPNumeric(mdp_ref_obj_nominal, this_tolerance)
    #dp_obj_robust = DPNumeric(mdp_ref_obj_robust, this_tolerance)
    
    def criter(x: Tuple[Tuple[int, ...], int]) -> int:
        return sum(x[0])

    opt_policy, value_function = dp_obj_nominal.get_optimal_policy_vi()
    #opt_policy_robust, value_function_robust = dp_obj_robust.get_optimal_policy_Robust_vi()        
    experiment_path = "Results/"
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
    if not os.path.exists(experiment_path + "/DP/"):
        os.makedirs(experiment_path + "/DP/")
    config_path = experiment_path + "/DP/config.json"
    with open(config_path, "w") as f:
        json.dump(kwargs, f)
    results = experiment_path + "/DP/VF_Pol.txt"
    output= open(results, 'wt')
    for s in States:
        output.write("States: {}, Optimal value function: {}\n".format(s, value_function[s]))
        #print("States: {}, Optimal value function: {}\n".format(s, value_function[s]))
    #output.write("#######################Policy########################")
    #output.write("Policy: {}\n".format(opt_policy))
    end = time.time()
    execution_time = end - start
    output.write("Execution time: {}".format(execution_time))
    output.close()



#     Data = {}
#     import matplotlib
#     import matplotlib.pyplot as plt
#     month = []
#     res_level = []
#     for s in States:
#         if s[2] == 7:
#             Data[s[0]] = value_function[s]
    
#     for key in Data.keys():
#         res_level.append(key)
#     res_level = sorted(set(res_level))
#     values = np.zeros(len(res_level))
#     for i, elem1 in enumerate(res_level):
#         values[i] = Data[elem1]
#     der_value = np.zeros(len(res_level)-1)
#     der_res = np.zeros(len(res_level)-1)
#     for num in range(1, len(res_level)):
#         der_value[num-1] = (values[num] - values[num-1]) /kwargs['gran']
#         der_res[num-1] = res_level[num] 
    
#     #res_level and derivative
#     fig = plt.figure()
#     fig.subplots_adjust()
#     ax1 = fig.add_subplot(111)
#     ax1.set_xlabel("reservior level")
#     ax1.set_title("Value Iteration")
#     ax1.set_xlim(kwargs['L_min'], kwargs['L_max'])
#     ax1.plot(der_res, der_value, label = "Der. of VF with granularity %s"%kwargs['gran'])
#     #ax1.plot(res_level, values, label = "Value function")
#     ax1.legend()
#     plt.show()    
#     policyplot(opt_policy, States, kwargs["prod_cap"])
#     policyplot(opt_policy_robust, States, kwargs["prod_cap"])
    