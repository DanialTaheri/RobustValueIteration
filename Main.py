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
import collections
from parser import parse_args
import math

StateType = Tuple[int, ...]
# DP algorithm

class InvEnv:
    def __init__(self, mu_price, args):
        self.L_max = args.L_max
        self.L_min = args.L_min
        self.epoch_disc_factor = args.epoch_disc_factor
        self.failure_cost = args.failure_cost
        self.condition_ini = args.condition_ini
        self.av_det = args.av_det
        self.adjust_rate = args.adjust_rate
        self.prod_cap = args.prod_cap
        self.len_month = args.len_month
        self.gran = args.gran
        self.mu_inflow = eval(args.inflow_mean)
        self.sigma_inflow = eval(args.inflow_std)
        self.sigma_price = eval(args.price_std)
        self.mu_price = mu_price
        self.model_type = args.model_type

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
        if self.av_det == 0:
            current_cond = list([0])
        else:
            current_cond = list(np.arange(self.condition_ini, 110, self.av_det))
        month = range(self.len_month)
        return set(product(
            *chain([on_hand_range], [current_cond], [month])
        ))


    def get_next_states_probs_rewards(
            self,
            state: StateType,
            action: int,
            inflow_probs:  Sequence[float],
            price_probs: Sequence[float], 
            model
    ) -> Mapping[StateType, Tuple[float, float]]:
        next_state_arr: ndarray = np.array(state)
        next_state_arr += np.append([-action *self.adjust_rate, self.av_det if action > 0. else\
                                     0],[1 if next_state_arr[2] < self.len_month-1 else\
                                     -(self.len_month-1)])
        #next_state_arr += np.append(-action *self.adjust_rate, self.av_det if action > 0 else 0.)

        # The next line represents state change due to demand
        temp_list = []

        for price, prob_pr in price_probs:
            #there is no need to create a cost for the failure since we can ensure that the plant never fails by considering 
            # the initial condition to be less than a threshold
            reward = action * price if next_state_arr[1] <= 100. else 0.
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
                ret[s] = (sum_p2, avg_r)
                

            elif model == "robust":
                r_list = [r for p1, _, r in tl if p1 != 0]

                if r_list:
                    worst_r = min(r_list)
                    #worst_r = sum((p1/len(inflow_probs) * worst_r 
                    #        for p1, _, _ in tl))/(sum_p1) if sum_p1 != 0. else 0.
                    
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
        return {s: {a: self.get_next_states_probs_rewards(s, a, ret_inflow[s[-1]], ret_price[s[-1]], model)
                    for a in np.arange(0, self.get_all_actions(s), self.gran)}
                for s in self.get_all_states()}

    def get_exogenous_state(self):
        # to project the inflow to the closest acceptable inflow representations 
        def find_closest_state(gran, initial_point):
            if gran - (initial_point % gran) < initial_point % gran:
                res = gran - (initial_point % gran)
            else:
                res = - (initial_point % gran)
            return initial_point + res + self.L_min
        # self.mu_price is a list
        rv_price = [0] * self.len_month
        rv_inflow = [0] * self.len_month
        raw_price_probs = [0] * self.len_month
        raw_inflow_probs = [0] * self.len_month
        pp_price = [0] * self.len_month
        pp_inflow = [0] * self.len_month
        ret_inflow = [[] for _ in range(self.len_month)]
        ret_price = [[] for _ in range(self.len_month)]
        self.gran_price = 1.
        if self.model_type == 'nominal':
            Start_cdf = 0.01
            End_cdf = 1.-Start_cdf
        else:
            Start_cdf = 0.2
            End_cdf = 1-Start_cdf
        for month in range(self.len_month):
            rv_price[month] = lognorm(s= self.sigma_price[month],\
                                      scale = math.exp(self.mu_price[month]))
            rv_inflow[month] = lognorm(s = self.sigma_inflow[month],\
                                       scale = math.exp(self.mu_inflow[month]))
            
            raw_price_probs[month] = \
            [rv_price[month].cdf(i) for i in np.arange(int(rv_price[month].ppf(Start_cdf)),\
                                                       int(rv_price[month].ppf(End_cdf)),\
                                                       self.gran_price)]
            # inflow: needs to project to the closest point feasible for the model
            Start_point_inflow = find_closest_state(self.gran, int(rv_inflow[month].ppf(Start_cdf)))
            raw_inflow_probs[month] = \
            [rv_inflow[month].cdf(i) for i in np.arange(Start_point_inflow,\
                                                       int(rv_inflow[month].ppf(End_cdf)),\
                                                        self.gran)]
            
            
            pp_price[month] = [p / sum(raw_price_probs[month]) for p in raw_price_probs[month]]

            pp_inflow[month] = [p / sum(raw_inflow_probs[month]) for p in\
                                raw_inflow_probs[month]]
            
            
            for i, num in enumerate(np.arange(int(rv_price[month].ppf(Start_cdf)),\
                                              int(rv_price[month].ppf(End_cdf)), self.gran_price)):
                ret_price[month].append([num, pp_price[month][i]])        
            for i, num in enumerate(np.arange(Start_point_inflow,\
                                              int(rv_inflow[month].ppf(End_cdf)), self.gran)):
                ret_inflow[month].append([num, pp_inflow[month][i]])
                
        return {month: ret_price[month] for month in range(self.len_month)}, {month: ret_inflow[month] for month in range(self.len_month)}
        
            
            

     
    def get_all_actions(self, state):
        if state[1] >= 100:
            cap = self.gran
        else:
            cap = max(min(self.prod_cap, (state[0] - self.L_min)* (1/self.adjust_rate)), self.gran)
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
    args = parse_args()
    print(args)
    if args.price_category == '1':
        mu_price = [2.549615, 2.501999, 2.431512, 2.354458, 2.288651, 2.249327, 2.245656,\
                    2.278644, 2.340898, 2.418321, 2.493343, 2.548944, 2.572555]
    elif args.price_category == '2':
        mu_price = [3.105615, 3.057999, 2.987512, 2.910458, 2.844651, 2.805327, 2.801656,\
                    2.834644, 2.896898, 2.974321, 3.049343, 3.104944, 3.128555]
    elif args.price_category == '3':
        mu_price = [3.515115, 3.467499, 3.397012, 3.319958, 3.254151, 3.214827, 3.211156,\
                    3.244144, 3.306398, 3.383821, 3.458843, 3.514444, 3.538055]
    elif args.price_category == '4':
        mu_price = [3.885615, 3.837999, 3.767512, 3.690458, 3.624651, 3.585327, 3.581656, \
                    3.614644, 3.676898, 3.754321, 3.829343, 3.884944, 3.908555]
    elif args.price_category == '5':
        mu_price = [4.445615, 4.397999, 4.327512, 4.250458, 4.184651, 4.145327, 4.141656,\
                    4.174644, 4.236898, 4.314321, 4.389343, 4.444944, 4.468555]
    else:
        raise ValueError("Please enter an acceptable price category")


    inv = InvEnv(mu_price, args)
    States = inv.get_all_states()
    
    if not inv.validate_spec():
        raise ValueError
    mdp_ref_obj = inv.get_mdp_refined(model = args.model_type)
    dp_obj = DPNumeric(mdp_ref_obj, args.this_tolerance)
    #dp_obj_robust = DPNumeric(mdp_ref_obj_robust, this_tolerance)
    
    if args.model_type == "nominal":
        print("calculating the value iteration")
        opt_policy, value_function = dp_obj.get_optimal_policy_vi()
    else:
        opt_policy, value_function = dp_obj.get_optimal_policy_Robust_vi()        

    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path)
    results = args.experiment_path + "VF-model%s-category%s-capacity%s.txt"%(args.model_type, args.price_category, args.prod_cap)
    print("results", results)
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
    
    # creating the table
    table = collections.defaultdict(list)
    for state in States:
        
        table[int(state[-1])].append([int (state[0]), int(state[1]), value_function[state]])
        
    Table_filename = args.experiment_path + "Table-model%s-category%s-capacity%s.json"%(args.model_type ,args.price_category, args.prod_cap)
    with open(Table_filename, "w") as f:
        json.dump(table, f)
        




    