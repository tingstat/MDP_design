from itertools import count
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.optimize import linear_sum_assignment 
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
import pandas as pd
import statsmodels.api as sm
from localreg import *
import random


NUMBER_OF_GRID_TILES_X = 9
NUMBER_OF_GRID_TILES_Y= 9
NUMBER_OF_TIME_STEPS = 20 

MAX_MANHATTAN_DISTANCE = 2

def manhattan_distance(p1,p2):
    return np.abs(p1[0]-p2[0]) + np.abs(p1[1]-p2[1])

################## generate waiting time ####################
NUMBER_OF_ORDERS = 100

LOWER_BOUND_WAITING_TIME = 0
UPPER_BOUND_WAITING_TIME = 5
MEAN_WAITING_TIME = 2.5
STANDARD_DEVIATION_WAITING_TIME = 2

waiting_time_sampler = stats.truncnorm(
    (LOWER_BOUND_WAITING_TIME - MEAN_WAITING_TIME) / STANDARD_DEVIATION_WAITING_TIME, (UPPER_BOUND_WAITING_TIME - MEAN_WAITING_TIME) / STANDARD_DEVIATION_WAITING_TIME, loc=MEAN_WAITING_TIME, scale=STANDARD_DEVIATION_WAITING_TIME)

waiting_times = waiting_time_sampler.rvs(NUMBER_OF_ORDERS)

### generate samples from mixture distribution #############
PROBABILITY_FIRST_GAUSSIAN = 1./3
PROBABILITY_SECOND_GAUSSIAN = 2./3

MEAN_FIRST_GAUSSIAN = [3,3,2] 
MEAN_SECOND_GAUSSIAN = [6,6,15]

STANDARD_DEVIATION_FIRST_GAUSSIAN = [2,2,3]
STANDARD_DEVIATION_SECOND_GAUSSIAN = [2,2,3]

LOWER_LIMITS_BY_DIMENSION = [0 for _ in range(3)]
UPPER_LIMITS_BY_DIMENSION = [NUMBER_OF_GRID_TILES_X - 1, NUMBER_OF_GRID_TILES_Y - 1, NUMBER_OF_TIME_STEPS - 1]

class TruncatedMultivariateNormalInteger():
    def __init__(self, normals):
        self._normals = []
        for [lower, upper, mean, standard_deviation] in normals:
            X = stats.truncnorm(
    (lower - mean) / standard_deviation, (upper - mean) / standard_deviation, loc=mean, scale=standard_deviation)
            self._normals.append(X)
    # size equals 3 (e.g., 3 independent truncated normals per mixture component) in our example
    def rvs(self, size):
        return np.array([[normal.rvs(size=1) for normal in self._normals] for _ in range(size)])

class MixtureModel():
    def __init__(self, submodels, weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.submodels = submodels
        self.weights = weights/np.sum(weights) 

    def rvs(self, size):
        rvs = [] 
        for i in range(size):
            random_model = np.random.choice(range(len(self.submodels)), p=self.weights)
            rvs.append(self.submodels[random_model].rvs(size=1))
        return np.round(np.squeeze(np.array(rvs))).astype(int)
    
first_truncated_multivariate_normal = TruncatedMultivariateNormalInteger([[LOWER_LIMITS_BY_DIMENSION[i], UPPER_LIMITS_BY_DIMENSION[i], MEAN_FIRST_GAUSSIAN[i], STANDARD_DEVIATION_FIRST_GAUSSIAN[i]] for i in range(3)])

second_truncated_multivariate_normal = TruncatedMultivariateNormalInteger([[LOWER_LIMITS_BY_DIMENSION[i], UPPER_LIMITS_BY_DIMENSION[i], MEAN_SECOND_GAUSSIAN[i], STANDARD_DEVIATION_SECOND_GAUSSIAN[i]] for i in range(3)])

mixture_gaussian_model = MixtureModel([first_truncated_multivariate_normal, second_truncated_multivariate_normal],[1./3,2./3])

############# destinations and initial locations of orders #####
def spawn_uniformly_x_y_location():
    return [np.random.choice(range(NUMBER_OF_GRID_TILES_X)), np.random.choice(range(NUMBER_OF_GRID_TILES_Y))]

########### discounted reward #################################
DISCOUNT_FACTOR = 0.9

def discounted_reward_mdp(gamma, T, R):
    total_gamma = 0
    discounted_gamma = 1
    for _ in range(T):
        total_gamma += discounted_gamma 
        discounted_gamma *= gamma
    return total_gamma * R / T


############ policy evaluation ################################
# Elements in state_transactions are quadruples state, action, reward, next_state
# action is a quadruple consisting of [idle = 0 / serve = 1, serving position [x,y], destination posotion [x,y]]
# V and N are 9*9*20 (3-d) matrices
# delta_t corresponds to the serving time, calculated based on the Manhanttan distance
time_ = 2 # The last component of state
def policy_evaluation(state_transactions, V, N, starting_index, method):
    if V is None:
        V = np.zeros(np.array(UPPER_LIMITS_BY_DIMENSION) - np.array(LOWER_LIMITS_BY_DIMENSION) + [1,1,1])
    if N is None:
        N = np.zeros(np.array(UPPER_LIMITS_BY_DIMENSION) - np.array(LOWER_LIMITS_BY_DIMENSION) + [1,1,1])
    for t in range(NUMBER_OF_TIME_STEPS, -1, -1):
        for state, action, reward, next_state in state_transactions[starting_index:]:
            if state[time_] == t:
                N[tuple(state)] += 1
                delta_t = 1
                if action[0] == 1:
                    delta_t += manhattan_distance(state[:2], action[1]) + manhattan_distance(action[1], action[2])    
                future_value = 0
                if next_state[time_] < NUMBER_OF_TIME_STEPS:
                    future_value = np.power(DISCOUNT_FACTOR, delta_t) * V[tuple(next_state)]
                if method == 'mdp':
                    modified_reward = discounted_reward_mdp(DISCOUNT_FACTOR, delta_t ,reward)
                elif method == 'myopic':
                    modified_reward = reward
                V[tuple(state)] += 1./(N[tuple(state)]) * (future_value + modified_reward - V[tuple(state)])
    return V, N

ratios_of_served_orders = []

################ Order dispatch ###############################
BASE_REWARD_PER_TRIP = 1
REWARD_FOR_DISTANCE_PARAMETER = 1

def real_time_order_dispatch_algorithm():

    NUM_EPISODES = 5000 
    BENCHMARK_RUNS = 50 
    NUM_INDEPENDENT_RUNS = NUM_EPISODES - BENCHMARK_RUNS 

    # Consider three settings where the number of drivers equals 25, 50 and 75, respectively. Recall that the total number of orders is given by 100.
    number_of_drivers_list = [50] #[25,50,75]
    # Consider three methods, distance (closest driver policy), myopic policy (gamma=0) and MDP policy
    method_list = ['mdp'] #['distance', 'myopic', 'mdp']
    # Consider two measurements, the completion rate and the average distance between orders and drivers
    measurement_keypoints = ['Total Revenue', 'ratio served', 'average distance to driver']

    stored_mdp_V_functions = [] 
    
    # Measures the performance for each algorithm
    benchmark_data = np.zeros((len(number_of_drivers_list), len(method_list), len(measurement_keypoints),BENCHMARK_RUNS))

    for number_of_drivers_ind, number_of_drivers in enumerate(number_of_drivers_list):
        for method_ind, method in enumerate(method_list):
            transition_data = []
            if method in ['mdp', 'myopic']:
                ## Initialize the value and the state counter
                V, N = policy_evaluation(transition_data, None, None, 0, method)
                starting_index = 0

            for episode in range(NUM_EPISODES): 
                
                order_driver_distances = []
                revenue_all= []
                number_of_aviable_drivers =[]
                number_of_call_orders=[]
                
                number_of_drivers = int(20) #int( random.uniform(25, 50) ) #int( random.uniform(25, 30) )
                

                # Generate 450 episodes as historical data and then 50 episodes to evaluate difference policies
                if episode >= NUM_INDEPENDENT_RUNS and method in ['mdp', 'myopic']:
                    V, N = policy_evaluation(transition_data, V, N, starting_index, method)
                    starting_index = len(transition_data)  # original transition

                destinations = []
                for _ in range(NUMBER_OF_ORDERS):
                    # destination is drawn uniformly randomly from the grid
                    destinations.append(spawn_uniformly_x_y_location())
                
                # in orders first entry is boolean corresponding to wether it is served.
                orders = list(map(list, zip([False] * NUMBER_OF_ORDERS, mixture_gaussian_model.rvs(NUMBER_OF_ORDERS), np.round(waiting_times).astype(int), destinations, range(NUMBER_OF_ORDERS))))
                drivers = []
                for i in range(number_of_drivers):
                    # first entry describes the first time the driver is available again
                    # driver's location is drawn uniformly randomly from the grid
                    drivers.append([0, spawn_uniformly_x_y_location(), i])
                    
                for t in range(NUMBER_OF_TIME_STEPS):
                    # obtain active orders
                    active_orders = [order for order in orders if (order[0] == False) and (order[1][2] <= t) and (order[1][2] + order[2] >= t)]
                    available_drivers = [driver for driver in drivers if driver[0] <= t]
                    # print(len(active_orders), len(available_drivers)) 
                    # print(drivers)
                    
                    #### generate state variables   ########
                    number_of_aviable_drivers.append( len(active_orders) )
                    number_of_call_orders.append( len(available_drivers) )
                    #########################################

                    allowed_match = np.ones((len(active_orders), len(available_drivers)), dtype=bool)
                    for order_count, active_order in enumerate(active_orders):
                        for driver_count, available_driver in enumerate(available_drivers):
                            # only consider drivers whose manhattan distance is slower than 2
                            if manhattan_distance(available_driver[1], active_order[1][:2]) > MAX_MANHATTAN_DISTANCE:
                                allowed_match[order_count, driver_count] = False
                    # print(allowed_match)

                    # computation of advantage function
                    if method in ['mdp', 'myopic']:
                        #Could also initialize with - infinity.
                        advantage_function = np.zeros((len(active_orders), len(available_drivers))) 
                        for order_count, active_order in enumerate(active_orders):
                            for driver_count, available_driver in enumerate(available_drivers):
                                if(allowed_match[order_count, driver_count]):
                                    # the pickup time
                                    delta_t = 1 + manhattan_distance(available_driver[1], active_order[1][:2]) + manhattan_distance(active_order[1][:2], active_order[3])    
                                    reward = BASE_REWARD_PER_TRIP + REWARD_FOR_DISTANCE_PARAMETER  * manhattan_distance(active_order[1][:2], active_order[3])  
                                    #If the completion time is later than the last time step, we just stop set the future value to zero
                                    future_value = 0.
                                    if t + delta_t < NUMBER_OF_TIME_STEPS: 
                                        discount = DISCOUNT_FACTOR
                                        if method == 'myopic':
                                            discount = 0.
                                        future_value = np.power(discount, delta_t) * V[active_order[3][0],active_order[3][1], t + delta_t]
                                    current_value = V[available_driver[1][0], available_driver[1][1], t]
                                    modified_reward = reward
                                    if method == 'mdp':
                                        modified_reward = discounted_reward_mdp(DISCOUNT_FACTOR, delta_t, reward)
                                    advantage_function[order_count, driver_count] = future_value - current_value + modified_reward
                        
                        # plot_sample_histogram(np.array([active_order[1] for active_order in active_orders]), "Active Orders at time {}".format(t))
                        # plot_sample_histogram(np.array([available_driver[1] for available_driver in available_drivers]), "Available Drivers at time {}".format(t))
                        
                        # print(advantage_function)
                        # print(advantage_function.shape)
                        
                        # Matchs orders to drivers. Note the important subtelty that the function linear_sum_assignment returns a full matching. But we are happy with a partial matching already.

                    row_ind = [] 
                    col_ind = []

                    # The initial independent runs should use the 'distance' policy to find the matching.
                    # Later runs could either use 'mdp', 'myopic' or 'distance' policy
                    if episode >= NUM_INDEPENDENT_RUNS and method in ['mdp','myopic']:
                        penalized_advantage_matrix = advantage_function
                        for i in range(len(active_orders)):
                            for j in range(len(available_drivers)):
                                if not allowed_match[i,j]:
                                    penalized_advantage_matrix[i,j] = - 100 * NUMBER_OF_ORDERS
                        row_ind, col_ind = linear_sum_assignment(-penalized_advantage_matrix)

                    else:
                        #Use distance matrix to compute assignment
                        distance_matrix = -np.ones((len(active_orders), len(available_drivers))) * 100 * NUMBER_OF_ORDERS
                        for i in range(len(active_orders)):
                            for j in range(len(available_drivers)):
                                if allowed_match[i,j]:
                                    distance_matrix[i,j] = -manhattan_distance(available_drivers[j][1], active_orders[i][1][:2]) 
                        row_ind, col_ind = linear_sum_assignment(-distance_matrix) 

                    matched_order_ind = []
                    matched_driver_ind = []

                    for i in range(len(row_ind)):
                        if row_ind[i] < len(active_orders) and col_ind[i] < len(available_drivers) and allowed_match[row_ind[i],col_ind[i]]:
                            matched_order_ind.append(row_ind[i])
                            matched_driver_ind.append(col_ind[i])

                    # print(f"Matched orders in iteration {t}")
                    revenue_temp = 0
                    for i in range(len(matched_order_ind)):
                        if allowed_match[matched_order_ind[i]][matched_driver_ind[i]]:
                            matched_order = active_orders[matched_order_ind[i]]
                            matched_driver = available_drivers[matched_driver_ind[i]]
                            # if method == 'mdp':
                                # print(f'Order {matched_order[-1]} at {matched_order[1][:2]} is matched to driver {matched_driver[-1]} at {matched_driver[1]}')
                            matched_order[0] = True

                            order_driver_distance = manhattan_distance(matched_driver[1], matched_order[1][:2])

                            # continue to run the code only when the assertion is satisfied. Stop and return an error otherwise
                            assert(order_driver_distance <= 2)

                            order_driver_distances.append(order_driver_distance)

                            delta_t = 1 + manhattan_distance(matched_driver[1], matched_order[1][:2]) + manhattan_distance(matched_order[1][:2], matched_order[3])    
                            matched_driver[0] = t + delta_t 

                            # update the matched driver's location
                            matched_driver[1]=matched_order[3]

                            #Append to transition data.
                            
                            reward = BASE_REWARD_PER_TRIP + REWARD_FOR_DISTANCE_PARAMETER  * manhattan_distance(matched_order[1][:2], matched_order[3])  
                            revenue_temp = revenue_temp + manhattan_distance(matched_order[1][:2], matched_order[3])
                            
                            transition = [[matched_driver[1][0], matched_driver[1][1], t], [1, matched_order[1][:2], matched_order[3]], reward, [matched_order[3][0], matched_order[3][1], t + delta_t]]
                            transition_data.append(transition.copy())
                    
                    ##### generate rewards  #####
                    revenue_all.append( revenue_temp)

                    # Set transition data for unmatched drivers 
                    for i, unmatched_driver in enumerate(available_drivers):
                        if i not in matched_driver_ind:
                            transition = [[unmatched_driver[1][0], unmatched_driver[1][1], t],[0],0,[unmatched_driver[1][0], unmatched_driver[1][1], t + 1]]
                            transition_data.append(transition.copy())

                if episode >= NUM_INDEPENDENT_RUNS: 
                    number_of_served_orders = 0
                    for i in range(len(orders)):
                        number_of_served_orders += orders[i][0]
                        
                    # calculate the completion rate
                    ratio_served = float(number_of_served_orders)/ NUMBER_OF_ORDERS
                    benchmark_data[number_of_drivers_ind, method_ind, :,  episode - NUM_INDEPENDENT_RUNS] = [np.sum(np.array(revenue_all)), ratio_served, np.mean(np.array(order_driver_distances))]
                    if method == 'mdp' and episode == NUM_EPISODES - 1:
                        # Used for visualising value functions
                        stored_mdp_V_functions.append(V.copy()) 
                        np.savez('Value_function_vary_order_driver_50.npz', V.copy())
                        

    #print(benchmark_data) 
    #plot_benchmarks(benchmark_data, number_of_drivers_list, method_list, measurement_keypoints) 
    #plot_value_functions(stored_mdp_V_functions)
    return benchmark_data, transition_data
      

####################################################################################
Value = np.load('Value_function_vary_order_driver_50.npz') # load value function
V = Value['arr_0']

BASE_REWARD_PER_TRIP = 1
REWARD_FOR_DISTANCE_PARAMETER = 1

def real_time_order_dispatch_algorithm_revised( allocation, NUM_EPISODES=500):

    #BENCHMARK_RUNS = 50 
    #NUM_INDEPENDENT_RUNS = NUM_EPISODES - BENCHMARK_RUNS 

    # Consider three settings where the number of drivers equals 25, 50 and 75, respectively. Recall that the total number of orders is given by 100.
    #number_of_drivers_list = [25,50,75]
    # Consider three methods, distance (closest driver policy), myopic policy (gamma=0) and MDP policy
    
    # Consider two measurements, the completion rate and the average distance between orders and drivers
    #measurement_keypoints = ['Total Revenue', 'ratio served', 'average distance to driver']

    method_list = ['distance', 'mdp', 'myopic']
    #method_list = [0, 1]

    transition_data = []
    
    data_all = []
    
    prob_or=0.5

    for episode in range(NUM_EPISODES): 
        
        #order_driver_distances = []
        #revenue_all= []
        #number_of_aviable_drivers =[]
        #number_of_call_orders=[]
        #A=[]
        #probs = []
        number_of_drivers =int(50)  #int( random.uniform(25, 50) ) #int( random.uniform(25, 30) )

        destinations = []
        for _ in range(NUMBER_OF_ORDERS):
            # destination is drawn uniformly randomly from the grid
            destinations.append(spawn_uniformly_x_y_location())
        
        # in orders first entry is boolean corresponding to wether it is served.
        orders = list(map(list, zip([False] * NUMBER_OF_ORDERS, mixture_gaussian_model.rvs(NUMBER_OF_ORDERS), np.round(waiting_times).astype(int), destinations, range(NUMBER_OF_ORDERS))))
        drivers = []
        for i in range(number_of_drivers):
            # first entry describes the first time the driver is available again
            # driver's location is drawn uniformly randomly from the grid
            drivers.append([0, spawn_uniformly_x_y_location(), i])
        
        
        active_orders_next = [order for order in orders if (order[0] == False) and (order[1][2] <= 0) and (order[1][2] + order[2] >= 0)]
        available_drivers_next = [driver for driver in drivers if driver[0] <= 0]
        
        for t in range(NUMBER_OF_TIME_STEPS):
            # obtain active orders
            
            active_orders = active_orders_next
            available_drivers = available_drivers_next
            # print(len(active_orders), len(available_drivers)) 
            # print(drivers)
            
            #### generate state variables   ########
            #number_of_aviable_drivers.append( len(active_orders) )
            #number_of_call_orders.append( len(available_drivers) )
            #########################################

            allowed_match = np.ones((len(active_orders), len(available_drivers)), dtype=bool)
            for order_count, active_order in enumerate(active_orders):
                for driver_count, available_driver in enumerate(available_drivers):
                    # only consider drivers whose manhattan distance is slower than 2
                    if manhattan_distance(available_driver[1], active_order[1][:2]) > MAX_MANHATTAN_DISTANCE:
                        allowed_match[order_count, driver_count] = False
            # print(allowed_match)
            
            ###### allocation methods ######
            if allocation==3:
                prob=allocation
                ind = episode < NUM_EPISODES/2  # fixed
                method = method_list[int(ind)]  
            elif allocation==2:
                prob=allocation
                ind = (episode*NUMBER_OF_TIME_STEPS+t)%2  #alternative
                method = method_list[int(ind)]
            elif allocation==4: # TMDP method
           
                if episode > int( NUM_EPISODES/2 ) and t==0:
                    data_for_update =pd.DataFrame( np.vstack((data_all)) )
                    data_for_update.columns=['n','T', 'orders', 'drivers', 'A', 'Prob', 'revenue', 'ordersNext', 'driversNext']
                    
                    eta1, TD_error1, beta1 = Q_eta_est_poly(data_for_update, 1)
                    eta0, TD_error0, beta0 = Q_eta_est_poly(data_for_update, 0) 
                    
                    #####################
                    prob = np.sqrt( np.mean( TD_error1**2 ) )/(  np.sqrt( np.mean( TD_error1**2 ) ) + np.sqrt( np.mean( TD_error0**2 ) )  )
                    
                    ######
                    prob_or = prob
                    
                    ind = np.random.binomial(1, prob, 1) # 0.5 for random
                    method = method_list[int(ind)]             
                elif episode > int( NUM_EPISODES/2 ) and t>0: 
                    #prob=1
                     #generate ind randmly (TMDP does not need this step, just for try, is inferior to following t=0)
                    method = method_list[int(ind)] # follow the method of t=0
                else:
                    prob=0.5
                    ind = episode < NUM_EPISODES/4  # fixed
                    method = method_list[int(ind)] 
                    
            elif allocation==5: # NMDP method
           
                if episode > int( NUM_EPISODES/2 ) and t==0:
                    data_for_update =pd.DataFrame( np.vstack((data_all)) )
                    data_for_update.columns=['n','T', 'orders', 'drivers', 'A', 'Prob', 'revenue', 'ordersNext', 'driversNext']
                    
                    TD_error1, beta_1, Q_value1 = Q_est(data_for_update, 1)
                    TD_error0, beta_0, Q_value0 = Q_est(data_for_update, 0)
                    
                    #####################
                    S = np.array( data_for_update[data_for_update['T']==0][['orders', 'drivers']] )
                    Next_S = np.array( data_for_update[data_for_update['T']==0][['ordersNext', 'driversNext']] )
                    A =data_for_update[data_for_update['T']==0]['A'].values

                    pre_S = np.array( [[len(active_orders), len(available_drivers)]])                    
                    #####################
                    prob, sigma_1, sigma_0 = Sigma_S_est(S, Next_S, A, TD_error1, TD_error0, pre_S, prob_or)
                    prob_or = prob
                    
                    ind = np.random.binomial(1, prob, 1) # 0.5 for random
                    method = method_list[int(ind)]             
                elif episode > int( NUM_EPISODES/2 ) and t>0:
                    if ind==1:
                        prob=1
                    else:
                        prob=0
                    method = method_list[int(ind)] 
                else:
                    if t==0:
                        prob=0.5
                    else:
                        prob = float( ind )
                    ind = episode < NUM_EPISODES/4  # fixed
                    method = method_list[int(ind)] 
            elif allocation==6:  # epsilon greedy
                if episode > int( NUM_EPISODES/2 ):
                    data_for_update =pd.DataFrame( np.vstack((data_all)) )
                    data_for_update.columns=['n','T', 'orders', 'drivers', 'A', 'Prob', 'revenue', 'ordersNext', 'driversNext']
                    
                    eta1, TD_error1, beta1 = Q_eta_est_poly(data_for_update, 1)
                    eta0, TD_error0, beta0 = Q_eta_est_poly(data_for_update, 0) 
                    
                    #####################
                    S = np.array( data_for_update[['orders', 'drivers']] )
                    Next_S = np.array( data_for_update[['ordersNext', 'driversNext']] )
                    A =data_for_update['A'].values

                    pre_S = np.array( [[len(active_orders), len(available_drivers)]])                    
                    #####################
                    pre_S_basis = phi_basis(pre_S)
                    Q1 = pre_S_basis.dot(beta1)
                    Q0 = pre_S_basis.dot(beta0)
                    epsilon = 0.1
                    
                    greedy_prob = np.random.uniform()
                    
                    if greedy_prob <epsilon:
                        prob=0.5
                        ind = np.random.binomial(1, prob, 1) # 0.5 for random
                        method = method_list[int(ind)]     
                    else:
                        prob=1
                        ind = 1*(Q1 >=Q0) + 0*(Q1<Q0)
                        method = method_list[int(ind)] 
        
            
                    prob_or = prob
                    
                            
                else:
                    prob=0.5
                    ind = np.random.binomial(1, prob, 1)  
                    method = method_list[int(ind)]  
                    
                    #prob=3
                    #ind = episode < NUM_EPISODES/4  # fixed
                    #method = method_list[int(ind)] 
                
            else:
                prob=allocation
                ind = np.random.binomial(1, prob, 1) # 0.5 for random
                method = method_list[int(ind)]    
            
            #A.append(method)
            #probs.append(prob)
            #################################

            # computation of advantage function
            if method in ['mdp', 'myopic']:
                #Could also initialize with - infinity.
                advantage_function = np.zeros((len(active_orders), len(available_drivers))) 
                for order_count, active_order in enumerate(active_orders):
                    for driver_count, available_driver in enumerate(available_drivers):
                        if(allowed_match[order_count, driver_count]):
                            # the pickup time
                            delta_t = 1 + manhattan_distance(available_driver[1], active_order[1][:2]) + manhattan_distance(active_order[1][:2], active_order[3])    
                            reward = BASE_REWARD_PER_TRIP + REWARD_FOR_DISTANCE_PARAMETER  * manhattan_distance(active_order[1][:2], active_order[3])  
                            #If the completion time is later than the last time step, we just stop set the future value to zero
                            future_value = 0.
                            if t + delta_t < NUMBER_OF_TIME_STEPS: 
                                discount = DISCOUNT_FACTOR
                                if method == 'myopic':
                                    discount = 0.
                                future_value = np.power(discount, delta_t) * V[active_order[3][0],active_order[3][1], t + delta_t]
                            current_value = V[available_driver[1][0], available_driver[1][1], t]
                            modified_reward = reward
                            if method == 'mdp':
                                modified_reward = discounted_reward_mdp(DISCOUNT_FACTOR, delta_t, reward)
                            advantage_function[order_count, driver_count] = future_value - current_value + modified_reward
                
                # plot_sample_histogram(np.array([active_order[1] for active_order in active_orders]), "Active Orders at time {}".format(t))
                # plot_sample_histogram(np.array([available_driver[1] for available_driver in available_drivers]), "Available Drivers at time {}".format(t))
                
                # print(advantage_function)
                # print(advantage_function.shape)
                
                # Matchs orders to drivers. Note the important subtelty that the function linear_sum_assignment returns a full matching. But we are happy with a partial matching already.

            row_ind = [] 
            col_ind = []

            # The initial independent runs should use the 'distance' policy to find the matching.
            # Later runs could either use 'mdp', 'myopic' or 'distance' policy
            if method in ['mdp','myopic']:
                penalized_advantage_matrix = advantage_function
                for i in range(len(active_orders)):
                    for j in range(len(available_drivers)):
                        if not allowed_match[i,j]:
                            penalized_advantage_matrix[i,j] = - 100 * NUMBER_OF_ORDERS
                row_ind, col_ind = linear_sum_assignment(-penalized_advantage_matrix)

            else:
                #Use distance matrix to compute assignment
                distance_matrix = -np.ones((len(active_orders), len(available_drivers))) * 100 * NUMBER_OF_ORDERS
                for i in range(len(active_orders)):
                    for j in range(len(available_drivers)):
                        if allowed_match[i,j]:
                            distance_matrix[i,j] = -manhattan_distance(available_drivers[j][1], active_orders[i][1][:2]) 
                row_ind, col_ind = linear_sum_assignment(-distance_matrix) 

            matched_order_ind = []
            matched_driver_ind = []

            for i in range(len(row_ind)):
                if row_ind[i] < len(active_orders) and col_ind[i] < len(available_drivers) and allowed_match[row_ind[i],col_ind[i]]:
                    matched_order_ind.append(row_ind[i])
                    matched_driver_ind.append(col_ind[i])

            # print(f"Matched orders in iteration {t}")
            revenue_temp = 0
            for i in range(len(matched_order_ind)):
                if allowed_match[matched_order_ind[i]][matched_driver_ind[i]]:
                    matched_order = active_orders[matched_order_ind[i]]
                    matched_driver = available_drivers[matched_driver_ind[i]]
                    # if method == 'mdp':
                        # print(f'Order {matched_order[-1]} at {matched_order[1][:2]} is matched to driver {matched_driver[-1]} at {matched_driver[1]}')
                    matched_order[0] = True

                    order_driver_distance = manhattan_distance(matched_driver[1], matched_order[1][:2])

                    # continue to run the code only when the assertion is satisfied. Stop and return an error otherwise
                    assert(order_driver_distance <= 2)

                    #order_driver_distances.append(order_driver_distance)

                    delta_t = 1 + manhattan_distance(matched_driver[1], matched_order[1][:2]) + manhattan_distance(matched_order[1][:2], matched_order[3])    
                    matched_driver[0] = t + delta_t 

                    # update the matched driver's location
                    matched_driver[1]=matched_order[3]

                    #Append to transition data.
                    
                    reward = BASE_REWARD_PER_TRIP + REWARD_FOR_DISTANCE_PARAMETER  * manhattan_distance(matched_order[1][:2], matched_order[3])  
                    revenue_temp = revenue_temp + manhattan_distance(matched_order[1][:2], matched_order[3])
                    
                    transition = [[matched_driver[1][0], matched_driver[1][1], t], [1, matched_order[1][:2], matched_order[3]], reward, [matched_order[3][0], matched_order[3][1], t + delta_t]]
                    transition_data.append(transition.copy())
            
            ##### generate rewards  #####
            #revenue_all.append( revenue_temp)

            # Set transition data for unmatched drivers 
            for i, unmatched_driver in enumerate(available_drivers):
                if i not in matched_driver_ind:
                    transition = [[unmatched_driver[1][0], unmatched_driver[1][1], t],[0],0,[unmatched_driver[1][0], unmatched_driver[1][1], t + 1]]
                    transition_data.append(transition.copy())
                    
            ############################
            active_orders_next = [order for order in orders if (order[0] == False) and (order[1][2] <= t+1) and (order[1][2] + order[2] >= t+1)]
            available_drivers_next = [driver for driver in drivers if driver[0] <= t+1]
            
            ###########################
            data_temp = [[episode, t, len(active_orders), len(available_drivers),int(ind), float(prob), revenue_temp,  len(active_orders_next), len(available_drivers_next) ]]
                
            
            data_all.append(data_temp)
    
         
    data_final =pd.DataFrame( np.vstack((data_all)) )
    data_final.columns=['n','T', 'orders', 'drivers', 'A', 'Prob', 'revenue', 'ordersNext', 'driversNext']
        
        
    return data_final

############################ Estimation ####################
def phi_basis(X):
    nx = np.shape(X)[1]
    
    phi_vector = []
    
    for i in range(nx):
        phi_vector.append( [ X[:, i], X[:, i]**2, X[:, i]**3 ] )
        
    phi_vector  = np.vstack((phi_vector)).T
    
    return phi_vector

def Q_est(data,treatment): ## estimation for NMDP
    
    data= data[data['A']==treatment]
    
    N = len( np.unique( data['n'].values ) )
    T = len( np.unique( data['T'].values ) )
    ##
    revenue = np.array( data['revenue'].values).reshape((N, T))
    cumulative_revenue = np.sum(revenue, axis=1)
    
    S = np.array( data[data['T']==0][['orders','drivers']] )
    
   
    phi_S =phi_basis(S)
    
    phi_S_with_intercept = np.hstack((np.ones( N ).reshape(-1,1),phi_S))
    
    
    beta_a = np.linalg.inv( phi_S_with_intercept.T.dot(phi_S_with_intercept)  + np.identity(np.shape(phi_S_with_intercept)[1])*1e-5 ).dot(phi_S_with_intercept.T).dot(cumulative_revenue) #
    
    Q_value = phi_S_with_intercept.dot(beta_a)
    
    TD_error = ( cumulative_revenue - Q_value )**2
    
    return  TD_error, beta_a, Q_value

def Q_est_TMDP(data,treatment, prob_s1):
    
    data['probS1']=prob_s1
    
    data= data[data['A']==treatment]
    
    N = len( np.unique( data['n'].values ) )
    T = len( np.unique( data['T'].values ) )
    ##
    revenue = np.array( data['revenue'].values).reshape((N, T))
    cumulative_revenue = np.repeat( np.sum(revenue, axis=1), T).reshape(N, T) + revenue - np.cumsum(revenue,1)
    
    S = np.array( data[['orders','drivers']] )
    
    time_h = data['T'].values.reshape(-1,1)
    indicator_morning = np.array( time_h==2 )+0
    indicator_night = np.array( time_h==15 )+0
    
    indicator_last_time = 1 - np.array( time_h==(T-1) )
    
   
    phi_S =phi_basis(S)
    
    phi_S_with_intercept = np.hstack((np.ones( N*T ).reshape(-1,1), time_h, indicator_morning, indicator_night ,phi_S))
    
    inverse_design = np.linalg.inv( phi_S_with_intercept.T.dot(phi_S_with_intercept)  + np.identity(np.shape(phi_S_with_intercept)[1])*1e-5 ).dot(phi_S_with_intercept.T)
    
    beta_a = inverse_design.dot(cumulative_revenue.reshape(-1,1)) #
    
    
    Next_all_covariates = indicator_last_time*np.vstack( (np.delete( phi_S_with_intercept, 0,axis=0), np.zeros(np.shape(phi_S_with_intercept)[1]).reshape(1,-1)) )
    
    Q_value_difference = (Next_all_covariates - phi_S_with_intercept).dot(beta_a)
    
    TD_error = revenue.reshape(-1,1) - Q_value_difference 
    
    density_ratio = phi_S_with_intercept.dot(inverse_design).dot(data['probS1'])
    
    density_ratio  = np.array(density_ratio < 0)*(1/T) + np.array(density_ratio > 0)*density_ratio 
    
    TD_error_density = (TD_error**2)*density_ratio.reshape(-1,1)
    
    TD_error_final = np.sum( TD_error_density.reshape(N, T), axis=1)
    
    return  TD_error_final, beta_a, Q_value_difference, density_ratio 

def Q_eta_est_poly(data, treatment):

    data_left = data[data['A']==treatment]
    
    #action = data_left['A'].values
    revenue = data_left['revenue'].values
    
    S = np.array( data_left[['orders','drivers']] )
    Next_S = np.array( data_left[['ordersNext','driversNext']] )
    
    phi_S = phi_basis(S)
    phi_next_S = phi_basis(Next_S)
    
    revenue_c =  revenue - np.mean(revenue)
    phi_S_c = phi_S  - np.mean(phi_S , axis=0)  
    phi_next_S_c = phi_next_S - np.mean(phi_next_S, axis=0)
    
    diff_phi_S_c = phi_S_c - phi_next_S_c
    
    #diff_phi_S_c[:,0] = diff_phi_S_c[:,0]+1
    
    beta_a = np.linalg.inv( diff_phi_S_c.T.dot(phi_S_c)  ).dot(phi_S_c.T).dot(revenue_c) #+ np.identity(np.shape(diff_phi_S_c)[1])*1e-5
    
    Q_diff_vec = diff_phi_S_c.dot(beta_a )
    
    eta_est = np.mean( revenue - Q_diff_vec )
    TD_error = revenue - Q_diff_vec  - eta_est
    
    return eta_est, TD_error, beta_a



def normfun(x, mu, sigma):

    pdf = np.exp(-((x - mu)**2)/(2*sigma**2)) / (sigma * np.sqrt(2*np.pi))

    return pdf

def Sigma_S_est(S, Next_S, A, TD_error1, TD_error0, pre_S, prob_or):
    
    nS = np.shape(S)[1]
    
    density_s_all = []
    density_s =1
    
    S_for_1_all = S[np.array(np.where(A==1))[0,:], :]
    S_for_0_all = S[np.array(np.where(A==0))[0,:], :]

    
            
    ###### calculate sigmas ####################
    sigma_1 = np.sqrt( TD_error1 )
    sigma_0 = np.sqrt( TD_error0 )
    
    if pre_S=='fitted':
        np_sigma_1 = localreg(S_for_1_all, np.array(sigma_1), x0=S, radius=6 )
        np_sigma_0 = localreg(S_for_0_all, np.array(sigma_0), x0=S,  radius=6 )
            
        prob_update = np_sigma_1/(np_sigma_1+np_sigma_0)
            
        invalid_probs = np.array(prob_update <0)+np.array(prob_update > 1) +0
            
        prob_update = prob_update*(1 - invalid_probs) + invalid_probs*prob_or
    else:
        np_sigma_1 = localreg(S_for_1_all, np.array(sigma_1), x0= np.array(pre_S).reshape(1,-1), radius=2 )
        np_sigma_0 = localreg(S_for_0_all, np.array(sigma_0), x0= np.array(pre_S).reshape(1,-1), radius=2 )
            
        prob_update = prob_or
        
        if np_sigma_1>0 and np_sigma_0>0:
            prob_update = np_sigma_1/(np_sigma_1+np_sigma_0)
               
    return prob_update, sigma_1, sigma_0 
        
        
    
    
    
    
    
    
    


