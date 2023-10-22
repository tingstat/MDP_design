#### create value function #######
benchmark_data, transition_data=real_time_order_dispatch_algorithm() #generate the value function, make sure that the settings are the same


###########
simu = 200

dd1 = real_time_order_dispatch_algorithm_revised(1, NUM_EPISODES=5000)
dd0 = real_time_order_dispatch_algorithm_revised(0, NUM_EPISODES=5000)

ATE_emprical_true = np.mean( dd1['revenue'] )  - np.mean( dd0['revenue'] )


import time
import warnings
warnings.filterwarnings("ignore")

NUM_EPISODES=50
#### random ####
ATE_random = []

for i in range(simu):
    print(i)
    time_start = time.time()
    dd_temp = real_time_order_dispatch_algorithm_revised(0.5, NUM_EPISODES)
    
    eta1_temp, TD1_temp, beta1_temp = Q_eta_est_poly(dd_temp, 1)
    eta0_temp, TD0_temp, beta0_temp = Q_eta_est_poly(dd_temp, 0)
    
    ATE_random.append(eta1_temp- eta0_temp)
    print(time.time() - time_start)
    
    
#### half ####
ATE_half = []

for i in range(simu):
    print(i)
    time_start = time.time()
    dd_temp = real_time_order_dispatch_algorithm_revised(3, NUM_EPISODES)
    
    
    eta1_temp, TD1_temp, beta1_temp = Q_eta_est_poly(dd_temp, 1)
    eta0_temp, TD0_temp, beta0_temp = Q_eta_est_poly(dd_temp, 0)
    
    ATE_half.append(eta1_temp- eta0_temp)
    print(time.time() - time_start)
    
 #### epsilon_greedy ####
ATE_greedy = []

for i in range(simu):
    print(i)
    time_start = time.time()
    dd_temp = real_time_order_dispatch_algorithm_revised(6, NUM_EPISODES)
    
     
    eta1_temp, TD1_temp, beta1_temp = Q_eta_est_poly(dd_temp, 1)
    eta0_temp, TD0_temp, beta0_temp = Q_eta_est_poly(dd_temp, 0)
     
    ATE_greedy .append(eta1_temp- eta0_temp)
    print(time.time() - time_start)   
    
### TMDP  ###
ATE_TMDP = []

for i in range(simu):
    print(i)
    time_start = time.time()
    dd_temp = real_time_order_dispatch_algorithm_revised(4, NUM_EPISODES)
    
    
    eta1_temp, TD1_temp, beta1_temp = Q_eta_est_poly(dd_temp, 1)
    eta0_temp, TD0_temp, beta0_temp = Q_eta_est_poly(dd_temp, 0)
    
    ATE_TMDP.append(eta1_temp- eta0_temp)

    print(time.time() - time_start)
    
### NMDP   ###
ATE_NMDP=[]

for i in range(simu):
    print(i)
    time_start = time.time()
    dd_temp = real_time_order_dispatch_algorithm_revised(5, NUM_EPISODES)
    
    
    eta1_temp, TD1_temp, beta1_temp = Q_eta_est_poly(dd_temp, 1)
    eta0_temp, TD0_temp, beta0_temp = Q_eta_est_poly(dd_temp, 0)
    
    ATE_NMDP.append(eta1_temp- eta0_temp)

    print(time.time() - time_start)
    
################
# Save results #
################
ATEs_all = {'ATE_random':np.array(ATE_random), 
            'ATE_half':np.array(ATE_half),
            'ATE_greedy':np.array(ATE_greedy), 
            'ATE_MDP':np.array(ATE_TMDP),  
            'ATE_NMDP':np.array(ATE_NMDP) }
ATEs_MSE_all = ( pd.DataFrame( ATEs_all) - ATE_emprical_true )**2
np.mean(ATEs_MSE_all, axis=0)

ATEs_MSE_all.to_csv('ATEs_MSE_reward_epi_'+str(NUM_EPISODES)+'_vary_order_driver_all.csv', index=False, header=True)


