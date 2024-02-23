#####################################
#Assignment part 1

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import math
from scipy.optimize import minimize
import scipy.stats as sp
import plots

np.random.seed(200) #23
#####################################
#Figure 2.1
def kalman_filter(df,sigma_sq_eta,sigma_sq_eps):
    number_obs = len(df)
    
    a = np.zeros(number_obs)
    v = np.zeros(number_obs)
    F = np.zeros(number_obs)
    K = np.zeros(number_obs)
    P = np.zeros(number_obs)

    a[0] = 0
    P[0] = 10**7
    
    for i in range(number_obs):
        v[i] = df[i] - a[i]
        F[i] = P[i] + sigma_sq_eps
        K[i] = P[i]/F[i]
    
        if(i != number_obs - 1):
            a[i + 1] = a[i] + K[i]*v[i]
            P[i + 1] = K[i]*sigma_sq_eps + sigma_sq_eta
       
    return a,v,F,K,P

#####################################
#figure 2.2
def kalman_smoother(df,a,v,F,K,P):
    number_obs = len(df)
    
    r = np.zeros(number_obs)
    N = np.zeros(number_obs)
    alpha = np.zeros(number_obs)
    V = np.zeros(number_obs)
    
    r[-1] = 0
    N[-1] = 0
    
    for i in reversed(range(number_obs)):
        r[i-1] = v[i]/F[i] + (1 - K[i])*r[i]
        alpha[i] = a[i] + P[i]*r[i-1]
        N[i-1] = 1/F[i] + (1 - K[i])**2 * N[i]
        V[i] = P[i] - ((P[i]**2) * N[i-1])
    return alpha,V,r,N  

#####################################
#figure 2.3
def kalman_obs_disturb(df,sigma_sq_eta,sigma_sq_eps,F,v,K,r,N):
    u = (F)**(-1) * v - K*r
    epsilon_hat = sigma_sq_eps*u
    
    D = (F)**(-1) + K**2 * N
    smoothed_sd_eps = (sigma_sq_eps - (sigma_sq_eps**2) * D)**(0.5)  
    
    eta_hat = sigma_sq_eta*r
    smoothed_sd_eta = (sigma_sq_eta - (sigma_sq_eta**2) * N)**(0.5)
    return epsilon_hat, smoothed_sd_eps, eta_hat, smoothed_sd_eta,u,D
    
#####################################
#figure 2.4

def simulation(df,sigma_sq_eta,sigma_sq_eps,epsilon_hat,eta_hat,epsilon_plus,eta_plus):
    number_obs = len(df)

    alpha_plus = np.zeros(number_obs)
    y_plus = np.zeros(number_obs)
    alpha_plus[0] = df[0]

    for i in range(number_obs):
        y_plus[i] = alpha_plus[i] + epsilon_plus[i]
        if(i != number_obs-1):
            alpha_plus[i+1] = alpha_plus[i] + eta_plus[i]
      
    a_hat,v_hat,F_hat,K_hat,P_hat  = kalman_filter(y_plus,sigma_sq_eta,sigma_sq_eps)
    alpha_hat,V_hat,r_hat,N_hat =   kalman_smoother(y_plus,a_hat,v_hat,F_hat,K_hat,P_hat)
    #epsilon_plus_hat, smoothed_sd_eps, eta_hat, smoothed_sd_eta,u,D =  kalman_obs_disturb(df,sigma_sq_eta,sigma_sq_eps,F_hat,v_hat,K_hat,r_hat,N_hat)
    
    u_hat = (F_hat)**(-1) * v_hat - K_hat*r_hat
    epsilon_hat_plus = sigma_sq_eps*u_hat
    
    epsilon_tilde = epsilon_plus - epsilon_hat_plus + epsilon_hat
    
    alpha_tilde = df - epsilon_tilde
    eta_tilde = [alpha_tilde[i+1] - alpha_tilde[i] for i in range(len(alpha_tilde)-1)]
    return(alpha_hat,alpha_plus,alpha_tilde,epsilon_hat,epsilon_tilde,eta_hat,eta_tilde)

#####################################
#figure 2.5
def kalman_weight(df_nile,sigma_sq_eta,sigma_sq_eps,break_begin1,break_end1,break_begin2,break_end2):
    number_obs = len(df_nile)
    broken_df = df_nile.copy()
    broken_nile = broken_df['Nile']
    broken_nile[break_begin1:break_end1] = np.nan    
    broken_nile[break_begin2:break_end2] = np.nan
    
    new_a = np.zeros(number_obs+1)
    new_v = np.zeros(number_obs)
    new_F = np.zeros(number_obs)
    new_K = np.zeros(number_obs)
    new_P = np.zeros(number_obs+1)
    
    new_a[0] = 0
    new_P[0] = 10**7
    
    for i in range(number_obs):
        new_v[i] = broken_nile[i] - new_a[i]
        new_F[i] = new_P[i] + sigma_sq_eps
        new_K[i] = new_P[i]/new_F[i]
        if(math.isnan(broken_nile[i])):
            new_a[i + 1] = new_a[i] 
            new_P[i + 1] = new_P[i] + sigma_sq_eta
        else:
            new_a[i + 1] = new_a[i] + new_K[i]*new_v[i]
            new_P[i + 1] = new_K[i]*sigma_sq_eps + sigma_sq_eta
          
    new_r = np.zeros(number_obs)
    new_N = np.zeros(number_obs)
    new_alpha = np.zeros(number_obs)
    new_V = np.zeros(number_obs)
    
    new_r[number_obs-1] = 0
    new_N[number_obs-1] = 0
    
    for i in reversed(range(number_obs)):
        if(math.isnan(broken_nile[i])):
            new_r[i-1] = new_r[i]
            new_alpha[i] = new_a[i] + new_P[i]*new_r[i-1]
            new_N[i-1] = new_N[i]
            new_V[i] = new_P[i] - ((new_P[i]**2) * new_N[i-1])
            
        else:
            new_r[i-1] = new_v[i]/new_F[i] + (1 - new_K[i])*new_r[i]
            new_alpha[i] = new_a[i] + new_P[i]*new_r[i-1]
            new_N[i-1] = 1/new_F[i] + (1 - new_K[i])**2 * new_N[i]
            new_V[i] = new_P[i] - ((new_P[i]**2) * new_N[i-1])
                  
    return broken_nile,new_a[1:],new_P[1:],new_alpha,new_V


#####################################
#figure 2.6
def forecasting(df,a,P,F,K,sigma_sq_eps,sigma_sq_eta):
    number_obs = len(df)
    time_start = int(df['Year'][0])
    forecast_end_time = 2000
    forecasting_time = np.arange(time_start,forecast_end_time+1,1)
    a_forecast = a
    P_forecast = P
    F_forecast = F

    end_range = forecast_end_time - int(df['Year'][number_obs-1]) + number_obs

    for i in range(number_obs, end_range):
        a_forecast = np.append(a_forecast,a_forecast[i-1] )
        P_forecast = np.append(P_forecast,P_forecast[i-1]+sigma_sq_eta)
        F_forecast = np.append(F_forecast,P_forecast[i-1]+sigma_sq_eps)
    
    forecasts = a_forecast[number_obs:]
    z_50 = 0.6745  # For 50% confidence interval
    lower_bound = forecasts - z_50 * np.sqrt(P_forecast[number_obs:])
    upper_bound = forecasts + z_50 * np.sqrt(P_forecast[number_obs:])

    return forecasting_time,a_forecast,P_forecast,F_forecast,lower_bound,upper_bound
  
#####################################
#figure 2.7
def standardised_errors(df,v,F):
    e = v/np.sqrt(F)
    return e

#####################################
#figure 2.8
def stand_smoothed_res(u,D,r,N):
    u_star = D**(-0.5)*u
    r_star = N**(-0.5)*r
    return u_star, r_star

#####################################
#Parameter estimation

def loglikilihood(parameters,y):
    sigma_sq_eps,sigma_sq_eta = parameters
    number_obs = len(y)
    a,v,F_star,K,P_star = kalman_filter(y,sigma_sq_eta,sigma_sq_eps)

    log_L = - (number_obs - 1)/2 * np.log(2*np.pi)
    for i in range(1,number_obs):
        log_L += -0.5*(np.log(F_star[i])+(v[i]**2/F_star[i]))
    
    return -log_L

def est_params(y):
    initial_values = [16000,1000]
    result = minimize(loglikilihood, initial_values, args=(y,), method='L-BFGS-B')
    return result

def main():
    df_nile = pd.read_excel("Data/Nile.xlsx")
    df_nile.rename(columns={'Unnamed: 0': 'Year'}, inplace=True)
    number_obs = len(df_nile)

    sigma_sq_eps = 15099
    sigma_sq_eta = 1469.1
    
    epsilon_plus =   np.random.normal(0, sigma_sq_eps**(0.5), number_obs)
    eta_plus = np.random.normal(0, sigma_sq_eta**(0.5), number_obs)
    
    break_begin1 = df_nile.loc[df_nile['Year']==1891].index[0]
    break_end1 = df_nile.loc[df_nile['Year']==1911].index[0]
    break_begin2 = df_nile.loc[df_nile['Year']==1931].index[0]
    break_end2 = df_nile.loc[df_nile['Year']==1951].index[0]
    
    #figure 1
    a,v,F,K,P = kalman_filter(df_nile['Nile'],sigma_sq_eta,sigma_sq_eps)
    plots.plot_kalman_filter(df_nile,a,v,F,K,P)
    
    #figure 2
    alpha,V,r,N  = kalman_smoother(df_nile,a,v,F,K,P)
    plots.plot_kalman_smoother(df_nile,alpha,V,r,N)
    
    #figure 3
    epsilon_hat, smoothed_sd_eps, eta_hat, smoothed_sd_eta,u,D = kalman_obs_disturb(df_nile,sigma_sq_eta,sigma_sq_eps,F,v,K,r,N)
    plots.plot_kalman_obs_dist(df_nile,epsilon_hat, smoothed_sd_eps, eta_hat, smoothed_sd_eta)
    
    #figure 4
    alpha_hat,alpha_plus,alpha_tilde,epsilon_hat,epsilon_tilde,eta_hat,eta_tilde = simulation(df_nile['Nile'],sigma_sq_eta,sigma_sq_eps,epsilon_hat,eta_hat,epsilon_plus,eta_plus)
    plots.plot_simulation(df_nile,alpha,alpha_plus,alpha_tilde,epsilon_hat,epsilon_tilde,eta_hat,eta_tilde)

    #figure 5
    broken_nile,new_a,new_P,new_alpha,new_V = kalman_weight(df_nile,sigma_sq_eta,sigma_sq_eps,break_begin1,break_end1,break_begin2,break_end2)
    plots.plot_kalman_weights(df_nile,broken_nile,new_a,new_P,new_alpha,new_V,break_begin1,break_end1,break_begin2,break_end2)

    #figure 6
    forecasting_time,a_forecast,P_forecast,F_forecast,lower_bound,upper_bound = forecasting(df_nile,a,P,F,K,sigma_sq_eps,sigma_sq_eta)
    plots.plot_forecasting(df_nile,forecasting_time,a_forecast,P_forecast,F_forecast,lower_bound,upper_bound)
    
    #figure 7
    e = standardised_errors(df_nile,v,F)
    plots.plot_standardised_errors(df_nile,e)
    
    #figure 8
    u_star, r_star = stand_smoothed_res(u,D,r,N)
    plots.plot_stand_smoothed_res(df_nile,u_star,r_star)
    
    res = est_params(df_nile['Nile'])
    print(res)
    print(res.x[0],res.x[1])
###########################################################
### call main
if __name__ == "__main__":
    main()



