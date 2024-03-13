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
import random

np.random.seed(200) #23
#####################################
#Figure 2.1
def kalman_filter(df,d,Z,H,T,R,Q,forecast):
    # important to note the at+1 and Pt+1 are also calculated, since in a plot of smoothed necessary
    number_obs = len(df)
    df = np.array(df)
    a = np.zeros(number_obs+1)
    v = np.zeros(number_obs)
    F = np.zeros(number_obs)
    K = np.zeros(number_obs)
    P = np.zeros(number_obs+1)
    #np.random.normal(0, Q/(1-T**2), 1)
    a[0] = 0
    P[0] = Q/(1-T**2)
    for i in range(number_obs):
        v[i] = df[i] - Z * a[i] - d
        F[i] = Z * P[i] * Z + H
        K[i] = T*P[i]*Z/F[i]
        if(math.isnan(df[i])):
            a[i + 1] = T*a[i] 
            P[i + 1] = P[i] + Q
        else:
            a[i + 1] = T*a[i] + K[i]*v[i]
            #P[i + 1] = K[i]*H + Q
            P[i + 1] = T* P[i] *T + R*Q*R - K[i] * F[i] * K[i]
    if forecast == True:
        return a,v,F,K,P
    else:
        return a[:-1],v,F,K,P[:-1]


#####################################
#figure 2.2 

def kalman_smoother(df,a,v,F,K,P):
    df = np.array(df)
    number_obs = len(df)

    r = np.zeros(number_obs)
    N = np.zeros(number_obs)
    alpha = np.zeros(number_obs)
    V = np.zeros(number_obs)
    r[-1] = 0
    N[-1] = 0
    for i in reversed(range(number_obs)):
        if(math.isnan(df[i])):
            r[i-1] = r[i]
            alpha[i] = a[i] + P[i]*r[i-1]
            N[i-1] = N[i]
            V[i] = P[i] - ((P[i]**2) * N[i-1])
        else:
            r[i-1] = v[i]/F[i] + (1 - K[i])*r[i]
            alpha[i] = a[i] + P[i]*r[i-1]
            N[i-1] = 1/F[i] + (1 - K[i])**2 * N[i]
            V[i] = P[i] - ((P[i]**2) * N[i-1])
    return alpha,V,r,N

def transformation(log_ret):
    mu = np.mean(log_ret)
    yt = np.log((log_ret - mu )**2)
    return yt

#Parameter estimation

def loglikilihood(parameters,y):
    kappa,T,Q = parameters
    number_obs = len(y)
    Z = 1
    R = 1
    H = (np.pi**2)/2
    a,v,F_star,K,P_star = kalman_filter(y,kappa,Z,H,T,R,Q,False)
    log_L = - (number_obs - 1)/2 * np.log(2*np.pi)
    for i in range(1,number_obs):
        log_L += -0.5*(np.log(F_star[i])+(v[i]**2/F_star[i]))
    return -log_L

def est_params(x):
    kappa = np.mean(x)
    phi = 0.9950 #estimate from DK book as intial value
    sigma_sq_eta = (1-phi**2)*(np.var(x)-(np.pi**2)/2)
    initial_values = [kappa,phi,sigma_sq_eta]
    print('initial_values',initial_values)
    bounds = [(None, None), (0.000000001, 0.999999999), (0.000000001, None)]
    result = minimize(loglikilihood, initial_values, args=(x,), method='L-BFGS-B', bounds=bounds)
    return result
    
def calc_return(p):
    return np.log(p[1:])-np.log(p[:-1])

def est_beta(X_star,F,v):
    part1 = 0
    part2 = 0
    for i in range(len(X_star)):
        part1 += X_star[i] * (1/F[i]) * X_star[i]
        part2 += X_star[i] * (1/F[i]) * v[i]
    return (1/part1)*part2

def bootstrap(yt,mu,phi,sigma_sq,Q,M):
    n = len(yt)
    va = np.zeros(n)
    vP = np.zeros(n)

    alpha_tilde = sp.norm.rvs(0, np.sqrt(Q / (1 - phi**2)), size=M)
    w = np.exp(-0.5*np.log(2*np.pi*sigma_sq) - 0.5*alpha_tilde - (1/(2*sigma_sq))*np.exp(-1*alpha_tilde)*(yt[0]-mu)**2)
    w_norm = w/sum(w) # to normalise weights
    va[0] = sum(w_norm*alpha_tilde)  #MC step given M simulations
    vP[0] = sum(w_norm*alpha_tilde**2) - va[0]**2  #MC step given M simulations
    alpha_resample = np.random.choice(alpha_tilde, size=M, replace=True, p=w_norm)
    
    for i in range(1,n):
        alpha_tilde = phi * alpha_resample + sp.norm.rvs(0, np.sqrt(Q), size = M)
        w = np.exp(-0.5*np.log(2*np.pi*sigma_sq) - 0.5*alpha_tilde - (1/(2*sigma_sq))*np.exp(-1*alpha_tilde)*(yt[i]-mu)**2)
        w_norm = w/sum(w) # to normalise weights
        va[i] = sum(w_norm*alpha_tilde)  #MC step given M simulations
        vP[i] = sum(w_norm*alpha_tilde**2) - va[i]**2  #MC step given M simulations
        alpha_resample = np.random.choice(alpha_tilde, size=M, replace=True, p=w_norm)
    return va, vP

########################################3

def main():
    #yt = pd.read_excel('/Users/martijnjansen/Desktop/sv.xlsx')
    #RV_data = pd.read_csv('/Users/martijnjansen/Desktop/realized_volatility.csv')
    yt = pd.read_excel('Data/sv.xlsx')
    RV_data = pd.read_csv('Data/realized_volatility.csv')
    yt = np.array(yt['GBPUSD'])
    
    ####### a
    #plots.plot_return(yt/100,"GBPUSD")
    
    #sample moments
    mean_return = np.mean(yt/100)
    variance_return = np.var(yt/100)
    skewness_return = stats.skew(yt/100)
    kurtosis_return = stats.kurtosis(yt/100)
    
    print(f"Mean: {mean_return:.4f}")
    print(f"Variance: {variance_return:.4f}")
    print(f"Skewness: {skewness_return:.4f}")
    print(f"Kurtosis: {kurtosis_return:.4f}")
    
    #plots.hist_ret(yt/100,"GBPUSD")
   
    ####### b
    xt = transformation(yt)
    print("Transformed Daily Log Returns (xt):")
    #print(xt)
    
    #plots.plot_trans_return(xt,"GBPUSD")

    ####### c
    Z = 1
    R = 1
    H = (np.pi**2)/2

    res = est_params(xt)
    print(res)
    kappa = res.x[0]
    d = kappa
    sigma_sq = np.exp(kappa+1.27)
    phi = res.x[1]
    T = phi
    sigma_sq_eta = res.x[2]
    Q = sigma_sq_eta
    param_exchange = [kappa,sigma_sq,phi,sigma_sq_eta]
    print('kappa\t',kappa)
    print('sigma_sq\t',sigma_sq)
    print('phi\t\t',phi)
    print('sigma_sq_eta\t',sigma_sq_eta)
    print('sigma_eta\t',np.sqrt(sigma_sq_eta))

    ####### d
    a_gbpusd,v,F,K,P = kalman_filter(xt,d,Z,H,T,R,Q,False)
    alpha,V,r,N = kalman_smoother(xt,a_gbpusd,v,F,K,P)
    plots.plot_KFS_data(a_gbpusd,alpha,xt)
    plots.plot_KFS(a_gbpusd,alpha)

    ####### e
    aex_data = RV_data[RV_data['Symbol']=='.AEX']
    RV_aex = np.array(aex_data['rv5_ss'])
    y_aex = np.array(aex_data['open_to_close'])
    y_aex = y_aex*100
    x_aex = transformation(y_aex)

    #plots.plot_return(y_aex,"AEX")
    #plots.plot_trans_return(x_aex,"AEX")

    res = est_params(x_aex)
    print(res)
    kappa = res.x[0]
    d = kappa
    sigma_sq = np.exp(kappa+1.27)
    phi = res.x[1]
    T = phi
    sigma_sq_eta = res.x[2]
    Q = sigma_sq_eta
    param_aex = [kappa,sigma_sq,phi,sigma_sq_eta]
    print('kappa\t',kappa)
    print('sigma_sq\t',sigma_sq)
    print('phi\t\t',phi)
    print('sigma_sq_eta\t',sigma_sq_eta)
    print('sigma_eta\t',np.sqrt(sigma_sq_eta))

    a_aex,v_star,F,K,P = kalman_filter(x_aex,d,Z,H,T,R,Q,False)
    alpha,V,r,N = kalman_smoother(x_aex,a_aex,v_star,F,K,P)
    #plots.plot_KFS_data(a_aex,alpha,x_aex)
    #plots.plot_KFS(a_aex,alpha)

    ## new part abput RV measure
    new_data = np.log(RV_aex) + kappa
    a,x_star,F_temp,K,P = kalman_filter(new_data,d,Z,H,T,R,Q,False)
    beta_hat = est_beta(x_star,F,v_star)
    print(beta_hat)
    aex_adj = x_aex - beta_hat * np.log(RV_aex)
    a,v,F,K,P = kalman_filter(aex_adj,d,Z,H,T,R,Q,False)
    alpha,V,r,N = kalman_smoother(aex_adj,a,v,F,K,P)
    #plots.plot_KFS_data(a,alpha,x_aex)
    #plots.plot_KFS(a,alpha)

    ####### f
    # First part about gbpusd data
    #yt = return data gbpusd
    #use params from before
    #param_exchange [kappa,sigma_sq,phi,sigma_sq_eta]
    sigma_sq = param_exchange[1]
    phi = param_exchange[2]
    sigma_sq_eta = param_exchange[3]
    mu = np.mean(yt)
    M = 100
    va_gbpusd, vP_gbpusd = bootstrap(yt,mu,phi,sigma_sq,sigma_sq_eta,M)

    plt.plot(va_gbpusd,label='Bootstrap filter')
    plt.plot(a_gbpusd,label='QML filter')
    plt.legend()
    plt.show()

    sigma_sq = param_aex[1]
    sigma_sq = np.sqrt(sigma_sq)
    phi = param_aex[2]
    sigma_sq_eta = param_aex[3]
    mu = np.mean(y_aex)
    
    M = 100
    va_aex, vP_gbpusd = bootstrap(y_aex,mu,phi,sigma_sq,sigma_sq_eta,M)

    plt.plot(va_aex,label='Bootstrap filter')
    plt.plot(a_aex,label='QML filter')
    plt.legend()
    plt.show()

###########################################################
### call main
if __name__ == "__main__":
    main()
