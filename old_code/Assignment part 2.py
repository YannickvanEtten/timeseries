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

########################################3

def main():
    yt = pd.read_excel("Data/sv.xlsx")
    RV_data = pd.read_csv("Data/realized_volatility.csv")
    yt = np.array(yt['GBPUSD'])
    
    ####### a
    plt.plot(yt/100)
    plt.show()
    
    ####### b
    xt = transformation(yt)
    plt.plot(xt)
    plt.show()

    ####### c
    Z = 1
    R = 1
    H = (np.pi**2)/2

    res = est_params(xt)
    print(res)
    kappa = res.x[0]
    d = kappa
    sigma = np.exp(kappa+1.27)
    phi = res.x[1]
    T = phi
    sigma_sq_eta = res.x[2]
    Q = sigma_sq_eta
    print('kappa',kappa)
    print('sigma',sigma)
    print('phi',phi)
    print('sigma_sq_eta',sigma_sq_eta)
    print('sigma_eta',np.sqrt(sigma_sq_eta))

    ####### d
    a,v,F,K,P = kalman_filter(xt,d,Z,H,T,R,Q,False)
    alpha,V,r,N = kalman_smoother(xt,a,v,F,K,P)
    plt.plot(a,label = 'filtered a', c='red')
    plt.plot(alpha,label = 'smoothed alpha', c='orange')
    plt.scatter(np.arange(len(xt)),xt, s=10)
    plt.legend()
    plt.show()
    plt.plot(a,label = 'filtered a')
    plt.plot(alpha,label = 'smoothed alpha')
    plt.legend()
    plt.show()

    ####### e
    aex_data = RV_data[RV_data['Symbol']=='.AEX']
    p_aex = np.array(aex_data['close_price'])
    RV_aex = np.array(aex_data['rv5_ss'])
    y_aex = calc_return(p_aex)
    x_aex = transformation(p_aex/100)

    plt.plot(y_aex/100)
    plt.show()
    
    plt.plot(x_aex)
    plt.show()

    res = est_params(x_aex)
    print(res)
    kappa = res.x[0]
    d = kappa
    sigma = np.exp(kappa+1.27)
    phi = res.x[1]
    T = phi
    sigma_sq_eta = res.x[2]
    Q = sigma_sq_eta
    print('kappa',kappa)
    print('sigma',sigma)
    print('phi',phi)
    print('sigma_sq_eta',sigma_sq_eta)
    print('sigma_eta',np.sqrt(sigma_sq_eta))

    a,v,F,K,P = kalman_filter(x_aex,d,Z,H,T,R,Q,False)
    alpha,V,r,N = kalman_smoother(x_aex,a,v,F,K,P)
    plt.plot(a,label = 'filtered a', c='red')
    plt.plot(alpha,label = 'smoothed alpha', c='orange')
    plt.scatter(np.arange(len(x_aex)),x_aex, s=10)
    plt.legend()
    plt.show()
    plt.plot(a,label = 'filtered a')
    plt.plot(alpha,label = 'smoothed alpha')
    plt.legend()
    plt.show()

    new_data = np.log(RV_aex) + kappa
    a,v,F,K,P = kalman_filter(new_data,d,Z,H,T,R,Q,False)
    beta_hat = est_beta(new_data,F,v)
    print(beta_hat)

    ####### f


###########################################################
### call main
if __name__ == "__main__":
    main()