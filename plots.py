import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats


def plot_kalman_filter(df,a,v,F,K,P):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8), tight_layout=True)
    z_95 = 1.645
    a_lower = a - z_95 * np.sqrt(P)
    a_upper = a + z_95 * np.sqrt(P)
    
    axs[0, 0].plot(df['Year'][1:],df['Nile'][1:], 'bo', markersize=2, label = 'Nile')
    axs[0, 0].plot(df['Year'][1:], a[1:],color='black')
    axs[0, 0].plot(df['Year'][1:], a_lower[1:],color='green')
    axs[0, 0].plot(df['Year'][1:], a_upper[1:],color='green')
    axs[0, 0].set_title('y_t and a_t')

    axs[0, 1].plot(df['Year'][1:], P[1:])
    axs[0, 1].set_title('P_t')

    axs[1, 0].plot(df['Year'][1:], v[1:])
    axs[1, 0].axhline(y=0, linestyle='-')
    axs[1, 0].set_title('v_t')

    axs[1, 1].plot(df['Year'][1:], F[1:])
    axs[1, 1].set_title('F_t')

    for ax in axs.flat:
        ax.tick_params(axis='both', which='major', labelsize=12)

    plt.show()

def plot_kalman_smoother(df,alpha,V,r,N):
    number_obs = len(df)
    fig, axs = plt.subplots(2, 2, figsize=(10, 8), tight_layout=True)
    
    z_95 = 1.645
    alpha_lower = alpha - z_95 * np.sqrt(V)
    alpha_upper = alpha + z_95 * np.sqrt(V)
    
    axs[0, 0].plot(df['Year'],df['Nile'], 'bo', markersize=2, label = 'Nile')
    axs[0, 0].plot(df['Year'], alpha,color='black')
    axs[0, 0].plot(df['Year'], alpha_lower,color='green')
    axs[0, 0].plot(df['Year'], alpha_upper,color='green')
    axs[0, 0].set_title('y_t and alpha_t')

    axs[0, 1].plot(df['Year'], V)
    axs[0, 1].set_title('V_t')

    axs[1, 0].plot(df['Year'][:number_obs-1], r[:number_obs-1])
    axs[1, 0].axhline(y=0, linestyle='-')
    axs[1, 0].set_title('r_t')

    axs[1, 1].plot(df['Year'][:number_obs-1], N[:number_obs-1])
    axs[1, 1].set_title('N_t')

    for ax in axs.flat:
        ax.tick_params(axis='both', which='major', labelsize=12)

    plt.show()

def plot_kalman_obs_dist(df,epsilon_hat, smoothed_sd_eps, eta_hat, smoothed_sd_eta):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8), tight_layout=True)
  
    axs[0, 0].plot(df['Year'], epsilon_hat)
    axs[0, 0].axhline(y=0, linestyle='-')
    axs[0, 0].set_title('epsilon hat')

    axs[0, 1].plot(df['Year'], smoothed_sd_eps)
    axs[0, 1].set_title('obs error var')

    axs[1, 0].plot(df['Year'],eta_hat)
    axs[1, 0].axhline(y=0, linestyle='-')
    axs[1, 0].set_title('eta hat')

    axs[1, 1].plot(df['Year'], smoothed_sd_eta)
    axs[1, 1].set_title('state error var')

    for ax in axs.flat:
        ax.tick_params(axis='both', which='major', labelsize=12)

    plt.show()

def plot_simulation(df,alpha,alpha_plus,alpha_tilde,epsilon_hat,epsilon_tilde,eta_hat,eta_tilde):
    number_obs = len(df)
    fig, axs = plt.subplots(2, 2, figsize=(10, 8), tight_layout=True)
  
    axs[0, 0].plot(df['Year'], alpha, color='black')
    axs[0, 0].plot(df['Year'],alpha_plus, 'bo', markersize=2)
    axs[0, 0].set_title('alpha hat and alpha plus')

    axs[0, 1].plot(df['Year'], alpha, color='black')
    axs[0, 1].plot(df['Year'],alpha_tilde, 'bo', markersize=2)
    axs[0, 1].set_title('alpha hat and alpha tilde')

    axs[1, 0].plot(df['Year'],epsilon_hat)
    axs[1, 0].plot(df['Year'],epsilon_tilde, 'bo', markersize=2)
    axs[1, 0].axhline(y=0, linestyle='-')
    axs[1, 0].set_title('epsilon hat')

    axs[1, 1].plot(df['Year'],eta_hat)
    axs[1, 1].plot(df['Year'][:number_obs-1],eta_tilde, 'bo', markersize=2)
    axs[1, 1].axhline(y=0, linestyle='-')
    axs[1, 1].set_title('eta hat')

    for ax in axs.flat:
        ax.tick_params(axis='both', which='major', labelsize=12)

    plt.show()

def plot_kalman_weights(df,broken_nile,new_a,new_P,new_alpha,new_V,break_begin1,break_end1,break_begin2,break_end2):
    number_obs = len(df)
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 8), tight_layout=True)
    
    axs[0, 0].plot(df['Year'], broken_nile, label = 'Nile')
    axs[0, 0].plot(df['Year'], new_a,color='black')
    axs[0, 0].set_title('data and filtered a')
    
    axs[0, 1].plot(df['Year'], new_P)
    axs[0, 1].set_title('new P')
   
    axs[1, 0].plot(df['Year'],broken_nile, label = 'Nile')
    axs[1, 0].plot(df['Year'], new_alpha,color='black')
    axs[1, 0].set_title('data and filtered alpha')
    
    axs[1, 1].plot(df['Year'], new_V)
    axs[1, 1].set_title('new V')
    
    for ax in axs.flat:
        ax.tick_params(axis='both', which='major', labelsize=12)

    plt.show()

def plot_forecasting(df,forecasting_time,a_forecast,P_forecast,F_forecast,lower_bound_50,upper_bound_50):
    number_obs = len(df)
    fig, axs = plt.subplots(2, 2, figsize=(10, 8), tight_layout=True)

    axs[0, 0].plot(df['Year'],df['Nile'], 'bo', markersize=2)
    axs[0, 0].plot(forecasting_time[1:], a_forecast[1:], color='black')
    axs[0, 0].plot(forecasting_time[number_obs:], lower_bound_50, color='grey')
    axs[0, 0].plot(forecasting_time[number_obs:], upper_bound_50, color='grey')
    axs[0, 0].set_title('a forecast and data')
    
    axs[0, 1].plot(forecasting_time[1:], P_forecast[1:], color='black')
    axs[0, 1].set_title('P forecast')
    
    axs[1, 0].plot(forecasting_time[1:], a_forecast[1:], color='black')
    axs[1, 0].set_title('a forecast')

    axs[1, 1].plot(forecasting_time[1:], F_forecast[1:], color='black')
    axs[1, 1].set_title('F forecast')
    
    for ax in axs.flat:
        ax.tick_params(axis='both', which='major', labelsize=12)

    plt.show()

def plot_standardised_errors(df,e):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8), tight_layout=True)

    axs[0, 0].plot(df['Year'],e,color='black') 
    axs[0, 0].axhline(y=0, linestyle='-')
    axs[0, 0].set_title('e_t')
    
    axs[0, 1].set_xlim(-3.8, 3.8)
    n, bins, patches = axs[0, 1].hist(e, bins=13, density=True, color='blue',edgecolor='black')
    x = np.linspace(-3.8, 3.8, 100)
    p = stats.norm.pdf(x, np.mean(e), np.std(e))
    axs[0, 1].plot(x, p, 'k', linewidth=2) 
    axs[0, 1].set_title('Histogram e_t')
    
    sorted_data = np.sort(e)
    expected_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, 100))
    axs[1,0].plot(expected_quantiles, sorted_data)
    axs[1,0].plot([-2.4,2.4], [-2.4,2.4], ls="--")
    axs[1,0].set_title('QQ plot e_t')
    
    #sm.graphics.tsa.plot_acf(e, lags=10,alpha = 0.9, ax=axs[1, 1],marker=None) 
    axs[1, 1].set_title('Correlogram e_t')

    #To do it in blocks instead of lines
    acf_vals = sm.tsa.acf(e, nlags=10)
    axs[1,1].bar(np.arange(1,len(acf_vals)), acf_vals[1:], width=0.4, color='blue', alpha=0.9)
    axs[1,1].axhline(y=0, linestyle='-')
    axs[1,1].set_ylim(-1, 1)

    for ax in axs.flat:
        ax.tick_params(axis='both', which='major', labelsize=12)

    plt.show()

def plot_stand_smoothed_res(df,u_star,r_star):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8), tight_layout=True)

    axs[0, 0].plot(df['Year'],u_star,color='black') 
    axs[0, 0].axhline(y=0, linestyle='-')
    axs[0, 0].set_title('u_star')   
    
    n, bins, patches = axs[0, 1].hist(u_star, bins=13, density=True, color='blue',edgecolor='black')
    x = np.linspace(-4, 3, 100)
    p = stats.norm.pdf(x, np.mean(u_star), np.std(u_star))
    axs[0, 1].plot(x, p, 'k', linewidth=2) 
    axs[0, 1].set_title('Histogram u_star')  
    
    axs[1, 0].plot(df['Year'],r_star,color='black') 
    axs[1, 0].axhline(y=0, linestyle='-')
    axs[1, 0].set_title('r_star')
    
    n, bins, patches = axs[1, 1].hist(r_star, bins=13, density=True, color='blue',edgecolor='black')
    x = np.linspace(-4, 3, 100)
    p = stats.norm.pdf(x, np.mean(r_star), np.std(r_star))
    axs[1, 1].plot(x, p, 'k', linewidth=2) 
    axs[1, 1].set_title('Histogram r_star')

    for ax in axs.flat:
        ax.tick_params(axis='both', which='major', labelsize=12)

    plt.show()