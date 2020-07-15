"""
Created on Wed Jul 16, 2020
by Anusar Farooqui @policytensor
""" 
# This code automatically pulls data from the CBOE and Kenneth French's website, constructs pricing factors, estimates a cross-sectional pricing model using
# standard 2-pass OLS, estimates quarterly rolling prices for market risk (MKT), size (SMB), value (HML), and vol term spread (SLOPE) which is a proxy for 
# the risk appetite of broker-dealers. We show that the risk premium on SLOPE dwarfs the risk premia on Fama-French factors even for portfolios sorted on 
# size and value. 
#%% Clock utility function
import time

def TicTocGenerator():
    ti = 0           
    tf = time.time()
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti 

TicToc = TicTocGenerator()

def toc(tempBool=True):
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    toc(False)
#%% Import libraries and volatility data
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import RobustScaler
robust_scaler = RobustScaler()

url = 'http://www.cboe.com/publish/scheduledtask/mktdata/datahouse/vixcurrent.csv'
VixTbl = pd.read_csv(url,header=1,names=['Date','Open','High','Low','Close'])
VixTbl['Date'] = pd.to_datetime(VixTbl.Date)
# Pull the vix3m
url = 'http://www.cboe.com/publish/scheduledtask/mktdata/datahouse/vix3mdailyprices.csv'
Vix3MTbl = pd.read_csv(url,header=2,names=['Date','Open','High','Low','Close'])
Vix3MTbl['Date'] = pd.to_datetime(Vix3MTbl.Date)
# Merge dataframes
df = VixTbl.merge(Vix3MTbl,how='inner',on='Date',suffixes=('_vix','_vix3m'))
#%% Import Fama-French data
url = 'http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/'
fileID = 'F-F_Research_Data_Factors_daily_CSV.zip'
dateparse = lambda x: pd.datetime.strptime(x,'%Y%m%d')
ff3 = pd.read_csv(url+fileID,header=4,names=['Date','MKTRF','SMB','HML','RF'],engine='python',
                  parse_dates=['Date'],date_parser=dateparse,skipfooter=1,error_bad_lines=False,warn_bad_lines=False)
#%% Import 100 portfolios sorted on ME and BM
url = 'http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/'
fileID = '100_Portfolios_10x10_Daily_CSV.zip'
portfolios = pd.read_csv(url+fileID,header=12,prefix='var',nrows=24746) # HARD CODED NUMBERS! I still haven't figured out how to automate this.
#%% Create numpy return array and date vector
portfolios['Date'] = pd.to_datetime(portfolios['Unnamed: 0'],format='%Y%m%d')
dates = pd.DataFrame({'Date': df.Date})
df2 = portfolios.merge(dates,how='inner',on='Date')
Date = df2.Date
portret = df2.to_numpy()
portret = portret[:,1:-1].astype('float')

for i in range(np.size(portret,0)):
    for j in range(np.size(portret,1)):
        if portret[i,j]==-99.99:
            portret[i,j] = np.nan
#%% Price the cross-section
dates = pd.DataFrame({'Date': Date})
df2 = df.merge(dates,how='inner',on='Date')
df3 = df2.merge(ff3,how='inner',on='Date')
# Define feature
riskfac = df3.Close_vix.values - df3.Close_vix3m.values

rf = df3.RF.values
m = np.zeros(np.size(portret,1))

X = np.vstack((robust_scaler.fit_transform(riskfac.reshape(-1,1)).T,
               robust_scaler.fit_transform(df3.MKTRF.values.reshape(-1,1)).T,
               robust_scaler.fit_transform(df3.SMB.values.reshape(-1,1)).T,
               robust_scaler.fit_transform(df3.HML.values.reshape(-1,1)).T)).T

numFac = np.size(X,1)
b = np.zeros((np.size(portret,1),numFac))

X = sm.add_constant(X)
# Obtain betas from first-pass time-series regressions
for i in range(np.size(portret,1)):
    y = portret[:,i] - rf # LHS variable is excess returns
    m[i] = y.mean() # store expected excess returns for the cross-sectional regressions
    model = sm.OLS(y,X,missing='drop') # use sm.RLM with M=sm.robust.norms.AndrewWave() for robust estimates (it doesn't change the result)
    results = model.fit()
    for j in range(numFac):
        b[i,j] = results.params[1+j]
# Obtain price of risk from the second-pass cross-sectional regression
b = sm.add_constant(b)
model = sm.OLS(m,b,missing='drop')
results = model.fit()
# Plot factor prices
fig, ax = plt.subplots()
plt.xticks(range(5))
ax.set_xticklabels(['zero-beta rate','SLOPE','MKT','SMB','HML'])
plt.errorbar(range(5),results.params,yerr=results.bse,ecolor='k',elinewidth=0.2,barsabove=True,marker='o',capsize=2,color='k',linewidth=0.2,markersize=2) 
plt.plot(range(5),np.zeros(5),':k')
plt.title('Factor Prices')
plt.savefig("Prices.png", dpi=150,quality=95)
#%% Rolling factor prices 
X = np.vstack((robust_scaler.fit_transform(riskfac.reshape(-1,1)).T,
               robust_scaler.fit_transform(df3.MKTRF.values.reshape(-1,1)).T,
               robust_scaler.fit_transform(df3.SMB.values.reshape(-1,1)).T,
               robust_scaler.fit_transform(df3.HML.values.reshape(-1,1)).T)).T
numFac = np.size(X,1)

X = sm.add_constant(X)
y = portret - rf.reshape(-1,1) # LHS variable is excess returns

win = 63 # quarterly rolling estimates
Lambda = np.zeros((np.size(Date,0),numFac+1))
# Expanding window 2-pass regressions
for num in range(np.size(Date,0)):
    if num > win:
        tic() # set timer to track the expanding window — especially useful when using computationally-intensive robust regressions
        print(num)
        m = np.zeros(np.size(portret,1))
        b = np.zeros((np.size(portret,1),numFac))
        for i in range(np.size(portret,1)):
            try: # catch error
                m[i] = y[num-win:num,i].mean() # store expected excess return for the second pass
                model = sm.OLS(y[num-win:num,i],X[num-win:num,:],missing='drop')
                results = model.fit()
                for j in range(numFac):
                    b[i,j] = results.params[1+j] # obtain betas
            except:
                print(i) # report problem asset
        b = sm.add_constant(b) # don't forget to add a constant to the cross-sectional regressions
        model = sm.OLS(m,b,missing='drop') # use sm.RLM with M=sm.robust.norms.AndrewWave() for robust estimates (it doesn't change the result)
        results = model.fit()
        for k in range(numFac+1):
            Lambda[num,k] = results.params[k] # store factor prices
        toc()
#%% Plot factor prices
plt.figure()
plt.plot(Date[win:],Lambda[win:,1],'-k')
plt.title('Price of Systematic Risk')
plt.savefig("Price.png", dpi=150,quality=95)
        
fig, ax = plt.subplots(2, 2)
fig.tight_layout(pad=1)
ax[0,0].set_title('SLOPE')
ax[0,0].set_ylim([-0.75,2.5])
ax[0,0].plot(Date[win:],Lambda[win:,1],'-k')
ax[0,0].plot(Date[win:],np.zeros(np.size(Date[win:],0)),':k')

ax[0,1].set_title('MKT')
ax[0,1].set_ylim([-0.75,2.5])
ax[0,1].plot(Date[win:],Lambda[win:,2],'-k')
ax[0,1].plot(Date[win:],np.zeros(np.size(Date[win:],0)),':k')

ax[1,0].set_title('SMB')
ax[1,0].set_ylim([-0.75,2.5])
ax[1,0].plot(Date[win:],Lambda[win:,3],'-k')
ax[1,0].plot(Date[win:],np.zeros(np.size(Date[win:],0)),':k')

ax[1,1].set_title('HML')
ax[1,1].set_ylim([-0.75,2.5])
ax[1,1].plot(Date[win:],Lambda[win:,4],'-k')
ax[1,1].plot(Date[win:],np.zeros(np.size(Date[win:],0)),':k')

plt.savefig("FactorPrices.png", dpi=150,quality=95)