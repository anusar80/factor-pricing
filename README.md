# factor-pricing
Python code for daily asset pricing. This code automatically pulls data from the CBOE and Kenneth French's website, constructs pricing factors, 
estimates a cross-sectional pricing model using standard 2-pass OLS, estimates quarterly rolling prices for market risk (MKT), size (SMB), value (HML), 
and vol term spread (SLOPE) which is a proxy for the risk appetite of broker-dealers. We show that the risk premium on SLOPE dwarfs the risk premia on 
Fama-French factors even for portfolios sorted on size and value. 
