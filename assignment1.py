from __future__ import division
from pylab import *
import sys
import math
import numpy as np
import scipy as sp
import statsmodels.api as sm
import pandas as pd
from pandas import pivot_table
import matplotlib.pyplot as plt

def log(x):
	return np.log(x)


data = pd.read_csv('GWAnnual (1).csv')
data = data.set_index('yyyy')

index = data.index
price = data.Index
divid = data.D12
earn = data.E12

p_d = log(divid) - log(price)
p_e = log(earn) - log(price)
d_yeild = log(divid) - log(price).shift(1)


eprem = log(price +divid) - log(price.shift(1)) - log(1+data.Rfree)


summary = pd.DataFrame({'dividend_price': p_d, 'earnings_price': p_e, 'dividend_yeild': d_yeild, 'earnings_prem': eprem}, index = index)

#------- Replicating Table 1 -------


# # x is the regressor, d is the date up to which to run the regression
# # returns a regression object
def reg(x,d):
	return sm.OLS(eprem.loc[:d],sm.add_constant(x.shift(1).loc[:d]), missing = 'drop').fit()

def errors_oos(x, type = 'uc'):
	start_1 = pd.Series.first_valid_index(x) - 1871 + 1
	start_2 = pd.Series.first_valid_index(eprem) - 1871
	start = max([start_1,start_2])+1
	errors = []
	errors.extend([np.nan]*(start+1))
	means = pd.expanding_mean(eprem)

	for i in range(start,len(index)-1):
		d = 1871 + i 
		model = reg(x,d)
		parms = model.params

		if type == 'uc':
			pred = parms[0]+x.loc[d]*parms[1]
			

		elif type == 'hm':
			if parms[1] < 0:
				pred = means.loc[d]
			else:
				pred = parms[0]+x.loc[d]*parms[1]

		elif type == 'zc':
			parms[1] = max(parms[1],0)
			
			pred = parms[0]+x.loc[d]*parms[1]

		else:
			pred = parms[0]+x.loc[d]*parms[1]
			pred = max(pred,0)

		errors.append(pred-eprem.loc[1+d])

	result = pd.Series(errors, index = index)
	return result


def errors_is(x):
	values = reg(x,2005).fittedvalues
	errors = values - eprem
	return errors

# x is a pd series
def errors_n(x):
	means = pd.Series.mean(x)
	errors = means - x 
	return errors

def errors_noos(x):
	errors = pd.expanding_mean(x).shift(1) - x
	return errors

#---- Replicating Table 1 --------
def rsqr_is(x):
	res = sm.OLS(eprem,sm.add_constant(x.shift(1)), missing = 'drop').fit()
	rsqr = res.rsquared
	return (rsqr-(1-rsqr)*(1/(len(index)-sum(x.isnull().values)-2)))*100

def rsqr_oss(x, type = 'uc'):
	sse_a = pd.Series.sum((errors_oos(x, type).loc[1891:])**2)
	sse_n = pd.Series.sum((errors_noos(eprem).loc[1891:])**2)
	return 1-sse_a/sse_n

def rsqr_oos_bar(x, type = 'uc'):
	return (rsqr_oss(x, type) - (1-rsqr_oss(x, type))*(1/((2005-1890)-2)))*100

def d_rmse(x, type = 'uc'):
	sse_a = pd.Series.sum((errors_oos(x, type).loc[1891:])**2)
	sse_n = pd.Series.sum((errors_noos(eprem).loc[1891:])**2)
	rmse = (sse_a**(1/2) - sse_n**(1/2))/(2005-1890)**(1/2)
	return rmse*100

def statistic(x,type,stat):
	if stat == 'rsqr':
		return rsqr_oos_bar(x,type)
	elif stat == 'rmse':
		return d_rmse(x, type)


# GOYAL - WELCH 
#-----Table 1--------
var = [p_d,p_e,d_yeild]
names = ['d/p','e/p','d/y']
var_names = pd.Series(['Dividend Price Ratio', 'Earning Price Ratio', 'Dividend Yeild'], index = names)
res_is = pd.Series(map(rsqr_is,var),index = names)
res_oos = pd.Series(map(rsqr_oos_bar,var), index = names)
rmse = pd.Series(map(d_rmse, var), index = names)

col_names = ['Variable', 'R^2 - IS', 'R^2 - OOS', 'RMSE']
table_data = pd.DataFrame({col_names[0]: var_names, col_names[1]: res_is, col_names[2]: res_oos, col_names[3]: rmse},
	 columns = col_names)


# CAMBELL - THOMPSON
# -----Table------
types = ['uc','hm', 'zc', 'zp']
type_names = ['Unconstrained', 'Historical Mean', 'Positive Slope', 'Positive Forcast']
stats = ['rsqr', 'rmse']
stat_names = ['R^2', 'RMSE']

summary_stats = []

# def permute(x_array, y_array, z_arra, fn):
# 	for i in x_array.length:
# 		for j in y_array.length:
# 			for k in z_arra.length:
# 				fn([i, j, k], [x_array[i], y_array[j], z_array[k]])

# def foo(indicies, vals):
# 	vec = [var_names[i],type_names[j],stat_names[h]]
# 	value = statistic(x,type,stat)
# 	vec.append(value)
# 	summary_stats.append(vec)

# permute(var, types, stats, foo)

# map(foo, zip(var, types, stats))

i = 0
for x in var:
	j = 0
	for type in types:
		h = 0
		for stat in stats:
			vec = [var_names[i],type_names[j],stat_names[h]]
			value = statistic(x,type,stat)
			vec.append(value)
			summary_stats.append(vec)
			h = h+1
		j=j+1
	i=i +1


df_stats = pd.DataFrame(summary_stats, columns = ['Variable', 'Constraint', 'Statistic', 'Value'])

df_stats = pivot_table(df_stats, values = 'Value', index = ['Variable'], columns = ['Constraint', 'Statistic'])

print df_stats.to_string()


dif_oos =[]
dif_oos.append((-(errors_oos(p_e)**2-errors_noos(eprem)**2)).loc[1891:].cumsum())
dif_oos.append((-(errors_oos(p_d)**2-errors_noos(eprem)**2)).loc[1891:].cumsum())
dif_oos.append((-(errors_oos(d_yeild)**2-errors_noos(eprem)**2)).loc[1891:].cumsum())

dif_is = (errors_n(eprem)**2).cumsum() - (errors_is(p_e)**2).cumsum()

#---Regressions from Various Constraints ------

def prediction(type):
	preds = []
	preds.extend([np.nan]*4)
	means = pd.expanding_mean(eprem)

	if type == 'uc':
		for i in range(3,len(index)-1):
			d = 1871 + i 
			model = reg(x,d)
			parms = model.params
			preds.append(parms[0]+x.loc[d]*parms[1])

	elif type == 'hm':
		for i in range(3,len(index)-1):
			d = 1871 + i 
			model = reg(x,d)
			parms = model.params
			if parms[1] < 0:
				pred = means.loc[d]
			else:
				pred = parms[0]+x.loc[d]*parms[1]
			preds.append(pred)

	elif type == 'zc':
		for i in range(3,len(index)-1):
			d = 1871 + i 
			model = reg(x,d)
			parms = model.params
			parms[1] = max(parms[1],0)
			pred = parms[0]+x.loc[d]*parms[1]
			preds.append(pred)

	elif type == 'zp':
		for i in range(3,len(index)-1):
			d = 1871 + i 
			model = reg(x,d)
			parms = model.params
			pred = parms[0]+x.loc[d]*parms[1]
			preds.append(max(pred,0))

	result = pd.Series(preds, index = index)
	return result*100

ex_returns = pd.Series(map(prediction, types), index = type_names)
his_means = pd.expanding_mean(eprem).shift(1)*100


# --- PLOTTING ------
#--- GOYAL - WELCH PLOT ----
fig = plt.figure(figsize = (8,10))
fig.subplots_adjust(hspace = .5)
def my_plot(fig,data,title, position):
	ax = fig.add_subplot(3,1,position)
	data.plot(color = "black")
	dif_is.plot(marker = ".", color = "grey")
	plt.xlabel('Year')
	plt.ylabel('Cummulative SSE Difference', fontsize = 10)
	plt.xlim((1881,2010))
	plt.ylim(-.2,.2)
	ax.set_yticks(np.arange(-.2,.3,.1))
	plt.axvline(x = 1974, ymin = 0, ymax = 1, linewidth=2, color = 'r')
	ax.text(.7, .9,'Oil Shock',
		ha= 'center',
		va = "center",
		rotation = 'vertical'
		)
	plt.title(title)
	

my_plot(fig,dif_oos[0],'ep', 1)
my_plot(fig,dif_oos[1],'dp', 2)
my_plot(fig,dif_oos[2],'dy', 3)


#-Cambell - Thompson Plots
fig = plt.figure(figsize = (8,10))
fig.subplots_adjust(hspace = .5)
def ct_plot(fig,data,title, position):
	ax = fig.add_subplot(3,1,position)
	data.plot(color = "black", label = 'Return Forcast')
	his_means.plot(marker = ".", color = "grey", label = 'Historical Mean')
	plt.xlabel('Year')
	plt.ylabel('Excess Returns', fontsize = 10)
	plt.xlim((1871,2010))
	plt.axvline(x = 1974, ymin = 0, ymax = 1, linewidth=2, color = 'r')
	ax.text(.7, .9,'Oil Shock',
		ha= 'center',
		va = "center",
		rotation = 'vertical'
		)
	plt.title(title)
	plt.legend(fontsize = 10, loc = 'upper left')

for i in range(3):
	type = type_names[i+1]
	ct_plot(fig,ex_returns[type].loc[1891:],type,i+1)

# plt.show()
savefig('foo.png')


# #Bootstrap the F-Statistic
# interval_95 = []

# for i in range(len(dif)):
# 	d = 1891 + i + 1
# 	er_a = errors_oos(p_e).loc[1891:d]
# 	er_n = errors_n(eprem).loc[1891:d]





