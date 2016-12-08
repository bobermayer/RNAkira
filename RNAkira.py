import os
import sys
import numpy as np
import pandas as pd
import scipy.stats
import scipy.interpolate
import scipy.optimize
import statsmodels.nonparametric.smoothers_lowess
from optparse import OptionParser
from collections import defaultdict,OrderedDict,Counter

def trim_mean_std(x):

	""" wrapper of trimmed mean and (adjusted) std for 5-95 percentile """

	ok=np.isfinite(x)
	return [scipy.stats.tmean(x[ok],limits=np.percentile(x[ok],[5,95])),\
			1.266*scipy.stats.tstd(x[ok],limits=np.percentile(x[ok],[5,95]))]

def p_adjust_bh(p):

	""" Benjamini-Hochberg p-value correction for multiple hypothesis testing
		(stolen and modified for nan values from here: http://stackoverflow.com/questions/7450957/how-to-implement-rs-p-adjust-in-python) """

	p = np.asfarray(p)
	ok = np.isfinite(p)
	by_descend = p[ok].argsort()[::-1]
	by_orig = by_descend.argsort()
	steps = float(len(p[ok])) / np.arange(len(p[ok]), 0, -1)
	q = np.zeros_like(p)
	q[ok] = np.minimum(1, np.minimum.accumulate(steps * p[ok][by_descend]))[by_orig]
	q[~ok] = np.nan
	return q

def get_steady_state_values (x,T,use_ribo,use_deriv=False):

	""" given synthesis, degradation, processing and potentially translation rates x=(a,b,c,d) and labeling time T
	returns instantaneous steady-state values for elu-precursor,elu-total,flowthrough-precursor,flowthrough-total,unlabeled-precursor,unlabeled-total
	ribo if use_ribo is set
	includes derivatives w.r.t. log(a),log(b),log(c) and maybe log(d) if use_deriv is set """

	if use_ribo:
		a,b,c,d=np.exp(x)
	else:
		a,b,c=np.exp(x)

	# for elu-precursor, elu-total (mature+precursor), flowthrough-precursor, flowthrough-total, unlabeled-precursor, unlabeled-total
	exp_mean=[a*(1.-np.exp(-c*T))/c,\
			  a*(b**2*(1.-np.exp(-c*T))-c**2*(1.-np.exp(-b*T)))/(b*c*(b-c)),\
			  a*np.exp(-c*T)/c,\
			  a*(b**2*np.exp(-c*T)-c**2*np.exp(-b*T))/(b*c*(b-c)),\
			  a/c,\
			  a/b+a/c]

	# add ribo value
	if use_ribo:
		exp_mean=exp_mean[:4]+[d*a/b]+exp_mean[4:]

	if use_deriv:
		# get the derivatives w.r.t. log(a),log(b),log(c)
		exp_mean_deriv=[[a*(1.-np.exp(-c*T))/c, \
						 a*(b**2*(1.-np.exp(-c*T))-c**2*(1.-np.exp(-b*T)))/(b*c*(b-c)),\
						 a*np.exp(-c*T)/c,\
						 a*(b**2*np.exp(-c*T)-c**2*np.exp(-b*T))/(b*c*(b-c)),\
						 a/c,
						 a/b+a/c],\
						[0, 
						 a*(b*(b-c)*(2*b*(1.-np.exp(-c*T))-c**2*T*np.exp(-b*T))-\
							(b**2*(1.-np.exp(-c*T))-c**2*(1.-np.exp(-b*T)))*(2*b-c))/(b*c*(b-c)**2),\
						 0,\
						 a*(b*(b-c)*(T*c**2*np.exp(-b*T)+2*b*np.exp(-c*T))-(b**2*np.exp(-c*T)-c**2*np.exp(-b*T))*(2*b-c))/(b*c*(b-c)**2),\
						 0,\
						 -a/b],\
						[a*(np.exp(-c*T)*(1.+c*T)-1)/c,\
						 a*(c*(b-c)*(b**2*T*np.exp(-c*T)-2*c*(1.-np.exp(-b*T)))-\
							(b**2*(1.-np.exp(-c*T))-c**2*(1.-np.exp(-b*T)))*(b-2*c))/(b*c*(b-c)**2),\
						 -a*np.exp(-c*T)*(1.+c*T)/c,\
						 a*(c*(b-c)*(-2*c*np.exp(-b*T)-T*b**2*np.exp(-c*T))-(b**2*np.exp(-c*T)-c**2*np.exp(-b*T))*(b-2*c))/(b*c*(b-c)**2),\
						 -a/c,\
						 -a/c]]

		if use_ribo:
			# add derivatives of ribo values and derivatives w.r.t. log(d)
			ribo_vals=[[d*a/b],[-d*a/b],[0]]
			exp_mean_deriv=[exp_mean_deriv[i][:4]+ribo_vals[i]+exp_mean_deriv[i][4:] for i in range(3)]+[[0,0,0,0,d*a/b,0,0]]

	if use_deriv:
		return (np.array(exp_mean),map(np.array,exp_mean_deriv))
	else:
		return np.array(exp_mean)

def get_rates(time_points, pars, errs=None):

	""" expands log rates over time points given rate change parameters """

	use_ribo='log_d0' in pars
	times=np.array(map(float,time_points))
	ntimes=len(times)
	
	# exponential trend in rates modeled by slopes alpha, beta, gamma and maybe delta (log rates have linear trend)
	log_a=pars['log_a0']*np.ones(ntimes)
	if 'alpha' in pars:
		log_a+=pars['alpha']*times
	log_b=pars['log_b0']*np.ones(ntimes)
	if 'beta' in pars:
		log_b+=pars['beta']*times
	log_c=pars['log_c0']*np.ones(ntimes)
	if 'gamma' in pars:
		log_c+=pars['gamma']*times

	rates=[log_a,log_b,log_c]

	if use_ribo:
		log_d=pars['log_d0']*np.ones(ntimes)
		if 'delta' in pars:
			log_d+=pars['delta']*times
		rates.append(log_d)

	if errs is None:

		return rates

	else:

		log_a_err=errs['log_a0']*np.ones(ntimes)
		if 'alpha' in errs:
			log_a_err+=errs['alpha']*times
		log_b_err=errs['log_b0']*np.ones(ntimes)
		if 'beta' in errs:
			log_b_err+=errs['beta']*times
		log_c_err=errs['log_c0']*np.ones(ntimes)
		if 'gamma' in errs:
			log_c_err+=errs['gamma']*times

		rate_errs=[log_a_err,log_b_err,log_c_err]

		if use_ribo:
			log_d_err=errs['log_d0']*np.ones(ntimes)
			if 'delta' in errs:
				log_d_err+=errs['delta']*times
			rate_errs.append(log_d_err)

		return zip(rates,rate_errs)
			
def steady_state_log_likelihood (x, obs_mean, obs_std, T, time_points, prior_mu, prior_std, model_pars, use_ribo, use_deriv):

	""" log-likelihood function for difference between expected and observed values, including all priors """

	nrates=(4 if use_ribo else 3)

	times=np.array(map(float,time_points))
	ntimes=len(times)

	# get instantaneous rates a,b,c,d at each timepoint
	log_rates=[x[i]*np.ones(ntimes) for i in range(nrates)]
	k=nrates
	if 'alpha' in model_pars:
		log_rates[0]+=x[k]*times
		k+=1
	if 'beta' in model_pars:
		log_rates[1]+=x[k]*times
		k+=1
	if 'gamma' in model_pars:
		log_rates[2]+=x[k]*times
		k+=1
	if nrates==4 and 'delta' in model_pars:
		log_rates[3]+=x[k]*times

	fun=0
	if use_deriv:
		grad=np.zeros(len(x))

	# add up model log-likelihoods for each time point and each replicate
	for i,t in enumerate(times):

		# instantaneous values of the rates at time t
		log_rates_here=np.array([lr[i] for lr in log_rates])

		# first add to log-likelihood the priors on rates at each time point from empirical distribution (parametrized by mu and std)
		diff=log_rates_here-prior_mu
		fun+=np.sum(-.5*(diff/prior_std)**2-.5*np.log(2.*np.pi)-np.log(prior_std))

		if use_deriv:

			g=list(-diff/prior_std**2)
			# derivatives w.r.t. alpha etc.
			if 'alpha' in model_pars:
				g.append(g[0]*t)
			if 'beta' in model_pars:
				g.append(g[1]*t)
			if 'gamma' in model_pars:
				g.append(g[2]*t)
			if 'delta' in model_pars:
				g.append(g[2]*t)
			grad+=np.array(g)

		if use_deriv:
			exp_mean,exp_mean_deriv=get_steady_state_values(log_rates_here,T,use_ribo,use_deriv)
			# add derivatives w.r.t. alpha etc. if necessary
			if 'alpha' in model_pars:
				exp_mean_deriv.append(exp_mean_deriv[0]*t)
			if 'beta' in model_pars:
				exp_mean_deriv.append(exp_mean_deriv[1]*t)
			if 'gamma' in model_pars:
				exp_mean_deriv.append(exp_mean_deriv[2]*t)
			if 'delta' in model_pars:
				exp_mean_deriv.append(exp_mean_deriv[3]*t)
		else:
			exp_mean=get_steady_state_values(log_rates_here,T,use_ribo)

		diff=obs_mean[i]-exp_mean

		fun+=np.sum(-.5*(diff/obs_std[i])**2-.5*np.log(2.*np.pi)-np.log(obs_std[i]))
		if use_deriv:
			grad+=np.dot(exp_mean_deriv,np.sum(diff/obs_std[i]**2,axis=0))

	if use_deriv:
		# return negative log likelihood and gradient
		return (-fun,-grad)
	else:
		# return negative log likelihood
		return -fun

def fit_model (obs_mean, obs_std, T, time_points, model_priors, parent, model, model_pars, use_ribo, min_args):

	""" fits a specific model to data """

	test_gradient=False

	nrates=(4 if use_ribo else 3)

	if not model.endswith('all'):

		# for most models, fit all time points simultaneously, using constant or linear trends on the rates
		if parent is not None:
			if model.endswith('0'):
				initial_estimate=parent['est_pars'].mean(axis=0)
			else:
				initial_estimate=np.concatenate([parent['est_pars'],[0]])
		else:
			# use initial estimate from priors
			raise Exception("shouldn't happen?")
			initial_estimate=np.concatenate([model_priors['mu'].values,[0]*(len(model_pars)-nrates)])

		# arguments to minimization
		args=(obs_mean, obs_std, T, time_points, model_priors['mu'].values[:nrates], model_priors['std'].values[:nrates],\
			  model_pars, use_ribo, min_args['jac'])

		# test gradient against numerical difference if necessary
		if test_gradient:

			args_here=args[:-2]+(False,)+args[-1:]
			pp=initial_estimate
			eps=1.e-6*np.abs(initial_estimate)+1.e-8
			fun,grad=steady_state_log_likelihood(pp,*args)
			my_grad=np.array([(steady_state_log_likelihood(np.array(list(pp[:i])+[pp[i]+eps[i]]+list(pp[i+1:])),*args_here)\
							   -steady_state_log_likelihood(np.array(list(pp[:i])+[pp[i]-eps[i]]+list(pp[i+1:])),*args_here))/(2*eps[i]) for i in range(len(pp))])
			diff=np.sum((grad-my_grad)**2)/np.sum(grad**2)
			if diff > 1.e-8:
				raise Exception('\n!!! gradient diff: {0:.3e}'.format(diff))

		# perform minimization
		res=scipy.optimize.minimize(steady_state_log_likelihood, \
									initial_estimate,\
									args=args, \
									**min_args)

		# get fit errors from Hessian inverse if possible
		if min_args['method']=='L-BFGS-B':
			err=np.sqrt(np.diag(res.hess_inv.todense()))
		elif min_args['method']=='BFGS':
			err=np.sqrt(np.diag(res.hess_inv))
		else:
			err=np.nan*np.ones(len(model_pars))

		# collect results
		result=dict(est_pars=pd.Series(res.x,index=model_pars),\
					est_err=pd.Series(err,index=model_pars),\
					L=-res.fun,\
					success=res.success,\
					npars=len(model_pars),
					model=model)

	else:

		# in the 'all' models, treat every time point independently
		x=[]
		err=[]
		L=0
		npars=0
		success=True

		if parent is None:
			# take prior estimates for each time point
			log_rates=[model_priors['mu'].ix[k]*np.ones(len(time_points)) for k in range(nrates)]
		else:
			
			initial_pars=parent['est_pars']
			log_rates=get_rates(time_points,initial_pars)

		for i,t in enumerate(time_points):

			initial_estimate=np.array([lr[i] for lr in log_rates])

			args=(obs_mean[i,None], obs_std[i,None], T, [t], model_priors['mu'].values[:nrates], model_priors['std'].values[:nrates],\
				  model_pars, use_ribo, min_args['jac'])

			# test gradient against numerical difference if necessary
			if test_gradient:

				args_here=args[:-2]+(False,)+args[-1:]
				pp=initial_estimate
				eps=1.e-6*np.abs(initial_estimate)+1.e-8
				fun,grad=steady_state_log_likelihood(pp,*args)
				my_grad=np.array([(steady_state_log_likelihood(np.array(list(pp[:i])+[pp[i]+eps[i]]+list(pp[i+1:])),*args_here)\
								   -steady_state_log_likelihood(np.array(list(pp[:i])+[pp[i]-eps[i]]+list(pp[i+1:])),*args_here))/(2*eps[i]) for i in range(len(pp))])
				diff=np.sum((grad-my_grad)**2)/np.sum(grad**2)
				if diff > 1.e-8:
					raise Exception('\n!!! gradient diff: {0:.3e}'.format(diff))

			res=scipy.optimize.minimize(steady_state_log_likelihood, \
										initial_estimate,\
										args=args, \
										**min_args)

			x.append(res.x)

			# get fit errors from Hessian inverse if possible
			if min_args['method']=='L-BFGS-B':
				err.append(np.sqrt(np.diag(res.hess_inv.todense())))
			elif min_args['method']=='BFGS':
				err.append(np.sqrt(np.diag(res.hess_inv)))
			else:
				err.append(np.nan*np.ones(len(model_pars)))

			L+=-res.fun
			success=success & res.success
			npars+=len(model_pars)

		result=dict(est_pars=pd.DataFrame(x,columns=model_pars,index=time_points),\
					est_err=pd.DataFrame(err,columns=model_pars,index=time_points),\
					L=L,\
					success=success,\
					npars=npars,\
					model=model)

	if parent is not None:
		# calculate p-value from LRT test using chi2 distribution
		pval=scipy.stats.chisqprob(2*np.abs(result['L']-parent['L']),np.abs(result['npars']-parent['npars']))
		result['LRT-p']=(pval if np.isfinite(pval) else np.nan)

	return result

def plot_data_rates_fits (time_points, replicates, obs_vals, T, parameters, results, use_ribo, title='', priors=None, sig_level=0.01):

	""" function to plot summary of results for a specific gene """

	from matplotlib import pyplot as plt

	times=np.array(map(float,time_points))
	ntimes=len(times)

	def jitter():
		return np.random.normal(loc=0,scale=.1*np.min(np.diff(times)),size=len(time_points))

	cols=['elu-precursor','elu-total','flowthrough-precursor','flowthrough-total','unlabeled-precursor','unlabeled-total']
	if use_ribo:
		cols=cols[:4]+['ribo']+cols[4:]
		nrates=4
	else:
		nrates=3

	ndim=len(cols)

	fig=plt.figure(figsize=(15,6))
	fig.clf()
	fig.subplots_adjust(wspace=.4,hspace=.4)

	ax=[fig.add_subplot(2,ndim,i+1) for i in range(ndim+nrates)]

	# first plot data and fits to data

	for i in range(ndim):
		for r in replicates:
			ax[i].plot(times,obs_vals[cols[i]].xs(r,level=1),'ko',label='_nolegend_',mfc='none')

	if parameters is not None:
		exp_vals=get_steady_state_values(get_rates(time_points,parameters),T,use_ribo)
		for i in range(ndim):
			ax[i].plot(times,exp_vals[i],'k-',label='theo. mean')

	for level,vals in enumerate(results):
		pred_vals=get_steady_state_values(get_rates(time_points,vals['est_pars']),T,use_ribo)
		for i in range(ndim):
			if level > 1:
				qval=vals['LRT-q']
				ax[i].plot(times,pred_vals[i],linestyle=('-' if qval < sig_level else '--'),\
						   label='{0} (q={1:.2g})'.format(vals['model'],qval))
			elif level==1:
				ax[i].plot(times,pred_vals[i],linestyle='-', label='{0}'.format(vals['model']))
			else:
				ax[i].plot(times,pred_vals[i],linestyle=':',label='initial')

	for i,c in enumerate(cols):
		ax[i].set_title(c,size=10)
		ax[i].set_xlabel('time')

	ax[0].set_ylabel('expression')

	# then plot rates and fits to rates
	if parameters is not None:
		for i,p in enumerate(get_rates(time_points, parameters)):
			ax[ndim+i].plot(times,p,'k.-',label='true')

	if priors is not None:
		for i in range(nrates):
			ax[ndim+i].fill_between(times,np.ones(ntimes)*(priors.ix[i,'mu']-1.96*priors.ix[i,'std']),\
								 y2=np.ones(ntimes)*(priors.ix[i,'mu']+1.96*priors.ix[i,'std']),color='Gray',alpha=.25,label='95% prior')
	for level,vals in enumerate(results):
		for i,p,err in enumerate(get_rates(time_points,vals['est_pars'],vals['est_err'])):
			if level > 1:
				qval=vals['LRT-q']
				ax[ndim+i].errorbar(times,p,yerr=err,linestyle=('-' if qval < sig_level else '--'),label='{0} (q={1:.2g})'.format(vals['model'],qval))
			elif level==1:
				ax[ndim+i].errorbar(times,p,yerr=err,linestyle='-',label='{0}'.format(vals['model']))
			else:
				ax[ndim+i].errorbar(times,p,yerr=err,linestyle=':',label='initial')

	for i,t in enumerate(['synthesis','degradation','processing','translation'][:nrates]):
		ax[ndim+i].set_title(t+' rate',size=10)
		ax[ndim+i].set_xlabel('time')

	ax[ndim].set_ylabel('log rate')
	ax[ndim+nrates-1].legend(loc=2,frameon=False,prop={'size':10},bbox_to_anchor=(1.1,1))

	fig.suptitle(title,size=10)

def RNAkira (values, T, sig_level=0.01, min_TPM_ribo=.1, maxlevel=None, priors=None):

	""" main routine in this package: given dataframe of TPM values and labeling time, estimates empirical priors and fits models of increasing complexity """

	time_points=np.unique(values.columns.get_level_values(2))
	nreps=len(np.unique(values.columns.get_level_values(3)))

	rna_cols=['elu-precursor','elu-total','flowthrough-precursor','flowthrough-total','unlabeled-precursor','unlabeled-total']
	ribo_cols=rna_cols[:4]+['ribo']+rna_cols[4:]

	ndim=7

	use_ribo=(values['mean','ribo'] > min_TPM_ribo).any(axis=1)
	use_rna=~use_ribo

	print >> sys.stderr, '\n[RNAkira] analyzing {0} genes with ribo+rna and {1} genes with rna only ({2} time points, {3} replicates)'.format(use_ribo.sum(),(~use_ribo).sum(),len(time_points),nreps)

	min_args=dict(method='L-BFGS-B',jac=True,options={'disp':False, 'ftol': 1.e-15, 'gtol': 1.e-10})
	#min_args=dict(method='L-BFGS-B',jac=False,options={'disp':False, 'ftol': 1.e-15, 'gtol': 1.e-10})
	#min_args=dict(method='BFGS',jac=True,options={'disp':False, 'gtol': 1.e-10})

	# define nested hierarchy of models (model name, model parameters, parent models)
	rna_pars=['log_a0','log_b0','log_c0']
	ribo_pars=rna_pars+['log_d0']
	models=OrderedDict([(0,OrderedDict([('ribo_all',(rna_pars+['log_d0'],[])),\
										('rna_all',(['log_a0','log_b0','log_c0'],[]))])),\
						(1,OrderedDict([('ribo_0',(rna_pars+['log_d0'],['ribo_all'])),\
										('rna_0',(['log_a0','log_b0','log_c0'],['rna_all']))])),\
						(2,OrderedDict([('ribo_a',(ribo_pars+['alpha'],['ribo_0'])),\
										('ribo_b',(ribo_pars+['beta'],['ribo_0'])),\
										('ribo_c',(ribo_pars+['gamma'],['ribo_0'])),\
										('ribo_d',(ribo_pars+['delta'],['ribo_0'])),\
										('rna_a',(rna_pars+['alpha'],['rna_0'])),\
										('rna_b',(rna_pars+['beta'],['rna_0'])),\
										('rna_c',(rna_pars+['gamma'],['rna_0']))])),\
						(3,OrderedDict([('ribo_ab',(ribo_pars+['alpha','beta'],['ribo_a','ribo_b'])),\
										('ribo_ac',(ribo_pars+['alpha','gamma'],['ribo_a','ribo_c'])),\
										('ribo_ad',(ribo_pars+['alpha','delta'],['ribo_a','ribo_d'])),\
										('ribo_bc',(ribo_pars+['beta','gamma'],['ribo_b','ribo_c'])),\
										('ribo_bd',(ribo_pars+['beta','delta'],['ribo_b','ribo_d'])),\
										('ribo_cd',(ribo_pars+['gamma','delta'],['ribo_c','ribo_d'])),\
										('rna_ab',(rna_pars+['alpha','beta'],['rna_a','rna_b'])),\
										('rna_ac',(rna_pars+['alpha','gamma'],['rna_a','rna_c'])),\
										('rna_bc',(rna_pars+['beta','gamma'],['rna_b','rna_c']))])),\
						(4,OrderedDict([('ribo_abc',(ribo_pars+['alpha','beta','gamma'],['ribo_ab','ribo_ac','ribo_bc'])),\
										('ribo_abd',(ribo_pars+['alpha','beta','delta'],['ribo_ab','ribo_ad','ribo_bd'])),\
										('ribo_acd',(ribo_pars+['alpha','gamma','delta'],['ribo_ac','ribo_ad','ribo_cd'])),\
										('ribo_bcd',(ribo_pars+['beta','gamma','delta'],['ribo_bc','ribo_bd','ribo_cd'])),\
										('rna_abc',(rna_pars+['alpha','beta','gamma'],['rna_ab','rna_ac','rna_bc']))])),\
						(5,OrderedDict([('ribo_abcd',(ribo_pars+['alpha','beta','gamma','delta'],['ribo_abc','ribo_abd','ribo_acd','ribo_bcd']))]))])

	if maxlevel is not None:
		nlevels=maxlevel+1
	else:
		nlevels=len(models)

	model_pars=dict((m,v[0]) for l in range(nlevels) for m,v in models[l].iteritems())

	results=defaultdict(dict)
	all_results=defaultdict(dict)
	print >> sys.stderr, '\n[RNAkira] running initial fits,',

	if priors is None:
		# initialize priors with some empirical data (use 45% reasonably expressed genes)
		print >> sys.stderr, 'estimating empirical priors'
		take_RNA=((values['mean','unlabeled-total'] > values['mean','unlabeled-total'].quantile(.5)) & \
				  (values['mean','unlabeled-total'] < values['mean','unlabeled-total'].quantile(.95))).any(axis=1)
		take_ribo=take_RNA & use_ribo & ((values['mean','ribo'] > values['mean','ribo'].quantile(.5)) & \
										 (values['mean','ribo'] < values['mean','ribo'].quantile(.95))).any(axis=1)
		log_a_est=np.log(values['mean','elu-total']/T)[take_RNA].values.flatten()
		log_b_est=np.log(values['mean','elu-total']/T/
						 (values['mean','unlabeled-total']-values['mean','unlabeled-precursor']))[take_RNA].values.flatten()
		log_c_est=np.log(values['mean','elu-total']/T/\
						 values['mean','unlabeled-precursor'])[take_RNA].values.flatten()
		log_d_est=np.log(values['mean','ribo']/(values['mean','unlabeled-total']-values['mean','unlabeled-precursor']))[take_ribo].values.flatten()

		model_priors=pd.DataFrame([trim_mean_std(x) for x in [log_a_est,log_b_est,log_c_est,log_d_est]],\
								  columns=['mu','std'],index=['log_a0','log_b0','log_c0','log_d0'])

		if model_priors.isnull().any().any():
			raise Exception("could not estimate finite model priors!")

	else:
		print >> sys.stderr, 'using given priors'
		model_priors=priors

	niter=0
	level=0
	while True:

		niter+=1

		print >> sys.stderr, '   est. priors for log_a: {0:.2g}/{1:.2g}, log_b: {2:.2g}/{3:.2g}, log_c: {4:.2g}/{5:.2g}, log_d: {6:.2g}/{7:.2g}'.format(*model_priors.ix[:4].values.flatten())

		nfits=0
		# do this for each gene
		for gene,vals in values.iterrows():

			print >> sys.stderr, '   iter {0}, {1} fits\r'.format(niter,nfits),

			if use_ribo[gene]:
				model='ribo_all'
				cols=ribo_cols
			else:
				model='rna_all'
				cols=rna_cols

			model_pars,parent_models=models[level][model]

			obs_mean=vals['mean'].unstack(level=0)[cols].stack().values.reshape((len(time_points),nreps,len(cols)))
			obs_std=vals['std'].unstack(level=0)[cols].stack().values.reshape((len(time_points),nreps,len(cols)))

			result=fit_model (obs_mean, obs_std, T, time_points, model_priors, None, model, model_pars, use_ribo[gene], min_args)

			results[gene][level]=result
			nfits+=1

		model_pars,_=models[0]['ribo_all']
		new_priors=pd.DataFrame([trim_mean_std(np.array([results[gene][level]['est_pars'][x] for gene in values.index \
														 if results[gene][level]['success'] and x in results[gene][level]['est_pars']])) for x in model_pars],\
								columns=['mu','std'],index=model_pars)

		if new_priors.isnull().any().any():
			raise Exception("could not estimate finite model priors!")

		prior_diff=((model_priors-new_priors)**2).sum().sum()/(new_priors**2).sum().sum()

		print >> sys.stderr, '   iter {0}: {1} fits, prior_diff: {2:.2g}'.format(niter,nfits,prior_diff)

		if prior_diff > 1.e-3 and niter < 10:
			model_priors=new_priors
		else:
			break

	print >> sys.stderr, '\n[RNAkira] running model selection'

	# now fit the rates using models of increasing complexity

	for level in range(1,nlevels):

		level_results=defaultdict(dict)
		for model,(model_pars,parent_models) in models[level].iteritems():

			model_results=dict()
			nfits=0
			# do this for each gene
			for gene,vals in values[use_ribo if 'ribo' in model else ~use_ribo].iterrows():
				
				if use_ribo[gene]:
					cols=ribo_cols
				else:
					cols=rna_cols

				print >> sys.stderr, '   model: {0}, {1} fits\r'.format(model,nfits),

				obs_mean=vals['mean'].unstack(level=0)[cols].stack().values.reshape((len(time_points),nreps,len(cols)))
				obs_std=vals['std'].unstack(level=0)[cols].stack().values.reshape((len(time_points),nreps,len(cols)))

				# use initial estimate from previous level
				if level-1 in results[gene] and results[gene][level-1]['model'] in parent_models:
					parent=results[gene][level-1]
				else:
					continue

				# only compute this model if the parent model was significant
				if level > 1 and not parent['LRT-q'] < sig_level:
					continue

				result=fit_model (obs_mean, obs_std, T, time_points, model_priors, parent, model, model_pars, use_ribo[gene], min_args)

				model_results[gene]=result
				level_results[gene][model]=result
				nfits+=1

			pvals=dict((gene,v['LRT-p']) for gene,v in model_results.iteritems() if 'LRT-p' in v)
			qvals=dict(zip(pvals.keys(),p_adjust_bh(pvals.values())))
			for gene,q in qvals.iteritems():
				model_results[gene]['LRT-q']=q
			nsig=sum(q < sig_level for q in qvals.values())
			if level > 1:
				print >> sys.stderr, '   model: {0}, {1} fits ({2} improved at FDR {3:.2g})'.format(model,nfits,nsig,sig_level)
			else:
				print >> sys.stderr, '   model: {0}, {1} fits ({2} insufficient at FDR {3:.2g})'.format(model,nfits,nsig,sig_level)

		print >> sys.stderr, '   selecting best models at level {0}'.format(level)
		for gene in values.index:
			all_results[gene][level]=level_results[gene]
			if len(level_results[gene]) > 0:
				results[gene][level]=min(level_results[gene].values(),key=lambda x: x['LRT-q'])

	for gene in values.index:
		max_level=max(results[gene].keys())
		if max_level > 0:
			max_fit=results[gene][max_level]
			pval=scipy.stats.chisqprob(2*np.abs(results[gene][0]['L']-max_fit['L']),np.abs(results[gene][0]['npars']-max_fit['npars']))
			results[gene][0]['LRT-p']=(pval if np.isfinite(pval) else 1)
		else:
			results[gene][0]['LRT-p']=0

	pvals=dict((gene,v[0]['LRT-p']) for gene,v in results.iteritems() if 'LRT-p' in v[0])
	qvals=dict(zip(pvals.keys(),p_adjust_bh(pvals.values())))
	nsig=sum(q < sig_level for q in qvals.values())
	for gene,q in qvals.iteritems():
		results[gene][0]['LRT-q']=q
	print >> sys.stderr, '   model: all, {0} improved at FDR {1:.2g}\n'.format(nsig,sig_level)

	print >> sys.stderr, '[RNAkira] done'

	return dict((k,v.values()) for k,v in results.iteritems())

def collect_results (results, time_points, sig_level=0.01):

	""" helper routine to put RNAkira results into a DataFrame """

	output=dict()
	for gene,res in results.iteritems():

		# first get initial fits
		initial_fit=res[0]
		pars=initial_fit['est_pars']
		errs=initial_fit['est_err']
		tmp=[('initial_synthesis_t{0}'.format(t),np.exp(pars.ix[t,'log_a0'])) for t in time_points]+\
			[('initial_synthesis_err_t{0}'.format(t),errs.ix[t,'log_a0']*np.exp(pars.ix[t,'log_a0'])) for t in time_points]+\
			[('initial_degradation_t{0}'.format(t),np.exp(pars.ix[t,'log_b0'])) for t in time_points]+\
			[('initial_degradation_err_t{0}'.format(t),errs.ix[t,'log_b0']*np.exp(pars.ix[t,'log_b0'])) for t in time_points]+\
			[('initial_processing_t{0}'.format(t),np.exp(pars.ix[t,'log_c0'])) for t in time_points]+\
			[('initial_processing_err_t{0}'.format(t),errs.ix[t,'log_c0']*np.exp(pars.ix[t,'log_c0'])) for t in time_points]
		if 'log_d0' in pars:
			tmp+=[('initial_translation_t{0}'.format(t),np.exp(pars.ix[t,'log_d0'])) for t in time_points]+\
				[('initial_translation_err_t{0}'.format(t),errs.ix[t,'log_d0']*np.exp(pars.ix[t,'log_d0'])) for t in time_points]
		else:
			tmp+=[('initial_translation_t{0}'.format(t),np.nan) for t in time_points]+\
				[('initial_translation_err_t{0}'.format(t),np.nan) for t in time_points]
		tmp+=[('initial_logL',initial_fit['L']),\
				  ('initial_fit_success',initial_fit['success']),\
				  ('initial_pval',initial_fit['LRT-p'] if 'LRT-p' in initial_fit else np.nan),\
				  ('initial_qval',initial_fit['LRT-q'] if 'LRT-q' in initial_fit else np.nan)]

		# take best significant model or constant otherwise
		best_fit=filter(lambda x: (x['model'].endswith('0') or x['LRT-q'] < sig_level),res)[-1]
		pars=best_fit['est_pars']
		errs=best_fit['est_err']
		if 'log_d0' in pars:
			(log_a,log_a_err),(log_b,log_b_err),(log_c,log_c_err),(log_d,log_d_err)=get_rates(time_points,pars,errs=errs)
		else:
			(log_a,log_a_err),(log_b,log_b_err),(log_c,log_c_err)=get_rates(time_points,pars,errs=errs)
			log_d,log_d_err=np.nan*np.ones(len(time_points)),np.nan*np.ones(len(time_points))
		tmp+=[('modeled_synthesis_t{0}'.format(t),np.exp(log_a[i])) for i,t in enumerate(time_points)]+\
			[('modeled_synthesis_err_t{0}'.format(t),log_a_err[i]*np.exp(log_a[i])) for i,t in enumerate(time_points)]+\
			[('modeled_degradation_t{0}'.format(t),np.exp(log_b[i])) for i,t in enumerate(time_points)]+\
			[('modeled_degradation_err_t{0}'.format(t),log_b_err[i]*np.exp(log_b[i])) for i,t in enumerate(time_points)]+\
			[('modeled_processing_t{0}'.format(t),np.exp(log_c[i])) for i,t in enumerate(time_points)]+\
			[('modeled_processing_err_t{0}'.format(t),log_c_err[i]*np.exp(log_c[i])) for i,t in enumerate(time_points)]+\
			[('modeled_translation_t{0}'.format(t),np.exp(log_d[i])) for i,t in enumerate(time_points)]+\
			[('modeled_translation_err_t{0}'.format(t),log_d_err[i]*np.exp(log_d[i])) for i,t in enumerate(time_points)]+\
			[('synthesis_log2FC',pars['alpha']/np.log(2) if 'alpha' in pars else 0),\
			 ('synthesis_log2FC_err',errs['alpha']/np.log(2) if 'alpha' in errs else 0),\
			 ('degradation_log2FC',pars['beta']/np.log(2) if 'beta' in pars else 0),\
			 ('degradation_log2FC_err',errs['beta']/np.log(2) if 'beta' in errs else 0),\
			 ('processing_log2FC',pars['gamma']/np.log(2) if 'gamma' in pars else 0),\
			 ('processing_log2FC_err',errs['gamma']/np.log(2) if 'gamma' in errs else 0),\
			 ('translation_log2FC',pars['delta']/np.log(2) if 'delta' in pars else 0),\
			 ('translation_log2FC_err',errs['delta']/np.log(2) if 'delta' in errs else 0),\
			 ('modeled_logL',best_fit['L']),\
			 ('modeled_fit_success',best_fit['success']),\
			 ('modeled_pval',best_fit['LRT-p'] if best_fit['model']!='0' else np.nan),\
			 ('modeled_qval',best_fit['LRT-q'] if best_fit['model']!='0' else np.nan),\
			 ('best_model',best_fit['model'])]

		output[gene]=OrderedDict(tmp)

	return pd.DataFrame.from_dict(output,orient='index')

def correct_TPM (TPM, samples, gene_stats):

	""" corrects 4sU incorporation bias and fixes library size normalization of TPM values """

	print >> sys.stderr, '\n[correct_TPM] correcting for 4sU incorporation bias and normalizing by linear regression'

	# select reliable genes with decent mean expression level in unlabeled mature
	reliable_genes=(gene_stats['gene_type']=='protein_coding') & (TPM['unlabeled-mature'] > TPM['unlabeled-mature'].dropna().quantile(.9)).any(axis=1)

	# collect corrected values
	corrected_TPM=pd.DataFrame(index=TPM.index)

	print >> sys.stderr, '   t =',

	for sample in samples:

		t,r=sample.split('-')

		print >> sys.stderr, '{0} ({1})'.format(t,r),

		log2_elu_ratio=np.log2(TPM['elu-mature',sample]/TPM['unlabeled-mature',sample])
		log2_FT_ratio=np.log2(TPM['flowthrough-mature',sample]/TPM['unlabeled-mature',sample])

		ok=np.isfinite(log2_elu_ratio) & reliable_genes
		lowess_elu=statsmodels.nonparametric.smoothers_lowess.lowess(log2_elu_ratio[ok],gene_stats['exon_ucount'][ok],frac=0.66,it=1).T
		interp_elu=scipy.interpolate.interp1d(lowess_elu[0],lowess_elu[1],bounds_error=False)
		log2_elu_ratio_no_bias=log2_elu_ratio-interp_elu(gene_stats['exon_ucount'])

		elu_percentiles=np.percentile(log2_elu_ratio_no_bias[ok],[5,95])
		FT_percentiles=np.percentile(log2_FT_ratio[ok],[5,95])
		ok=np.isfinite(log2_elu_ratio_no_bias) & np.isfinite(log2_FT_ratio) & reliable_genes & \
			(log2_elu_ratio_no_bias > elu_percentiles[0]) & (log2_elu_ratio_no_bias < elu_percentiles[1]) & \
			(log2_FT_ratio > FT_percentiles[0]) & (log2_FT_ratio < FT_percentiles[1])
		slope,intercept,_,_,_=scipy.stats.linregress(2**log2_elu_ratio_no_bias[ok],2**log2_FT_ratio[ok])

		if intercept < 0 or slope > 0:
			raise Exception('invalid slope ({0:.2g}) or intercept ({1:.2g})'.format(slope,intercept))

		# corrected values for precursors
		corrected_TPM[('unlabeled-precursor',t,r)]=TPM['unlabeled-precursor',sample]
		corrected_TPM[('elu-precursor',t,r)]=TPM['elu-precursor',sample]*2**(-np.log2(-intercept/slope))
		corrected_TPM[('flowthrough-precursor',t,r)]=TPM['flowthrough-precursor',sample]*2**(-np.log2(intercept))

		# corrected values for total (mature+precursor) after correcting for 4sU incorporation bias in mature
		corrected_TPM[('unlabeled-total',t,r)]=TPM['unlabeled-mature',sample]+corrected_TPM[('unlabeled-precursor',t,r)]
		corrected_TPM[('elu-total',t,r)]=TPM['unlabeled-mature',sample]*2**(log2_elu_ratio_no_bias-np.log2(-intercept/slope))\
										   +corrected_TPM[('elu-precursor',t,r)]
		corrected_TPM[('flowthrough-total',t,r)]=TPM['unlabeled-mature',sample]*2**(log2_FT_ratio-np.log2(intercept))\
											 +corrected_TPM[('flowthrough-precursor',t,r)]

		# uncorrected values for ribo
		corrected_TPM[('ribo',t,r)]=TPM['ribo',sample]

	corrected_TPM.columns=pd.MultiIndex.from_tuples(corrected_TPM.columns)

	print >> sys.stderr, '\n'

	return corrected_TPM

def estimate_dispersion (TPM, cols): 

	""" estimates dispersion by simply smoothing the mean-variance plot taken across all samples to the columns given in vols """

	std_TPM=[]
	print >> sys.stderr, '\n[estimate_dispersion] averaging samples:\n   ',
	for c in cols:
		print >> sys.stderr, c, 
		means=TPM[c].mean(axis=1,level=0)
		CV=TPM[c].std(axis=1,level=0)/means
		ok=np.isfinite(means).all(axis=1) & np.isfinite(CV).all(axis=1)
		lowess=statsmodels.nonparametric.smoothers_lowess.lowess(CV[ok].values.flatten(),means[ok].values.flatten(),frac=0.5,it=1).T
		interp=scipy.interpolate.interp1d(lowess[0],lowess[1],bounds_error=False)
		std_TPM.append(TPM[c].apply(interp)*TPM[c])

	std_TPM=pd.concat(std_TPM,axis=1,keys=cols)

	print >> sys.stderr, '\n'

	return pd.concat([TPM,std_TPM],axis=1,keys=['mean','std']).sort_index(axis=1)

def get_TPM (fc_table):
	
	""" computes RPKM and then TPM values from featureCounts output """

	rpkm=fc_table.ix[:,1:].divide(fc_table['length']/1.e3,axis=0).divide(fc_table.ix[:,1:].sum(axis=0)/1.e6,axis=1)
	return rpkm.divide(rpkm.sum(axis=0)/1.e6,axis=1)

if __name__ == '__main__':

	parser=OptionParser()
	parser.add_option('-g','--gene_stats',dest='gene_stats',help="gene stats file (created by prepare_annotation.py)")
	parser.add_option('-i','--input_TPM',dest='input_TPM',help='csv file with corrected TPM values')
	parser.add_option('-e','--elu_introns',dest='elu_introns',help="featureCounts output for eluate RNA mapped to introns")
	parser.add_option('-E','--elu_exons',dest='elu_exons',help="featureCounts output for eluate RNA mapped to exons")
	parser.add_option('-f','--flowthrough_introns',dest='flowthrough_introns',help="featureCounts output for flowthrough RNA mapped to introns")
	parser.add_option('-F','--flowthrough_exons',dest='flowthrough_exons',help="featureCounts output for flowthrough RNA mapped to exons")
	parser.add_option('-r','--ribo_CDS',dest='ribo',help="featureCounts output for RPF mapped to CDS")
	parser.add_option('-u','--unlabeled_introns',dest='unlabeled_introns',help="featureCounts output for unlabeled RNA mapped to introns")
	parser.add_option('-U','--unlabeled_exons',dest='unlabeled_exons',help="featureCounts output for unlabeled RNA mapped to exons")
	parser.add_option('-t','--time_points',dest='time_points',help="comma-separated list of time points (integer or floating-point), MUST correspondg to the data columns in the featureCount outputs and be the same for all fractions")
	parser.add_option('-T','--labeling_time',dest='T',help="labeling time",type=float)
	parser.add_option('-o','--outfile',dest='outf',help="output file (default: stdout)")
	parser.add_option('','--alpha',dest='alpha',help="FDR cutoff (default: 0.05)",default=0.05,type=float)
	parser.add_option('','--maxlevel',dest='maxlevel',help="max level to test (default: 5)",default=5,type=int)
	parser.add_option('','--min_TPM_total',dest='min_TPM_total',help="min TPM for total (default: .1)",default=.1,type=float)
	parser.add_option('','--min_TPM_precursor',dest='min_TPM_precursor',help="min TPM for precursor (default: .001)",default=.001,type=float)
	parser.add_option('','--min_TPM_ribo',dest='min_TPM_ribo',help="min TPM for ribo (default: .1)",default=.1,type=float)

	options,args=parser.parse_args()

	try:
		time_points=options.time_points.split(',')
	except:
		raise Exception("couldn't parse time points {0}".format(options.time_points))

	if len(set(Counter(time_points).values())) > 1:
		raise Exception("unequal number of replicates at timepoints; can't deal with that")
	nreps=Counter(time_points)[time_points[0]]

	samples=[]
	for t in time_points:
		samples.append(t+'-Rep'+str(sum(x.split('-')[0]==t for x in samples)+1))
	nsamples=len(samples)

	if options.input_TPM is not None:
		
		print >> sys.stderr, '\n[main] reading corrected TPM values from '+options.input_TPM
		print >> sys.stderr, '       ignoring options -geEfFruU'
		try:
			TPM=pd.read_csv(options.input_TPM,index_col=0,header=range(3))
		except:
			raise Exception("couldn't read TPM file "+options.input_TPM)

	else:

		print >> sys.stderr, "\n[main] reading count data"

		try:
			print >> sys.stderr, '   elu-introns:\t\t'+options.elu_introns
			elu_introns=pd.read_csv(options.elu_introns,sep='\t',comment='#',index_col=0,header=0).ix[:,4:]
			elu_introns.columns=['length']+samples
		except:
			raise Exception("couldn't read input file "+options.elu_introns)

		try:
			print >> sys.stderr, '   elu-exons:\t\t'+options.elu_exons
			elu_exons=pd.read_csv(options.elu_exons,sep='\t',comment='#',index_col=0,header=0).ix[:,4:]
			elu_exons.columns=['length']+samples
		except:
			raise Exception("couldn't read input file "+options.elu_exons)

		try:
			print >> sys.stderr, '   flowthrough-introns:\t'+options.flowthrough_introns
			flowthrough_introns=pd.read_csv(options.flowthrough_introns,sep='\t',comment='#',index_col=0,header=0).ix[:,4:]
			flowthrough_introns.columns=['length']+samples
		except:
			raise Exception("couldn't read input file "+options.flowthrough_introns)

		try:
			print >> sys.stderr, '   flowthrough-exons:\t'+options.flowthrough_exons
			flowthrough_exons=pd.read_csv(options.flowthrough_exons,sep='\t',comment='#',index_col=0,header=0).ix[:,4:]
			flowthrough_exons.columns=['length']+samples
		except:
			raise Exception("couldn't read input file "+options.flowthrough_exons)

		try:
			print >> sys.stderr, '   ribo:\t\t'+options.ribo
			ribo=pd.read_csv(options.ribo,sep='\t',comment='#',index_col=0,header=0).ix[:,4:]
			ribo.columns=['length']+samples
		except:
			raise Exception("couldn't read input file "+options.ribo)

		try:
			print >> sys.stderr, '   unlabeled-introns:\t'+options.unlabeled_introns
			unlabeled_introns=pd.read_csv(options.unlabeled_introns,sep='\t',comment='#',index_col=0,header=0).ix[:,4:]
			unlabeled_introns.columns=['length']+samples
		except:
			raise Exception("couldn't read input file "+options.unlabeled_introns)

		try:
			print >> sys.stderr, '   unlabeled-exons:\t'+options.unlabeled_exons
			unlabeled_exons=pd.read_csv(options.unlabeled_exons,sep='\t',comment='#',index_col=0,header=0).ix[:,4:]
			unlabeled_exons.columns=['length']+samples
		except:
			raise Exception("couldn't read input file "+options.unlabeled_exons)

		print >> sys.stderr, "\n[main] merging count values and computing TPM using gene stats from "+options.gene_stats

		cols=['elu-precursor','elu-mature','flowthrough-precursor','flowthrough-mature','ribo','unlabeled-precursor','unlabeled-mature']
		TPM=pd.concat(map(get_TPM,[elu_introns,elu_exons,flowthrough_introns,flowthrough_exons,ribo,unlabeled_introns,unlabeled_exons]),axis=1,keys=cols)

		TPM.to_csv('TPM.csv')

		try:
			gene_stats=pd.read_csv(options.gene_stats,index_col=0,header=0).loc[TPM.index]
		except:
			raise Exception("couldn't read gene stats file "+options.gene_stats)

		print >> sys.stderr, '\n[main] correcting TPM values'

		TPM=correct_TPM (TPM, samples, gene_stats)

		TPM.to_csv("corrected_TPM.csv")

	print >> sys.stderr, '\n[main] estimating dispersion'

	cols=['elu-precursor','elu-total','flowthrough-precursor','flowthrough-total','ribo','unlabeled-precursor','unlabeled-total']
	TPM=estimate_dispersion (TPM, cols)

	# select genes based on TPM cutoffs for total, precursor in any of the time points
	ok=(TPM['mean','unlabeled-total'] > options.min_TPM_total).any(axis=1) &\
		(TPM['mean','unlabeled-precursor'] > options.min_TPM_precursor).any(axis=1)
	
	# take only those genes
	TPM_here=TPM[ok].fillna(0)

	print >> sys.stderr, '[main] running RNAkira'
	results=RNAkira(TPM_here, options.T, sig_level=options.alpha, \
					min_TPM_ribo=options.min_TPM_ribo,maxlevel=options.maxlevel)

	print >> sys.stderr, '\n[main] collecting output'
	output=collect_results(results, time_points, sig_level=options.alpha)

	try:
		outf=open(options.outf,'w');
		print >> sys.stderr, '\n   writing results to {0}'.format(options.outf)
	except:
		outf=sys.stdout
		print >> sys.stderr, '\n   writing results to stdout'

	output.to_csv(outf)

