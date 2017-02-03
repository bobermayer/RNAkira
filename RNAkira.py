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
		(taken and modified for nan values from here: http://stackoverflow.com/questions/7450957/how-to-implement-rs-p-adjust-in-python) """

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

	""" given synthesis, degradation, processing rates and potentially translation efficiencies x=(a,b,c,d) and labeling time T
	returns instantaneous steady-state values for elu-precursor,elu-mature,flowthrough-precursor,flowthrough-mature,unlabeled-precursor,unlabeled-mature
	ribo if use_ribo is set
	includes derivatives w.r.t. log(a),log(b),log(c) and maybe log(d) if use_deriv is set """

	if use_ribo:
		a,b,c,d=np.exp(x)
	else:
		a,b,c=np.exp(x)

	# for elu-precursor, elu-mature, flowthrough-precursor, flowthrough-mature, unlabeled-precursor, unlabeled-mature
	exp_mean=[a*(1.-np.exp(-c*T))/c,\
			  a*(b*(1.-np.exp(-c*T))-c*(1.-np.exp(-b*T)))/(b*(b-c)),\
			  a*np.exp(-c*T)/c,\
			  a*(b*np.exp(-c*T)-c*np.exp(-b*T))/(b*(b-c)),\
			  a/c,\
			  a/b]

	# add ribo value
	if use_ribo:
		exp_mean=exp_mean[:4]+[d*a/b]+exp_mean[4:]

	if use_deriv:
		# get the derivatives w.r.t. log(a),log(b),log(c)
		exp_mean_deriv=[[a*(1.-np.exp(-c*T))/c,\
						 a*(b*(1.-np.exp(-c*T))-c*(1.-np.exp(-b*T)))/(b*(b-c)),\
						 a*np.exp(-c*T)/c,\
						 a*(b*np.exp(-c*T)-c*np.exp(-b*T))/(b*(b-c)),\
						 a/c,\
						 a/b],\
						[0, 
						 a*(b*(b-c)*(1.-np.exp(-c*T)-c*T*np.exp(-b*T))-\
							(2*b-c)*(b*(1.-np.exp(-c*T))-c*(1.-np.exp(-b*T))))/(b*(b-c)**2),\
						 0,\
						 a*(b*(b-c)*(T*c*np.exp(-b*T)+np.exp(-c*T))-(2*b-c)*(b*np.exp(-c*T)-c*np.exp(-b*T)))/(b*(b-c)**2),\
						 0,\
						 -a/b],\
						[a*(np.exp(-c*T)*(1.+c*T)-1)/c,\
						 a*c*((b-c)*(b*T*np.exp(-c*T)-1.+np.exp(-b*T))+\
							  (b*(1.-np.exp(-c*T))-c*(1.-np.exp(-b*T))))/(b*(b-c)**2),\
						 -a*np.exp(-c*T)*(1.+c*T)/c,\
						 a*c*((b-c)*(-np.exp(-b*T)-T*b*np.exp(-c*T))+(b*np.exp(-c*T)-c*np.exp(-b*T)))/(b*(b-c)**2),\
						 -a/c,\
						 0]]

		if use_ribo:
			# add derivatives of ribo values and derivatives w.r.t. log(d)
			ribo_vals=[[d*a/b],[-d*a/b],[0]]
			exp_mean_deriv=[exp_mean_deriv[i][:4]+ribo_vals[i]+exp_mean_deriv[i][4:] for i in range(3)]+[[0,0,0,0,d*a/b,0,0]]

	if use_deriv:
		return (np.array(exp_mean),map(np.array,exp_mean_deriv))
	else:
		return np.array(exp_mean)

def get_rates(time_points, pars):

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

	return rates

		
def steady_state_log_likelihood (x, obs_mean, obs_std, nf, T, time_points, prior_mu, prior_std, model_pars, use_ribo, statsmodel, use_deriv):

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

		if statsmodel=='gaussian':
			diff=obs_mean[i]-exp_mean/nf[i]
			fun+=np.sum(-.5*(diff/obs_std[i])**2-.5*np.log(2.*np.pi)-np.log(obs_std[i]))
			if use_deriv:
				grad+=np.dot(exp_mean_deriv,np.sum(diff/obs_std[i]**2/nf[i],axis=0))
		elif statsmodel=='nbinom':
			# this doesn't work, n can be smaller than zero and p is not between 0 and 1
			mu=exp_mean/nf[i]
			nn=mu**2/(obs_std[i]**2-mu)
			pp=mu/obs_std[i]**2
			fun+=np.sum(scipy.stats.nbinom.logpmf(obs_mean[i],nn,pp))
			if use_deriv:
				tmp=((np.log(pp)-scipy.special.psi(nn)+scipy.special.psi(obs_mean[i]+nn))*(nf[i]*obs_std[i]**4/(obs_std[i]**2*nf[i]-exp_mean)**2-1./nf[i])
					 +(nn/pp-obs_mean[i]/(1.-pp))/(nf[i]*obs_std[i]))
				grad+=np.dot(exp_mean_deriv,np.sum(tmp,axis=0))

	if use_deriv:
		# return negative log likelihood and gradient
		return (-fun,-grad)
	else:
		# return negative log likelihood
		return -fun

def fit_model (obs_mean, obs_std, nf, T, time_points, model_priors, parent, model, model_pars, use_ribo, statsmodel, min_args):

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
		args=(obs_mean, obs_std, nf, T, time_points, model_priors['mu'].values[:nrates], model_priors['std'].values[:nrates],\
			  model_pars, use_ribo, statsmodel, min_args['jac'])

		# test gradient against numerical difference if necessary
		if test_gradient:

			pp=initial_estimate
			eps=1.e-6*np.abs(initial_estimate)+1.e-8
			fun,grad=steady_state_log_likelihood(pp,*args)
			my_grad=np.array([(steady_state_log_likelihood(np.array(list(pp[:i])+[pp[i]+eps[i]]+list(pp[i+1:])),*args)[0]\
							   -steady_state_log_likelihood(np.array(list(pp[:i])+[pp[i]-eps[i]]+list(pp[i+1:])),*args)[0])/(2*eps[i]) for i in range(len(pp))])
			diff=np.sum((grad-my_grad)**2)/np.sum(grad**2)
			if diff > 1.e-8:
				raise Exception('\n!!! gradient diff: {0:.3e}'.format(diff))

		# perform minimization
		res=scipy.optimize.minimize(steady_state_log_likelihood, \
									initial_estimate,\
									args=args, \
									**min_args)

		# collect results
		result=dict(est_pars=pd.Series(res.x,index=model_pars),\
					L=-res.fun,\
					success=res.success,\
					message=res.message,\
					npars=len(model_pars),
					model=model)

	else:

		# in the 'all' models, treat every time point independently
		x=[]
		L=0
		npars=0
		success=True
		messages=[]

		if parent is None:
			# take prior estimates for each time point
			log_rates=[model_priors['mu'].ix[k]*np.ones(len(time_points)) for k in range(nrates)]
		else:
			
			initial_pars=parent['est_pars']
			log_rates=get_rates(time_points,initial_pars)

		for i,t in enumerate(time_points):

			initial_estimate=np.array([lr[i] for lr in log_rates])

			args=(obs_mean[i,None], obs_std[i,None], nf[i,None], T, [t], model_priors['mu'].values[:nrates], model_priors['std'].values[:nrates],\
				  model_pars, use_ribo, statsmodel, min_args['jac'])

			# test gradient against numerical difference if necessary
			if test_gradient:

				pp=initial_estimate
				eps=1.e-6*np.abs(initial_estimate)+1.e-8
				fun,grad=steady_state_log_likelihood(pp,*args)
				my_grad=np.array([(steady_state_log_likelihood(np.array(list(pp[:i])+[pp[i]+eps[i]]+list(pp[i+1:])),*args)[0]\
								   -steady_state_log_likelihood(np.array(list(pp[:i])+[pp[i]-eps[i]]+list(pp[i+1:])),*args)[0])/(2*eps[i]) for i in range(len(pp))])
				diff=np.sum((grad-my_grad)**2)/np.sum(grad**2)
				if diff > 1.e-8:
					raise Exception('\n!!! gradient diff: {0:.3e}'.format(diff))

			res=scipy.optimize.minimize(steady_state_log_likelihood, \
										initial_estimate,\
										args=args, \
										**min_args)

			x.append(res.x)

			L+=-res.fun
			success=success & res.success
			messages.append(res.message)
			npars+=len(model_pars)

		result=dict(est_pars=pd.DataFrame(x,columns=model_pars,index=time_points),\
					L=L,\
					success=success,\
					message='|'.join(messages),\
					npars=npars,\
					model=model)

	if parent is not None:
		# calculate p-value from LRT test using chi2 distribution
		#pval=scipy.stats.chisqprob(2*np.abs(result['L']-parent['L']),np.abs(result['npars']-parent['npars']))
		pval=scipy.stats.chi2.sf(2*np.abs(result['L']-parent['L']),np.abs(result['npars']-parent['npars']))
		result['LRT-p']=(pval if np.isfinite(pval) else np.nan)

	return result

def plot_data_rates_fits (time_points, replicates, obs_vals, T, parameters, results, use_ribo, title='', priors=None, sig_level=0.01):

	""" function to plot summary of results for a specific gene """

	import matplotlib
	matplotlib.use('Agg')
	from matplotlib import pyplot as plt

	times=np.array(map(float,time_points))
	ntimes=len(times)

	def jitter():
		return np.random.normal(loc=0,scale=.1*np.min(np.diff(times)),size=len(time_points))

	cols=['elu-precursor','elu-mature','flowthrough-precursor','flowthrough-mature','unlabeled-precursor','unlabeled-mature']
	if use_ribo:
		cols=cols[:4]+['ribo']+cols[4:]
		nrates=4
	else:
		nrates=3

	ndim=len(cols)

	fig=plt.figure(figsize=(15,6))
	fig.clf()
	fig.subplots_adjust(wspace=.5,hspace=.4,left=.05,right=.98)

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
		ax[i].set_title(c)
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
		for i,p in enumerate(get_rates(time_points,vals['est_pars'])):
			qval=vals['LRT-q']
			if level >= 1:
				ax[ndim+i].plot(times,p,linestyle=('-' if qval < sig_level else '--'),label='{0} (q={1:.2g})'.format(vals['model'],qval))
			else:
				ax[ndim+i].plot(times,p,linestyle=':',label='initial')

	for i,t in enumerate(['synthesis','degradation','processing','translation'][:nrates]):
		ax[ndim+i].set_title(t+' rate')
		ax[ndim+i].set_xlabel('time')

	ax[ndim].set_ylabel('log rate')
	ax[ndim+nrates-1].legend(loc=2,frameon=False,prop={'size':10},bbox_to_anchor=(1.1,1))

	fig.suptitle(title)

def RNAkira (mean_vals, std_vals, NF, T, sig_level=0.01, min_TPM_ribo=1, maxlevel=None, priors=None, statsmodel='gaussian'):

	""" main routine in this package: given dataframe of mean and std TPM values and labeling time, 
	    estimates empirical priors and fits models of increasing complexity """

	genes=mean_vals.index
	time_points=np.unique(mean_vals.columns.get_level_values(1))
	nreps=len(np.unique(mean_vals.columns.get_level_values(2)))

	rna_cols=['elu-precursor','elu-mature','flowthrough-precursor','flowthrough-mature','unlabeled-precursor','unlabeled-mature']
	ribo_cols=rna_cols[:4]+['ribo']+rna_cols[4:]

	ndim=7
	TPM=mean_vals*NF

	use_ribo=(TPM['ribo'] > min_TPM_ribo).any(axis=1)
	use_rna=~use_ribo

	print >> sys.stderr, '\n[RNAkira] analyzing {0} genes with ribo+rna and {1} genes with rna only ({2} time points, {3} replicates)'.format(use_ribo.sum(),(~use_ribo).sum(),len(time_points),nreps)

	if statsmodel=='gaussian':
		min_args=dict(method='L-BFGS-B',jac=True,options={'disp':False, 'ftol': 1.e-15, 'gtol': 1.e-10})
	elif statsmodel=='nbinom':
		# derivative is not yet fully implemented for negative binomial
		min_args=dict(method='L-BFGS-B',jac=False,options={'disp':False, 'ftol': 1.e-15, 'gtol': 1.e-10})

	# define nested hierarchy of models (model name, model parameters, parent models)
	rna_pars=['log_a0','log_b0','log_c0']
	ribo_pars=rna_pars+['log_d0']
	models=OrderedDict([(0,OrderedDict([('ribo_all',(ribo_pars,[])),\
										('rna_all',(['log_a0','log_b0','log_c0'],[]))])),\
						(1,OrderedDict([('ribo_0',(ribo_pars,['ribo_all'])),\
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
		take_RNA=((TPM['unlabeled-mature'] > TPM['unlabeled-mature'].quantile(.5)) & \
				  (TPM['unlabeled-mature'] < TPM['unlabeled-mature'].quantile(.95))).any(axis=1)
		take_ribo=take_RNA & use_ribo & ((TPM['ribo'] > TPM['ribo'].quantile(.5)) & \
										 (TPM['ribo'] < TPM['ribo'].quantile(.95))).any(axis=1)
		log_a_est=np.log((TPM['elu-mature']+TPM['elu-precursor'])/T)[take_RNA].values.flatten()
		log_b_est=np.log((TPM['elu-mature']+TPM['elu-precursor'])/T/TPM['unlabeled-mature'])[take_RNA].values.flatten()
		log_c_est=np.log((TPM['elu-mature']+TPM['elu-precursor'])/T/\
						 TPM['unlabeled-precursor'])[take_RNA].values.flatten()
		log_d_est=np.log(TPM['ribo']/TPM['unlabeled-mature'])[take_ribo].values.flatten()

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
		for gene in genes:

			print >> sys.stderr, '   iter {0}, {1} fits\r'.format(niter,nfits),

			if use_ribo[gene]:
				model='ribo_all'
				cols=ribo_cols
			else:
				model='rna_all'
				cols=rna_cols

			model_pars,parent_models=models[level][model]
			
			# make numpy arrays out of mean and std vals for computational efficiency (therefore we need equal number of replicates for each time point!)
			obs_mean=mean_vals.loc[gene].unstack(level=0)[cols].stack().values.reshape((len(time_points),nreps,len(cols)))
			obs_std=std_vals.loc[gene].unstack(level=0)[cols].stack().values.reshape((len(time_points),nreps,len(cols)))
			nf=NF.loc[gene].unstack(level=0)[cols].stack().values.reshape((len(time_points),nreps,len(cols)))

			result=fit_model (obs_mean, obs_std, nf, T, time_points, model_priors, None, model, model_pars, use_ribo[gene], statsmodel, min_args)

			results[gene][level]=result
			nfits+=1

		model_pars,_=models[0]['ribo_all']
		new_priors=pd.DataFrame([trim_mean_std(np.array([results[gene][level]['est_pars'][x] for gene in genes \
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
			for gene in genes[use_ribo if 'ribo' in model else ~use_ribo]:
				
				if use_ribo[gene]:
					cols=ribo_cols
				else:
					cols=rna_cols

				print >> sys.stderr, '   model: {0}, {1} fits\r'.format(model,nfits),

				# make numpy arrays out of mean and std vals for computational efficiency (therefore we need equal number of replicates for each time point!)
				obs_mean=mean_vals.loc[gene].unstack(level=0)[cols].stack().values.reshape((len(time_points),nreps,len(cols)))
				obs_std=std_vals.loc[gene].unstack(level=0)[cols].stack().values.reshape((len(time_points),nreps,len(cols)))
				nf=NF.loc[gene].unstack(level=0)[cols].stack().values.reshape((len(time_points),nreps,len(cols)))

				# use initial estimate from previous level
				if level-1 in results[gene] and results[gene][level-1]['model'] in parent_models:
					parent=results[gene][level-1]
				else:
					continue

				# only compute this model if the parent model was significant
				if level > 1 and not parent['LRT-q'] < sig_level:
					continue

				result=fit_model (obs_mean, obs_std, nf, T, time_points, model_priors, parent, model, model_pars, use_ribo[gene], statsmodel, min_args)

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
		for gene in genes:
			all_results[gene][level]=level_results[gene]
			if len(level_results[gene]) > 0:
				results[gene][level]=min(level_results[gene].values(),key=lambda x: x['LRT-q'])

	for gene in genes:
		max_level=max(results[gene].keys())
		if max_level > 0:
			max_fit=results[gene][max_level]
			#pval=scipy.stats.chisqprob(2*np.abs(results[gene][0]['L']-max_fit['L']),np.abs(results[gene][0]['npars']-max_fit['npars']))
			pval=scipy.stats.chi2.sf(2*np.abs(results[gene][0]['L']-max_fit['L']),np.abs(results[gene][0]['npars']-max_fit['npars']))
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
		tmp=[('initial_synthesis_t{0}'.format(t),np.exp(pars.ix[t,'log_a0'])) for t in time_points]+\
			[('initial_degradation_t{0}'.format(t),np.exp(pars.ix[t,'log_b0'])) for t in time_points]+\
			[('initial_processing_t{0}'.format(t),np.exp(pars.ix[t,'log_c0'])) for t in time_points]
		if 'log_d0' in pars:
			tmp+=[('initial_translation_t{0}'.format(t),np.exp(pars.ix[t,'log_d0'])) for t in time_points]
		else:
			tmp+=[('initial_translation_t{0}'.format(t),np.nan) for t in time_points]
		tmp+=[('initial_logL',initial_fit['L']),\
			  ('initial_fit_success',initial_fit['success']),\
			  ('initial_pval',initial_fit['LRT-p'] if 'LRT-p' in initial_fit else np.nan),\
			  ('initial_qval',initial_fit['LRT-q'] if 'LRT-q' in initial_fit else np.nan)]

		# take best significant model or constant otherwise
		best_fit=filter(lambda x: (x['model'].endswith('0') or x['LRT-q'] < sig_level),res)[-1]
		pars=best_fit['est_pars']
		if 'log_d0' in pars:
			log_a,log_b,log_c,log_d=get_rates(time_points,pars)
		else:
			log_a,log_b,log_c=get_rates(time_points,pars)
			log_d=np.nan*np.ones(len(time_points))
		tmp+=[('modeled_synthesis_t{0}'.format(t),np.exp(log_a[i])) for i,t in enumerate(time_points)]+\
			[('modeled_degradation_t{0}'.format(t),np.exp(log_b[i])) for i,t in enumerate(time_points)]+\
			[('modeled_processing_t{0}'.format(t),np.exp(log_c[i])) for i,t in enumerate(time_points)]+\
			[('modeled_translation_t{0}'.format(t),np.exp(log_d[i])) for i,t in enumerate(time_points)]+\
			[('synthesis_log2FC',pars['alpha']/np.log(2) if 'alpha' in pars else 0),\
			 ('degradation_log2FC',pars['beta']/np.log(2) if 'beta' in pars else 0),\
			 ('processing_log2FC',pars['gamma']/np.log(2) if 'gamma' in pars else 0),\
			 ('translation_log2FC',pars['delta']/np.log(2) if 'delta' in pars else 0),\
			 ('modeled_logL',best_fit['L']),\
			 ('modeled_fit_success',best_fit['success']),\
			 ('modeled_pval',best_fit['LRT-p'] if best_fit['model']!='0' else np.nan),\
			 ('modeled_qval',best_fit['LRT-q'] if best_fit['model']!='0' else np.nan),\
			 ('best_model',best_fit['model'])]

		output[gene]=OrderedDict(tmp)

	return pd.DataFrame.from_dict(output,orient='index')

def normalize_elu_flowthrough (TPM, samples, gene_stats, fig_name=None):

	if fig_name is not None:
		import matplotlib
		matplotlib.use('Agg')
		from matplotlib import pyplot as plt
		fig=plt.figure(figsize=(8,3*len(samples)))
		fig.subplots_adjust(bottom=.02,top=.98,hspace=.4,wspace=.3)

	""" corrects 4sU incorporation bias and fixes library size normalization of TPM values """

	print >> sys.stderr, '\n[normalize_elu_flowthrough] correcting for 4sU incorporation bias and normalizing by linear regression'

	# select reliable genes with decent mean expression level in unlabeled mature
	reliable_genes=(gene_stats['gene_type']=='protein_coding') & (TPM['unlabeled-mature'] > TPM['unlabeled-mature'].dropna().quantile(.2)).any(axis=1)

	# collect correction_factors
	CF=pd.DataFrame(1,index=TPM.index,columns=TPM.columns)

	print >> sys.stderr, '   t =',

	for n,(t,r) in enumerate(samples):

		print >> sys.stderr, '{0} ({1})'.format(t,r),

		log2_elu_ratio=np.log2(TPM['elu-mature',t,r]/TPM['unlabeled-mature',t,r]).replace([np.inf,-np.inf],np.nan)
		log2_FT_ratio=np.log2(TPM['flowthrough-mature',t,r]/TPM['unlabeled-mature',t,r]).replace([np.inf,-np.inf],np.nan)

		log10_ucount=np.log10(1+gene_stats['exon_ucount'])
		ok=np.isfinite(log2_elu_ratio) & reliable_genes & np.isfinite(log10_ucount)
		log10_ucount_range=np.abs(log10_ucount[ok].max()-log10_ucount[ok].min())
		lowess=statsmodels.nonparametric.smoothers_lowess.lowess(log2_elu_ratio[ok],log10_ucount[ok],\
																 frac=0.66,it=1,delta=.001*log10_ucount_range).T
		interp_elu=scipy.interpolate.interp1d(lowess[0],lowess[1],bounds_error=False)
		log2_elu_ratio_no_bias=log2_elu_ratio-interp_elu(log10_ucount)

		elu_percentiles=np.percentile(log2_elu_ratio_no_bias[ok].dropna(),[5,95])
		FT_percentiles=np.percentile(log2_FT_ratio[ok].dropna(),[5,95])
		ok=np.isfinite(log2_elu_ratio_no_bias) & np.isfinite(log2_FT_ratio) & reliable_genes & \
			(log2_elu_ratio_no_bias > elu_percentiles[0]) & (log2_elu_ratio_no_bias < elu_percentiles[1]) & \
			(log2_FT_ratio > FT_percentiles[0]) & (log2_FT_ratio < FT_percentiles[1])
		slope,intercept,_,_,_=scipy.stats.linregress(2**log2_elu_ratio_no_bias[ok],2**log2_FT_ratio[ok])

		if intercept < 0 or slope > 0:
			raise Exception('invalid slope ({0:.2g}) or intercept ({1:.2g})'.format(slope,intercept))

		# normalization factors for introns
		CF['elu-precursor',t,r]=2**(-np.log2(-intercept/slope))
		CF['flowthrough-precursor',t,r]=2**(-np.log2(intercept))
		# correct for 4sU incorporation bias
		CF['elu-mature',t,r]=2**(log2_elu_ratio_no_bias.fillna(0)-np.log2(-intercept/slope))
		CF['flowthrough-mature',t,r]=2**(log2_FT_ratio.fillna(0)-np.log2(intercept))
		# no correction for unlabeled + ribo
		CF['unlabeled-precursor',t,r]=1
		CF['unlabeled-mature',t,r]=1
		CF['ribo',t,r]=1

		if fig_name is not None:

			ok=np.isfinite(log2_elu_ratio) & reliable_genes & np.isfinite(log10_ucount)

			ax=fig.add_subplot(len(samples),2,2*n+1)
			ax.hexbin(gene_stats['exon_ucount'][ok],log2_elu_ratio[ok],bins='log',extent=(0,5000,-2,2.5),lw=0,cmap=plt.cm.Greys,vmin=-1,mincnt=1)
			ax.plot(10**np.arange(0,4,.1),interp_elu(np.arange(0,4,.1)),'r-')
			ax.set_xlim([0,5000])
			ax.set_ylim([-2,2.5])
			ax.set_title('{0} {1}'.format(t,r,ok.sum()))
			ax.set_ylabel('log2 elu/unlabeled')
			ax.set_xlabel('# U residues')

			ax=fig.add_subplot(len(samples),2,2*n+2)
			ax.hexbin(2**log2_elu_ratio_no_bias[ok],2**log2_FT_ratio[ok],bins='log',lw=0,extent=(-.1,5,-.1,3),cmap=plt.cm.Greys,vmin=-1,mincnt=1)
			ax.plot(np.arange(0,5,.01),intercept+slope*np.arange(0,5,.01),'r-')
			ax.set_xlim([-.1,5])
			ax.set_ylim([-.1,3])
			ax.set_title('{0} {1}'.format(t,r,ok.sum()))
			ax.set_xlabel('elu/unlabeled')
			ax.set_ylabel('FT/unlabeled')

	if fig_name is not None:
		print >> sys.stderr, '[normalize_elu_flowthrough] saving figure to {0}'.format(fig_name)
		fig.savefig(fig_name)

	print >> sys.stderr, '\n'

	return CF

def estimate_dispersion (TPM, cols, fig_name=None, disp_weight=1.8):

	""" estimates dispersion by a weighted average of the calculated std dev and a smoothened std dev from the mean-variance plot """

	if fig_name is not None:
		import matplotlib
		matplotlib.use('Agg')
		from matplotlib import pyplot as plt
		fig=plt.figure(figsize=(3*len(cols),3))
		fig.subplots_adjust(left=.05,right=.98,hspace=.4,wspace=.3,bottom=.15)

	std_TPM=[]
	reps=TPM.columns.get_level_values(2).unique()
	nreps=len(reps)

	print >> sys.stderr, '\n[estimate_dispersion] averaging samples:\n   ',
	for n,c in enumerate(cols):

		print >> sys.stderr, c, 
		log10_std=np.log10(TPM[c].std(axis=1,level=0))
		# perform lowess regression on log CV vs log mean
		log10_means=np.log10(TPM[c].mean(axis=1,level=0))
		log10_CV=log10_std-log10_means
		ok=(np.isfinite(log10_means) & np.isfinite(log10_CV)).all(axis=1)
		log10_mean_range=np.abs(log10_means[ok].max().max()-log10_means[ok].min().min())
		lowess=statsmodels.nonparametric.smoothers_lowess.lowess(log10_CV[ok].values.flatten(),\
																 log10_means[ok].values.flatten(),frac=0.2,it=1,delta=.01*log10_mean_range).T
		# this interpolates log10 CV from log10 mean
		interp=scipy.interpolate.interp1d(lowess[0],lowess[1],bounds_error=False)
		# get interpolated std dev
		log10_std_smooth=interp(log10_means)+log10_means
		# estimated std dev is weighted average of real and smoothened
		log10_std_est=(1.-disp_weight/nreps)*log10_std+disp_weight*log10_std_smooth/nreps
		# get regularized CV
		log10_CV_est=log10_std_est-log10_means

		# add estimated std dev for each replicate
		std_TPM.append(pd.concat([10**log10_std_est]*nreps,axis=1,keys=reps).swaplevel(0,1,axis=1).sort_index(axis=1))

		if fig_name is not None:

			ax=fig.add_subplot(1,len(cols),n+1)
			ax.hexbin(log10_means.values.flatten(),log10_CV.values.flatten(),bins='log',lw=0,extent=(-3,5,-4,1),cmap=plt.cm.Greys,vmin=-1,mincnt=1)
			# plot estimated values
			ax.plot(log10_means.values.flatten()[::10],log10_CV_est.values.flatten()[::10],'c.',markersize=2)
			ax.plot(np.arange(-3,5,.1),interp(np.arange(-3,5,.1)),'r-')
			ax.set_xlim([-3,5])
			ax.set_ylim([-4,1])
			ax.set_title('{0}'.format(c))
			ax.set_xlabel('log10 mean')
			ax.set_ylabel('log10 CV')

	std_TPM=pd.concat(std_TPM,axis=1,keys=cols).sort_index(axis=1)

	if fig_name is not None:
		print >> sys.stderr, '[estimate_dispersion] saving figure to {0}'.format(fig_name)
		fig.savefig(fig_name)

	print >> sys.stderr, '\n'

	return std_TPM

def read_featureCounts_output (inf,samples):

	""" gets read counts per kb values (RPK) from featureCount output file inf and adds samples as column names """

	fc_table=pd.read_csv(inf,sep='\t',comment='#',index_col=0,header=0)
	counts=fc_table.ix[:,5:].astype(int)
	length=fc_table.ix[:,4]/1.e3
	#RPK=counts.divide(length,axis=0)
	#RPK.columns=samples
	counts.columns=pd.MultiIndex.from_tuples(samples)
	return counts,length

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
	parser.add_option('-o','--out_prefix',dest='out_prefix',default='RNAkira',help="output prefix [RNAkira]")
	parser.add_option('','--alpha',dest='alpha',help="FDR cutoff [0.05]",default=0.05,type=float)
	parser.add_option('','--maxlevel',dest='maxlevel',help="max level to test [5]",default=5,type=int)
	parser.add_option('','--min_TPM_mature',dest='min_TPM_mature',help="min TPM for mature [1]",default=1,type=float)
	parser.add_option('','--min_TPM_precursor',dest='min_TPM_precursor',help="min TPM for precursor [.1]",default=.1,type=float)
	parser.add_option('','--min_TPM_ribo',dest='min_TPM_ribo',help="min TPM for ribo [1]",default=1,type=float)
	parser.add_option('','--disp_weight',dest='disp_weight',help="weighting parameter for dispersion estimation (should be smaller than number of replicates) [1.8]",default=1.8,type=float)
	parser.add_option('','--statsmodel',dest='statsmodel',help="statistical model to use (gaussian or neg binom)",default='nbinom')
	parser.add_option('','--no_plots',dest='no_plots',help="don't create plots for 4sU bias correction and normalization",action='store_true')

	# ignore warning about division by zero or over-/underflows
	np.seterr(divide='ignore',over='ignore',under='ignore')

	options,args=parser.parse_args()

	time_points=options.time_points.split(',')

	if len(set(Counter(time_points).values())) > 1:
		raise Exception("unequal number of replicates at timepoints; can't deal with that")

	nreps=Counter(time_points)[time_points[0]]

	samples=[]
	for t in time_points:
		samples.append((t,'Rep'+str(sum(x[0]==t for x in samples)+1)))
	nsamples=len(samples)

	if options.input_TPM is not None:
		
		print >> sys.stderr, '\n[main] reading TPM values from '+options.input_TPM
		print >> sys.stderr, '       ignoring options -eEfFruU'
		TPM=pd.read_csv(options.input_TPM,index_col=0,header=range(3))

	else:

		print >> sys.stderr, "\n[main] reading count data"

		print >> sys.stderr, '   elu-introns:\t\t'+options.elu_introns
		elu_introns,elu_intron_length=read_featureCounts_output(options.elu_introns,samples)

		print >> sys.stderr, '   elu-exons:\t\t'+options.elu_exons
		elu_exons,elu_exon_length=read_featureCounts_output(options.elu_exons,samples)

		print >> sys.stderr, '   flowthrough-introns:\t'+options.flowthrough_introns
		flowthrough_introns,flowthrough_intron_length=read_featureCounts_output(options.flowthrough_introns,samples)

		print >> sys.stderr, '   flowthrough-exons:\t'+options.flowthrough_exons
		flowthrough_exons,flowthrough_exon_length=read_featureCounts_output(options.flowthrough_exons,samples)

		print >> sys.stderr, '   ribo:\t\t'+options.ribo
		ribo,ribo_length=read_featureCounts_output(options.ribo,samples)

		print >> sys.stderr, '   unlabeled-introns:\t'+options.unlabeled_introns
		unlabeled_introns,unlabeled_intron_length=read_featureCounts_output(options.unlabeled_introns,samples)

		print >> sys.stderr, '   unlabeled-exons:\t'+options.unlabeled_exons
		unlabeled_exons,unlabeled_exon_length=read_featureCounts_output(options.unlabeled_exons,samples)

		print >> sys.stderr, "\n[main] merging count values and computing TPM"

		cols=['elu-precursor','elu-mature','flowthrough-precursor','flowthrough-mature','ribo','unlabeled-precursor','unlabeled-mature']

		counts=pd.concat([elu_introns,elu_exons,\
						  flowthrough_introns,flowthrough_exons,\
						  ribo,\
						  unlabeled_introns,unlabeled_exons],axis=1,keys=cols)

		# combine length factors
		LF=pd.concat([elu_intron_length,elu_exon_length,\
					  flowthrough_intron_length,flowthrough_exon_length,\
					  ribo_length,\
					  unlabeled_intron_length,unlabeled_exon_length],axis=1,keys=cols)

		# for size factors, add up RPK values for different fractions (introns and exons), count missing entries as zero
		RPK=counts.divide(LF,axis=0,level=0)
		elu_factor=RPK['elu-mature'].add(RPK['elu-precursor'],fill_value=0).sum(axis=0)/1.e6
		flowthrough_factor=RPK['flowthrough-mature'].add(RPK['flowthrough-precursor'],fill_value=0).sum(axis=0)/1.e6
		unlabeled_factor=RPK['unlabeled-mature'].add(RPK['unlabeled-precursor'],fill_value=0).sum(axis=0)/1.e6
		# for ribo, do as usual
		ribo_factor=RPK['ribo'].sum(axis=0)/1.e6


		# size factors
		SF=pd.concat([elu_factor,elu_factor,\
					  flowthrough_factor,flowthrough_factor,\
					  ribo_factor,\
					  unlabeled_factor,unlabeled_factor],axis=0,keys=cols)
		
		# compute TPM
		TPM=RPK.divide(SF,axis=1).sort_index(axis=1)

		print >> sys.stderr, '[main] saving TPM values to '+options.out_prefix+'_TPM.csv'
		TPM.to_csv(options.out_prefix+'_TPM.csv')

	gene_stats=pd.read_csv(options.gene_stats,index_col=0,header=0).loc[TPM.index]

	print >> sys.stderr, '\n[main] correcting TPM values using gene stats from '+options.gene_stats

	# correction factors
	CF=normalize_elu_flowthrough (TPM, samples, gene_stats, fig_name=(None if options.no_plots else options.out_prefix+'_TPM_correction.pdf'))

	print >> sys.stderr, '[main] saving corrected TPM values to '+options.out_prefix+'_corrected_TPM.csv'
	TPM=TPM.multiply(CF)
	TPM.to_csv(options.out_prefix+'_corrected_TPM.csv')

	print >> sys.stderr, '\n[main] estimating dispersion'

	cols=['elu-precursor','elu-mature','flowthrough-precursor','flowthrough-mature','ribo','unlabeled-precursor','unlabeled-mature']
	std_TPM=estimate_dispersion (TPM, cols, fig_name=(None if options.no_plots else options.out_prefix+'_dispersion.pdf'), disp_weight=options.disp_weight)

	# select genes based on TPM cutoffs for mature, precursor in any of the time points
	take=(TPM['unlabeled-mature'] > options.min_TPM_mature).any(axis=1) &\
		(TPM['unlabeled-precursor'] > options.min_TPM_precursor).any(axis=1)

	if options.input_TPM is not None:
		statsmodel='gaussian'
		print >> sys.stderr, '[main] running RNAkira with TPM input (model: {0})'.format(statsmodel)
		NF=pd.DataFrame(1,index=TPM.index,columns=TPM.columns)
		results=RNAkira(TPM[take].fillna(0), std_TPM[take].fillna(0), NF[take], options.T, \
						sig_level=options.alpha, min_TPM_ribo=options.min_TPM_ribo,maxlevel=options.maxlevel, statsmodel=statsmodel)
	else:
		statsmodel=options.statsmodel
		print >> sys.stderr, '[main] running RNAkira with counts input (model: {0})'.format(statsmodel)
		# normalization factor for counts combines length and size factors with TPM correction (such that TPM = counts.multiply(NF) )
		NF=CF.divide(LF,axis=0,level=0).divide(SF,axis=1).fillna(1)
		results=RNAkira(counts[take].fillna(0), std_TPM[take].fillna(0), NF[take], options.T, \
						sig_level=options.alpha, min_TPM_ribo=options.min_TPM_ribo,maxlevel=options.maxlevel, statsmodel=statsmodel)

	print >> sys.stderr, '\n[main] collecting output'
	output=collect_results(results, time_points, sig_level=options.alpha)

	print >> sys.stderr, '\n   writing results to {0}'.format(options.out_prefix+'_results.csv')

	output.to_csv(options.out_prefix+'_results.csv')

