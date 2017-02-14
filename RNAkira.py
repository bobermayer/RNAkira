import os
import sys
import numpy as np
import pandas as pd
import itertools
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

def get_steady_state_values (x,T,use_precursor,use_ribo,use_deriv=False):

	""" given synthesis, degradation, processing rates and translation efficiencies x=(a,b,c,d) and labeling time T
	returns instantaneous steady-state values for elu, flowthrough, unlabeled mature
	including precursors if use_precursor is set
	ribo if use_ribo is set
	includes derivatives w.r.t. log(a),log(b),log(c) and log(d) if use_deriv is set """

	if use_precursor:

		if use_ribo:
			a,b,c,d=np.exp(x)
		else:
			a,b,c=np.exp(x)

		# for elu-mature, flowthrough-mature, unlabeled-mature
		exp_mean_mature=[a*(b*(1.-np.exp(-c*T))-c*(1.-np.exp(-b*T)))/(b*(b-c)),\
						 a*(b*np.exp(-c*T)-c*np.exp(-b*T))/(b*(b-c)),\
						 a/b]

		# add elu-precursor, flowthrough-precursor, unlabeled-precursor
		exp_mean_precursor=[a*(1.-np.exp(-c*T))/c,\
							a*np.exp(-c*T)/c,\
							a/c]

		exp_mean=exp_mean_mature+exp_mean_precursor

	else:

		if use_ribo:
			a,b,d=np.exp(x)
		else:
			a,b=np.exp(x)

		# for elu-mature, flowthrough-mature, unlabeled-mature
		exp_mean_mature=[a*(1.-np.exp(-b*T))/b,\
						 a*np.exp(-b*T)/b,\
						 a/b]

		exp_mean=exp_mean_mature[:]

	# add ribo value
	if use_ribo:
		exp_mean_ribo=[d*a/b]
		exp_mean+=exp_mean_ribo

	if use_deriv:

		# get the derivatives of mature vals w.r.t. log(a)
		exp_mean_mature_deriv_log_a=exp_mean_mature[:]
		
		if use_precursor:

			# get the derivatives of mature vals w.r.t. log(b)
			exp_mean_mature_deriv_log_b=[a*(b*(b-c)*(1.-np.exp(-c*T)-c*T*np.exp(-b*T))-\
											(2*b-c)*(b*(1.-np.exp(-c*T))-c*(1.-np.exp(-b*T))))/(b*(b-c)**2),\
										 a*(b*(b-c)*(T*c*np.exp(-b*T)+np.exp(-c*T))-(2*b-c)*(b*np.exp(-c*T)-c*np.exp(-b*T)))/(b*(b-c)**2),\
										 -a/b]

			# get deriv of mature w.r.t. log(c)
			exp_mean_mature_deriv_log_c=[a*c*((b-c)*(b*T*np.exp(-c*T)-1.+np.exp(-b*T))+\
											  (b*(1.-np.exp(-c*T))-c*(1.-np.exp(-b*T))))/(b*(b-c)**2),\
										 a*c*((b-c)*(-np.exp(-b*T)-T*b*np.exp(-c*T))+(b*np.exp(-c*T)-c*np.exp(-b*T)))/(b*(b-c)**2),\
										 0]

			exp_mean_mature_deriv=[exp_mean_mature_deriv_log_a,\
								   exp_mean_mature_deriv_log_b,\
								   exp_mean_mature_deriv_log_c]

			# then get derivatives of precursor w.r.t. log(a),log(b),log(c)
			exp_mean_precursor_deriv=[exp_mean_precursor[:],\
									  [0,0,0],\
									  [a*(np.exp(-c*T)*(1.+c*T)-1)/c,\
									   -a*np.exp(-c*T)*(1.+c*T)/c,\
									   -a/c]]

			exp_mean_deriv=[exp_mean_mature_deriv[i]+exp_mean_precursor_deriv[i] for i in range(3)]

			if use_ribo:
				# add derivatives w.r.t. log(d)
				exp_mean_deriv_log_d=[[0,0,0,0,0,0]]
				exp_mean_deriv+=exp_mean_deriv_log_d
				# add derivatives of ribo values w.r.t. log(a),log(b),log(c),log(d)
				exp_mean_ribo_deriv=[[d*a/b],[-d*a/b],[0],[d*a/b]]
				exp_mean_deriv=[exp_mean_deriv[i]+exp_mean_ribo_deriv[i] for i in range(4)]

		else:

			# get the derivatives of mature vals w.r.t. log(b)
			exp_mean_mature_deriv_log_b=[a*(np.exp(-b*T)*(1.+b*T)-1)/b,\
										 -a*np.exp(-b*T)*(1.+b*T)/b,\
										 -a/b]
		
			exp_mean_deriv=[exp_mean_mature_deriv_log_a,\
							exp_mean_mature_deriv_log_b]

			if use_ribo:
				# add derivatives w.r.t. log(d)
				exp_mean_mature_deriv_log_d=[0,0,0]
				exp_mean_deriv=[exp_mean_mature_deriv_log_a,\
								exp_mean_mature_deriv_log_b,
								exp_mean_mature_deriv_log_d]
				# add derivatives of ribo values w.r.t. log(a),log(b),log(d)
				exp_mean_ribo_deriv=[[d*a/b],[-d*a/b],[d*a/b]]
				exp_mean_deriv=[exp_mean_deriv[i]+exp_mean_ribo_deriv[i] for i in range(3)]

	if use_deriv:
		return (np.array(exp_mean),map(np.array,exp_mean_deriv))
	else:
		return np.array(exp_mean)

def get_rates(time_points, pars):

	""" expands log rates over time points given rate change parameters """

	times=np.array(map(float,time_points))
	ntimes=len(times)
	
	# exponential trend in rates modeled by slopes alpha, beta, gamma and maybe delta (log rates have linear trend)

	log_a=pars['log_a0']*np.ones(ntimes)
	if 'alpha' in pars:
		log_a+=pars['alpha']*times
	log_b=pars['log_b0']*np.ones(ntimes)
	if 'beta' in pars:
		log_b+=pars['beta']*times

	rates=[log_a,log_b]

	if 'log_c0' in pars:
		log_c=pars['log_c0']*np.ones(ntimes)
		if 'gamma' in pars:
			log_c+=pars['gamma']*times
		rates.append(log_c)

	if 'log_d0' in pars:
		log_d=pars['log_d0']*np.ones(ntimes)
		if 'delta' in pars:
			log_d+=pars['delta']*times
		rates.append(log_d)

	return rates

		
def steady_state_log_likelihood (x, vals, std, T, time_points, prior_mu, prior_std, model_pars, use_precursor, use_ribo, use_deriv):

	""" log-likelihood function for difference between expected and observed values, including all priors """

	nrates=2
	if use_precursor:
		nrates+=1
	if use_ribo:
		nrates+=1

	times=np.array(map(float,time_points))
	ntimes=len(times)

	# get instantaneous rates a,b,c,d at each timepoint
	log_rates=[x[i]*np.ones(ntimes) for i in range(nrates)]
	k=0
	if 'alpha' in model_pars:
		log_rates[0]+=x[nrates+k]*times
		k+=1
	if 'beta' in model_pars:
		log_rates[1]+=x[nrates+k]*times
		k+=1

	if use_precursor and 'gamma' in model_pars:
		log_rates[2]+=x[nrates+k]*times
		k+=1

	if use_ribo and 'delta' in model_pars:
		log_rates[nrates-1]+=x[nrates+k]*times

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
				g.append(g[nrates-1]*t)
			grad+=np.array(g)

		if use_deriv:
			exp_mean,exp_mean_deriv=get_steady_state_values(log_rates_here,T,use_precursor,use_ribo,use_deriv)

			# add derivatives w.r.t. alpha etc. if necessary
			if 'alpha' in model_pars:
				exp_mean_deriv.append(exp_mean_deriv[0]*t)
			if 'beta' in model_pars:
				exp_mean_deriv.append(exp_mean_deriv[1]*t)
			if 'gamma' in model_pars:
				exp_mean_deriv.append(exp_mean_deriv[2]*t)
			if 'delta' in model_pars:
				exp_mean_deriv.append(exp_mean_deriv[nrates-1]*t)

		else:

			exp_mean=get_steady_state_values(log_rates_here,T,use_precursor,use_ribo)

		diff=vals[i]-exp_mean
		fun+=np.sum(-.5*(diff/std[i])**2-.5*np.log(2.*np.pi)-np.log(std[i]))
		if use_deriv:
			grad+=np.dot(exp_mean_deriv,np.sum(diff/std[i]**2,axis=0))

	if fun is np.nan:
		raise Exception("invalid value in steady_state_log_likelihood!")

	if use_deriv:
		# return negative log likelihood and gradient
		return (-fun,-grad)
	else:
		# return negative log likelihood
		return -fun

def fit_model (obs_mean, obs_std, T, time_points, model_priors, parent, model, model_pars, use_precursor, use_ribo, min_args):

	""" fits a specific model to data """

	test_gradient=True

	nrates=2+use_precursor+use_ribo

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
			  model_pars, use_precursor, use_ribo, min_args['jac'])

		# test gradient against numerical difference if necessary
		if test_gradient:

			pp=initial_estimate
			eps=1.e-6*np.abs(initial_estimate)+1.e-8
			fun,grad=steady_state_log_likelihood(pp,*args)
			my_grad=np.array([(steady_state_log_likelihood(np.array(list(pp[:i])+[pp[i]+eps[i]]+list(pp[i+1:])),*args)[0]\
							   -steady_state_log_likelihood(np.array(list(pp[:i])+[pp[i]-eps[i]]+list(pp[i+1:])),*args)[0])/(2*eps[i]) for i in range(len(pp))])
			diff=np.sum((grad-my_grad)**2)/np.sum(grad**2)
			if diff > 1.e-8:
				raise Warning('\n!!! gradient diff: {0:.3e}'.format(diff))

		# perform minimization
		res=scipy.optimize.minimize(steady_state_log_likelihood, \
									initial_estimate,\
									args=args, \
									**min_args)

		# collect results
		result=dict(est_pars=pd.Series(res.x,index=model_pars),\
					L=-res.fun,\
					success=res.success,\
					npars=len(model_pars),
					model=model)

	else:

		# in the 'all' models, treat every time point independently
		x=[]
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
				  model_pars, use_precursor, use_ribo, min_args['jac'])

			# test gradient against numerical difference if necessary
			if test_gradient:

				pp=initial_estimate
				eps=1.e-6*np.abs(initial_estimate)+1.e-8
				fun,grad=steady_state_log_likelihood(pp,*args)
				my_grad=np.array([(steady_state_log_likelihood(np.array(list(pp[:i])+[pp[i]+eps[i]]+list(pp[i+1:])),*args)[0]\
								   -steady_state_log_likelihood(np.array(list(pp[:i])+[pp[i]-eps[i]]+list(pp[i+1:])),*args)[0])/(2*eps[i]) for i in range(len(pp))])
				diff=np.sum((grad-my_grad)**2)/np.sum(grad**2)
				if diff > 1.e-8:
					raise Warning('\n!!! gradient diff: {0:.3e}'.format(diff))

			res=scipy.optimize.minimize(steady_state_log_likelihood, \
										initial_estimate,\
										args=args, \
										**min_args)

			x.append(res.x)

			L+=-res.fun
			success=success & res.success
			npars+=len(model_pars)

		result=dict(est_pars=pd.DataFrame(x,columns=model_pars,index=time_points),\
					L=L,\
					success=success,\
					npars=npars,\
					model=model)

	if parent is not None:
		# calculate p-value from LRT test using chi2 distribution
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

def RNAkira (vals, stddev, T, sig_level=0.01, min_precursor=1, min_ribo=1, maxlevel=None, priors=None):

	""" main routine in this package: given dataframe of TPM values, stddev, labeling time, estimates empirical priors and fits models of increasing complexity """

	genes=vals.index
	time_points=np.unique(vals.columns.get_level_values(1))
	nreps=len(np.unique(vals.columns.get_level_values(2)))

	ndim=7

	use_precursor=(vals['unlabeled-precursor'] > min_precursor).any(axis=1)
	use_ribo=(vals['ribo'] > min_ribo).any(axis=1)

	# these are the features to use depending on cutoffs
	mature_cols=['elu-mature','flowthrough-mature','unlabeled-mature']
	precursor_cols=['elu-precursor','flowthrough-precursor','unlabeled-precursor']
	ribo_cols=['ribo']

	print >> sys.stderr, '[RNAkira] analyzing {0} MPR genes, {1} MR genes, {2} MP genes, {3} M genes'.format((use_precursor & use_ribo).sum(),(~use_precursor & use_ribo).sum(),(use_precursor & ~use_ribo).sum(),(~use_precursor & ~use_ribo).sum())
	print >> sys.stderr, '[RNAkira] using {0} time points, {1} replicates'.format(len(time_points),nreps)

	#min_args=dict(method='L-BFGS-B',jac=False,options={'disp':False, 'ftol': 1.e-15, 'gtol': 1.e-10})
	min_args=dict(method='L-BFGS-B',jac=True,options={'disp':False, 'ftol': 1.e-15, 'gtol': 1.e-10})

	if maxlevel is not None:
		nlevels=maxlevel+1
	else:
		nlevels=6

	mature_pars=['log_a0','log_b0']
	precursor_pars=['log_c0']
	ribo_pars=['log_d0']

	model_types=['MPR','MR','MP','M']

	fixed_pars=dict(MPR=mature_pars+precursor_pars+ribo_pars,\
					MR=mature_pars+ribo_pars,\
					MP=mature_pars+precursor_pars,\
					M=mature_pars)

	variable_pars=dict(MPR=['alpha','beta','gamma','delta'],\
					   MR=['alpha','beta','delta'],\
					   MP=['alpha','beta','gamma'],\
					   M=['alpha','beta'])

	par_code=dict(alpha='a',beta='b',gamma='c',delta='d')

	# define nested hierarchy of models (model name, model parameters, parent models)
	models=OrderedDict()
	for level in range(nlevels):
		level_models=OrderedDict()
		for mt in model_types:
			if level==0:
				level_models.update({mt+'_all': (fixed_pars[mt],[])})
			elif level==1:
				level_models.update({mt+'_0': (fixed_pars[mt],[mt+'_all'])})
			else:
				fp=fixed_pars[mt]
				vp=variable_pars[mt]
				# get all combination of the variable pars
				for par_comb in itertools.combinations(vp,level-1):
					# find parent model that use a subset of these parameters
					if level==2:
						parent_models=[mt+'_0']
					else:
						parent_models=[mt+'_'+''.join(par_code[p] for p in x) for x in itertools.combinations(vp,level-2) if set(x) < set(par_comb)]
					level_models.update({mt+'_'+''.join(par_code[p] for p in par_comb): (fixed_pars[mt]+list(par_comb),parent_models)})
		models.update({level: level_models})
		
	model_pars=dict((m,v[0]) for l in range(nlevels) for m,v in models[l].iteritems())

	results=defaultdict(dict)
	all_results=defaultdict(dict)
	print >> sys.stderr, '\n[RNAkira] running initial fits,',

	if priors is None:

		# initialize priors with some empirical data (use 45% reasonably expressed genes)

		print >> sys.stderr, 'estimating empirical priors'

		take_mature=((vals['unlabeled-mature'] > vals['unlabeled-mature'].quantile(.5)) & \
					 (vals['unlabeled-mature'] < vals['unlabeled-mature'].quantile(.95))).any(axis=1)
		take_precursor=take_mature & use_precursor & ((vals['unlabeled-precursor'] > vals['unlabeled-precursor'].quantile(.5)) & \
													  (vals['unlabeled-precursor'] < vals['unlabeled-precursor'].quantile(.95))).any(axis=1)
		take_ribo=take_mature & use_ribo & ((vals['ribo'] > vals['ribo'].quantile(.5)) & \
											(vals['ribo'] < vals['ribo'].quantile(.95))).any(axis=1)

		log_a_est=np.log((vals['elu-mature']+vals['elu-precursor'])/T)[take_precursor].values.flatten()
		log_b_est=np.log(np.log(1+vals['elu-mature']/vals['flowthrough-mature'])/T)[take_mature].values.flatten()
		log_c_est=np.log(np.log(1+vals['elu-precursor']/vals['flowthrough-precursor'])/T)[take_precursor].values.flatten()
		log_d_est=np.log(vals['ribo']/vals['unlabeled-mature'])[take_ribo].values.flatten()

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

			model='M'
			cols=mature_cols[:]
			if use_precursor[gene]:
				cols+=precursor_cols[:]
				model+='P'
			if use_ribo[gene]:
				cols+=ribo_cols[:]
				model+='R'
			model+='_all'

			model_pars,parent_models=models[level][model]
			
			# make numpy arrays out of vals, disp and NF for computational efficiency (therefore we need equal number of replicates for each time point!)
			vals_here=vals.loc[gene].unstack(level=0)[cols].stack().values.reshape((len(time_points),nreps,len(cols)))
			stddev_here=stddev.loc[gene].unstack(level=0)[cols].stack().values.reshape((len(time_points),nreps,len(cols)))

			result=fit_model (vals_here, stddev_here, T, time_points, model_priors, None, model, model_pars, use_precursor[gene], use_ribo[gene], min_args)

			results[gene][level]=result
			nfits+=1

		if priors is not None:
			break

		model_pars,_=models[0]['MPR_all']
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
			for gene in genes[(use_ribo if 'R' in model else ~use_ribo) & (use_precursor if 'P' in model else ~use_precursor)]:

				cols=mature_cols[:]
				if use_precursor[gene]:
					cols+=precursor_cols[:]
				if use_ribo[gene]:
					cols+=ribo_cols[:]

				print >> sys.stderr, '   model: {0}, {1} fits\r'.format(model,nfits),

				# make numpy arrays out of vals, stddev for computational efficiency (therefore we need equal number of replicates for each time point!)
				vals_here=vals.loc[gene].unstack(level=0)[cols].stack().values.reshape((len(time_points),nreps,len(cols)))
				stddev_here=stddev.loc[gene].unstack(level=0)[cols].stack().values.reshape((len(time_points),nreps,len(cols)))

				# use initial estimate from previous level
				if level-1 in results[gene] and results[gene][level-1]['model'] in parent_models:
					parent=results[gene][level-1]
				else:
					continue

				# only compute this model if the parent model was significant
				if level > 1 and not parent['LRT-q'] < sig_level:
					continue

				result=fit_model (vals_here, stddev_here, T, time_points, model_priors, parent, model, model_pars, use_precursor[gene], use_ribo[gene], min_args)

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
			[('initial_processing_t{0}'.format(t),np.exp(pars.ix[t,'log_c0']) if 'log_c0' in pars else np.nan) for t in time_points]+\
			[('initial_translation_t{0}'.format(t),np.exp(pars.ix[t,'log_d0']) if 'log_d0' in pars else np.nan) for t in time_points]
		tmp+=[('initial_logL',initial_fit['L']),\
			  ('initial_fit_success',initial_fit['success']),\
			  ('initial_pval',initial_fit['LRT-p'] if 'LRT-p' in initial_fit else np.nan),\
			  ('initial_qval',initial_fit['LRT-q'] if 'LRT-q' in initial_fit else np.nan)]

		# take best significant model or constant otherwise
		best_fit=filter(lambda x: (x['model'].endswith('0') or x['LRT-q'] < sig_level),res)[-1]
		pars=best_fit['est_pars']
		rates=get_rates(time_points,pars)
		log_a,log_b=rates[:2]
		log_c=rates[2] if 'log_c0' in pars else np.nan*np.ones(len(time_points))
		log_d=rates[-1] if 'log_d0' in pars else np.nan*np.ones(len(time_points))
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

	# collect corrected values
	corrected_TPM=pd.DataFrame(index=TPM.index,columns=TPM.columns)

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

		# corrected values for precursors
		corrected_TPM['unlabeled-precursor',t,r]=TPM['unlabeled-precursor',t,r]
		corrected_TPM['elu-precursor',t,r]=TPM['elu-precursor',t,r]*2**(-np.log2(-intercept/slope))
		corrected_TPM['flowthrough-precursor',t,r]=TPM['flowthrough-precursor',t,r]*2**(-np.log2(intercept))

		# corrected values for mature after correcting for 4sU incorporation bias in mature
		corrected_TPM['unlabeled-mature',t,r]=TPM['unlabeled-mature',t,r]
		corrected_TPM['elu-mature',t,r]=TPM['unlabeled-mature',t,r]*2**(log2_elu_ratio_no_bias.fillna(0)-np.log2(-intercept/slope))
		corrected_TPM['flowthrough-mature',t,r]=TPM['unlabeled-mature',t,r]*2**(log2_FT_ratio.fillna(0)-np.log2(intercept))

		# uncorrected values for ribo
		corrected_TPM['ribo',t,r]=TPM['ribo',t,r]

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

	corrected_TPM.columns=pd.MultiIndex.from_tuples(corrected_TPM.columns)

	if fig_name is not None:
		print >> sys.stderr, '[normalize_elu_flowthrough] saving figure to {0}'.format(fig_name)
		fig.savefig(fig_name)

	print >> sys.stderr, '\n'

	return corrected_TPM

def estimate_stddev (TPM, fig_name=None, weight=1.8):

	""" estimates stddev by a weighted average of the calculated std dev and a smoothened std dev from the mean-variance plot """

	cols=TPM.columns.get_level_values(0).unique()
	time_points=TPM.columns.get_level_values(1).unique()
	stddev=pd.DataFrame(1,index=TPM.index,columns=TPM.columns)
	reps=TPM.columns.get_level_values(2).unique()
	nreps=len(reps)

	if fig_name is not None:
		import matplotlib
		matplotlib.use('Agg')
		from matplotlib import pyplot as plt
		fig=plt.figure(figsize=(3*len(cols),3))
		fig.subplots_adjust(left=.05,right=.98,hspace=.4,wspace=.3,bottom=.15)

	print >> sys.stderr, '\n[estimate_stddev] averaging samples:\n   ',
	for n,c in enumerate(cols):

		print >> sys.stderr, c, 

		# first do within-group stddev
		log10_std_within=np.log10(TPM[c].std(axis=1,level=0))
		# perform lowess regression on log CV vs log mean
		log10_means_within=np.log10(TPM[c].mean(axis=1,level=0))
		log10_CV_within=log10_std_within-log10_means_within
		ok=np.isfinite(log10_means_within) & np.isfinite(log10_CV_within)
		log10_mean_range=np.abs(log10_means_within[ok].max().max()-log10_means_within[ok].min().min())
		lowess_within=statsmodels.nonparametric.smoothers_lowess.lowess(log10_CV_within[ok].values.flatten(),\
																		log10_means_within[ok].values.flatten(),\
																		frac=0.2,it=1,delta=.01*log10_mean_range).T
		# this interpolates log10 CV from log10 mean
		interp_within=scipy.interpolate.interp1d(lowess_within[0],lowess_within[1],bounds_error=False)

		# then do stddev over all samples
		log10_std_all=np.log10(TPM[c].std(axis=1))
		# perform lowess regression on log CV vs log mean
		log10_means_all=np.log10(TPM[c].mean(axis=1))
		log10_CV_all=log10_std_all-log10_means_all
		ok=np.isfinite(log10_means_all) & np.isfinite(log10_CV_all)
		log10_mean_range=np.abs(log10_means_all[ok].max()-log10_means_all[ok].min())
		lowess_all=statsmodels.nonparametric.smoothers_lowess.lowess(log10_CV_all[ok].values,log10_means_all[ok].values,\
																	 frac=0.2,it=1,delta=.01*log10_mean_range).T
		interp_all=scipy.interpolate.interp1d(lowess_all[0],lowess_all[1],bounds_error=False)

		# get interpolated std dev within groups
		log10_std_smooth=interp_within(log10_means_within)+log10_means_within
		# estimated std dev is weighted average of real and smoothened
		log10_std_est=(1.-weight/nreps)*log10_std_within+weight*log10_std_smooth/nreps
		replace=log10_std_est.isnull().any(axis=1)
		if replace.sum() > 0:
			repl=((1.-weight/nreps)*log10_std_all+weight*(interp_all(log10_means_all)+log10_means_all)/nreps)[replace]
			log10_std_est[replace]=pd.concat([repl]*len(time_points),axis=1,keys=time_points)

		# add estimated std dev for each replicate (NaN values only where mean=0)
		stddev[c]=pd.concat([10**log10_std_est]*nreps,axis=1,keys=reps).reorder_levels([1,0],axis=1).fillna(1)

		if fig_name is not None:

			# get regularized CV
			log10_CV_est=log10_std_est-log10_means_within

			ax=fig.add_subplot(1,len(cols),n+1)
			ax.hexbin(log10_means_within.values.flatten(),log10_CV_within.values.flatten(),\
					  bins='log',lw=0,extent=(-2,5,-2,1),cmap=plt.cm.Greys,vmin=-1,mincnt=1)
			# plot estimated values
			ax.plot(log10_means_within.values.flatten()[::10],log10_CV_est.values.flatten()[::10],'c.',markersize=2)
			ax.plot(np.arange(-2,5,.1),interp_within(np.arange(-2,5,.1)),'r-')
			ax.set_xlim([-2,5])
			ax.set_ylim([-2,1])
			ax.set_title('{0}'.format(c))
			ax.set_xlabel('log10 mean')
			ax.set_ylabel('log10 CV')

	if fig_name is not None:
		print >> sys.stderr, '\n[estimate_stddev] saving figure to {0}'.format(fig_name),
		fig.savefig(fig_name)

	print >> sys.stderr, '\n'

	return stddev

def get_RPK_from_featureCount (inf,samples):

	""" gets read counts per kb values (RPK) from featureCount output file inf and adds samples as column names """

	fc_table=pd.read_csv(inf,sep='\t',comment='#',index_col=0,header=0)
	RPK=fc_table.ix[:,5:].divide(fc_table.ix[:,4]/1.e3,axis=0)
	RPK.columns=samples
	return RPK

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
	parser.add_option('','--min_mature',dest='min_mature',help="min TPM for mature [1]",default=1,type=float)
	parser.add_option('','--min_precursor',dest='min_precursor',help="min TPM for precursor [.1]",default=.1,type=float)
	parser.add_option('','--min_ribo',dest='min_ribo',help="min TPM for ribo [1]",default=1,type=float)
	parser.add_option('','--weight',dest='weight',help="weighting parameter for stddev estimation (should be smaller than number of replicates) [1.8]",default=1.8,type=float)
	parser.add_option('','--no_plots',dest='no_plots',help="don't create plots for 4sU bias correction and normalization",action='store_false')

	# ignore warning about division by zero or over-/underflows
	np.seterr(divide='ignore',over='ignore',under='ignore',invalid='ignore')

	options,args=parser.parse_args()

	time_points=options.time_points.split(',')

	if len(set(Counter(time_points).values())) > 1:
		raise Exception("unequal number of replicates at timepoints; can't deal with that")

	nreps=Counter(time_points)[time_points[0]]

	samples=[]
	for t in time_points:
		samples.append((t,'Rep'+str(sum(x.split('-')[0]==t for x in samples)+1)))
	nsamples=len(samples)

	if options.input_TPM is not None:
		
		print >> sys.stderr, '\n[main] reading TPM values from '+options.input_TPM
		print >> sys.stderr, '       ignoring options -eEfFruU'
		TPM=pd.read_csv(options.input_TPM,index_col=0,header=range(3))

	else:

		print >> sys.stderr, "\n[main] reading count data"

		print >> sys.stderr, '   elu-introns:\t\t'+options.elu_introns
		elu_introns=get_RPK_from_featureCount(options.elu_introns,samples)

		print >> sys.stderr, '   elu-exons:\t\t'+options.elu_exons
		elu_exons=get_RPK_from_featureCount(options.elu_exons,samples)

		print >> sys.stderr, '   flowthrough-introns:\t'+options.flowthrough_introns
		flowthrough_introns=get_RPK_from_featureCount(options.flowthrough_introns,samples)

		print >> sys.stderr, '   flowthrough-exons:\t'+options.flowthrough_exons
		flowthrough_exons=get_RPK_from_featureCount(options.flowthrough_exons,samples)

		print >> sys.stderr, '   ribo:\t\t'+options.ribo
		ribo=get_RPK_from_featureCount(options.ribo,samples)

		print >> sys.stderr, '   unlabeled-introns:\t'+options.unlabeled_introns
		unlabeled_introns=get_RPK_from_featureCount(options.unlabeled_introns,samples)

		print >> sys.stderr, '   unlabeled-exons:\t'+options.unlabeled_exons
		unlabeled_exons=get_RPK_from_featureCount(options.unlabeled_exons,samples)

		print >> sys.stderr, "\n[main] merging count values and computing TPM"

		cols=['elu-mature','flowthrough-mature','unlabeled-mature','elu-precursor','flowthrough-precursor','unlabeled-precursor','ribo']

		# add up RPK values for different fractions (introns and exons), count missing entries as zero
		elu_factor=elu_exons.add(elu_introns,fill_value=0).sum(axis=0)/1.e6
		flowthrough_factor=flowthrough_exons.add(flowthrough_introns,fill_value=0).sum(axis=0)/1.e6
		unlabeled_factor=unlabeled_exons.add(unlabeled_introns,fill_value=0).sum(axis=0)/1.e6
		# for ribo, do as usual
		ribo_factor=ribo.sum(axis=0)/1.e6

		# normalize RPK values to TPM and combine in TPM dataframe
		TPM=pd.concat([elu_exons.divide(elu_factor,axis=1),\
					   flowthrough_exons.divide(flowthrough_factor,axis=1),\
					   unlabeled_exons.divide(unlabeled_factor,axis=1),\
					   elu_introns.divide(elu_factor,axis=1),\
					   flowthrough_introns.divide(flowthrough_factor,axis=1),\
					   unlabeled_introns.divide(unlabeled_factor,axis=1),\
					   ribo.divide(ribo_factor,axis=1)],axis=1,keys=cols)

		print >> sys.stderr, '[main] saving TPM values to '+options.out_prefix+'_TPM.csv'
		TPM.to_csv(options.out_prefix+'_TPM.csv')

	gene_stats=pd.read_csv(options.gene_stats,index_col=0,header=0).loc[TPM.index]

	print >> sys.stderr, '\n[main] correcting TPM values using gene stats from '+options.gene_stats

	TPM=normalize_elu_flowthrough (TPM, samples, gene_stats, fig_name=(None if options.no_plots else options.out_prefix+'_TPM_correction.pdf'))

	print >> sys.stderr, '[main] saving corrected TPM values to '+options.out_prefix+'_corrected_TPM.csv'
	TPM.to_csv(options.out_prefix+'_corrected_TPM.csv')

	print >> sys.stderr, '\n[main] estimating dispersion'

	stddev=estimate_stddev (TPM, fig_name=(None if options.no_plots else options.out_prefix+'_dispersion.pdf'), weight=options.weight)

	# select genes based on TPM cutoffs for mature, precursor in any of the time points
	take=(TPM['unlabeled-mature'] > options.min_mature).any(axis=1)

	print >> sys.stderr, '[main] running RNAkira'
	results=RNAkira(TPM[take], stddev[take], options.T, sig_level=options.alpha, \
					min_precursor=options.min_precursor,min_ribo=options.min_ribo,\
					maxlevel=options.maxlevel)

	print >> sys.stderr, '\n[main] collecting output'
	output=collect_results(results, time_points, sig_level=options.alpha)

	print >> sys.stderr, '\n   writing results to {0}'.format(options.out_prefix+'_results.csv')

	output.to_csv(options.out_prefix+'_results.csv')

