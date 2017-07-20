import os
import sys
import numpy as np
import pandas as pd
from string import maketrans
import itertools
import scipy.stats
import scipy.interpolate
import scipy.odr
import scipy.optimize
import statsmodels.nonparametric.smoothers_lowess
import statsmodels.regression.linear_model
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

def odr_regression(xvals,yvals,beta0=[1,0],we=None,wd=None):

    """ orthogonal distance regression """

    def linear_function (B,x):
        return B[0]*x + B[1]
    linear=scipy.odr.Model(linear_function)
    data=scipy.odr.Data(xvals,yvals,we=we,wd=wd)
    myodr=scipy.odr.ODR(data,linear,beta0=beta0)
    return myodr.run().beta

def get_model_definitions(nlevels):

    """ returns a nested hierarchy of models (uppercase denotes variable, lowercase constant rates) """

    model_types=['abcd','abd','abc','ab']
    # define nested hierarchy of models (model name & parent models)
    models=OrderedDict()
    for level in range(nlevels):
        level_models=OrderedDict()
        for pars in model_types:
            if level==0:
                level_models.update({pars.upper(): []})
            elif level==1:
                level_models.update({pars: ([pars.upper()])})
            elif level <= len(pars):
                # get all combination of the variable pars
                for pc in itertools.combinations(pars,level-1):
                    # find parent model that use a subset of these parameters
                    if level==2:
                        parent_models=[pars]
                    else:
                        parent_models=[pars.translate(maketrans(''.join(x),''.join(x).upper())) for x in itertools.combinations(pars,level-2) if set(x) < set(pc)]
                    level_models.update({pars.translate(maketrans(''.join(pc),''.join(pc).upper())): parent_models})
        models.update({level: level_models})

    return models

def get_steady_state_values (x, T, model, use_deriv=False):

    """ given synthesis, degradation, processing rates and translation efficiencies x=(a,b,c,d) and labeling time T
    returns instantaneous steady-state values for elu, flowthrough, unlabeled mature including precursors and ribo depending on model
    includes derivatives w.r.t. log(a),log(b),log(c) and log(d) if use_deriv is set """

    use_precursor='c' in model.lower()
    use_ribo='d' in model.lower()

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
        return (np.array(exp_mean),np.array(exp_mean_deriv))
    else:
        return np.array(exp_mean)

def get_rates (x, ntimes, model):

    """ expands log rates over time points given rate parameters x, number of time points, and model """

    nrates=len(model)

    # get instantaneous rates a,b,c,d at each timepoint by expanding the ones that don't vary over time
    log_rates=np.zeros((nrates,ntimes))
    k=0
    for n,mp in enumerate(model):
        # uppercase letters in the model definition mean this parameter varies over time
        if mp.isupper():
            log_rates[n]=x[k:k+ntimes]
            k+=ntimes
        else:
            log_rates[n]=x[k]
            k+=1
    
    return log_rates.T
        
def steady_state_log_likelihood (x, vals, var, nf, T, time_points, prior_mu, prior_std, model, statsmodel, use_deriv):

    """ log-likelihood function for difference between expected and observed values, including all priors """

    ntimes=len(time_points)

    fun=0
    log_rates=get_rates(x, ntimes, model)
    if use_deriv:
        grad=np.zeros(len(x))
        # index array for gradient vector knows on which parameters vary over time
        gg=defaultdict(list)
        k=0
        for n,mp in enumerate(model):
            if mp.isupper():
                for i,t in enumerate(time_points):
                    gg[t].append(k+i)
                k+=ntimes
            else:
                for t in time_points:
                    gg[t].append(k)
                k+=1
        gg=dict((k,np.array(v)) for k,v in gg.iteritems())

    # add up model log-likelihoods for each time point and each replicate
    for i,t in enumerate(time_points):

        # first add to log-likelihood the priors on rates at each time point (normal distribution with mu and std)
        diff=log_rates[i]-prior_mu
        fun+=np.sum(-.5*(diff/prior_std)**2-.5*np.log(2.*np.pi)-np.log(prior_std))

        if use_deriv:
            # add derivatives of rate parameters
            grad[gg[t]]+=(-diff/prior_std**2)

        # get expected mean values (and their derivatives)
        if use_deriv:
            exp_mean,exp_mean_deriv=get_steady_state_values(log_rates[i],T,model,use_deriv)
        else:
            exp_mean=get_steady_state_values(log_rates[i],T,model)

        if statsmodel=='gaussian':
            # here "var" is stddev
            diff=vals[i]*nf[i]-exp_mean
            #fun+=np.sum(-.5*(diff/var[i])**2-.5*np.log(2.*np.pi)-np.log(var[i]))
            fun+=np.sum(scipy.stats.norm.logpdf(diff,scale=var[i]))
            if use_deriv:
                grad[gg[t]]+=np.dot(exp_mean_deriv,np.sum(diff/var[i]**2,axis=0))

        elif statsmodel=='nbinom':
            # here "var" is dispersion
            mu=exp_mean/nf[i]
            nn=1./var[i]
            pp=1./(1.+var[i]*mu)
            fun+=np.sum(scipy.stats.nbinom.logpmf(vals[i],nn,pp))
            if use_deriv:
                tmp=(vals[i]-nn*var[i]*mu)/(exp_mean*(1.+var[i]*mu))
                grad[gg[t]]+=np.dot(exp_mean_deriv,np.sum(tmp,axis=0))

    if fun is np.nan:
        raise Exception("invalid value in steady_state_log_likelihood!")

    if use_deriv:
        # return negative log likelihood and gradient
        return (-fun,-grad)
    else:
        # return negative log likelihood
        return -fun

def steady_state_residuals (x, vals, nf, T, time_points, model):

    """ residuals between observed and expected values for each fraction separately"""

    ntimes=len(time_points)
    log_rates=get_rates(x, ntimes, model)
    ncols=3+3*('c' in model.lower())+('d' in model.lower())
    
    res=np.zeros(ncols)
    # add up model residuals for each time point and each replicate
    for i in range(ntimes):
        exp_mean=get_steady_state_values(log_rates[i],T,model)
        res+=np.sum((vals[i]*nf[i]-exp_mean)**2,axis=0)

    if res is np.nan:
        raise Exception("invalid value in steady_state_residuals")

    return res

def fit_model (vals, var, nf, T, time_points, priors, parent, model, statsmodel, min_args, test_gradient=False):

    """ fits a specific model to data given variance, normalization factors, labeling time, time points, priors, initial estimate from a parent model etc. """

    ntimes=len(time_points)
    nrates=sum(ntimes if mp.isupper() else 1 for mp in model)
    ncols=3+3*('c' in model.lower())+('d' in model.lower())
    iRNA=2
    iribo=3+3*('c' in model.lower())

    if parent is not None:
        if model.lower()==model:
            initial_estimate=parent['est_pars'].mean(axis=0)
        else:
            initial_estimate=np.concatenate([parent['est_pars'][mp.lower()] if mp.isupper() else [parent['est_pars'][mp.lower()].mean()] for mp in model])
    else:
        # use initial estimate from priors
        initial_estimate=np.repeat(priors['mu'].values,ntimes)

    # arguments to minimization
    args=(vals, var, nf, T, time_points, priors['mu'].values, priors['std'].values, model, statsmodel, min_args['jac'])

    # test gradient against numerical difference if necessary
    if test_gradient:

        pp=initial_estimate
        eps=1.e-6*np.abs(initial_estimate)+1.e-8
        fun,grad=steady_state_log_likelihood(pp,*args)
        my_grad=np.array([(steady_state_log_likelihood(np.array(list(pp[:i])+[pp[i]+eps[i]]+list(pp[i+1:])),*args)[0]\
                           -steady_state_log_likelihood(np.array(list(pp[:i])+[pp[i]-eps[i]]+list(pp[i+1:])),*args)[0])/(2*eps[i]) for i in range(len(pp))])
        diff=np.sum((grad-my_grad)**2)/np.sum(grad**2)
        if diff > 1.e-7:
            raise Warning('\n!!! gradient diff: {0:.3e}'.format(diff))

    # perform minimization
    res=scipy.optimize.minimize(steady_state_log_likelihood, \
                                initial_estimate,\
                                args=args, \
                                **min_args)

    # get sums of squares for coefficient of determination
    SSres=steady_state_residuals(res.x,vals,nf,T,time_points,model)
    mean_vals=np.sum(np.sum(vals*nf,axis=0),axis=0)/np.prod(vals.shape[:2])
    SStot=np.sum(np.sum((vals*nf-mean_vals)**2,axis=0),axis=0)

    # collect results
    rates=get_rates(res.x, ntimes, model)
    result=dict(est_pars=pd.DataFrame(rates,columns=list(model.lower()),index=time_points),\
                logL=-res.fun,\
                R2_tot=1-SSres.sum()/np.sum((vals*nf-np.mean(vals*nf))**2),\
                R2_RNA=(1-SSres/SStot)[iRNA],\
                R2_ribo=(1-SSres/SStot)[iribo] if 'd' in model.lower() else np.nan,\
                AIC=2*(len(res.x)+1+res.fun),\
                success=res.success,\
                message=res.message,\
                npars=len(res.x),
                model=model)

    if parent is not None:
        # calculate p-value from LRT test using chi2 distribution
        pval=scipy.stats.chi2.sf(2*np.abs(result['logL']-parent['logL']),np.abs(result['npars']-parent['npars']))
        result['LRT-p']=(pval if np.isfinite(pval) else np.nan)
        # calculate AIC improvement
        result['dAIC']=np.abs(result['AIC']-parent['AIC'])/np.mean([result['AIC'],parent['AIC']])

    return result

def plot_data_rates_fits (time_points, replicates, TPM, T, parameters, results, use_precursor, use_ribo, title='', priors=None, alpha=0.01):

    """ function to plot summary of results for a specific gene; NEEDS TO BE FIXED!! """

    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    times=np.array(map(float,time_points))
    ntimes=len(times)

    def jitter():
        return np.random.normal(loc=0,scale=.1*np.min(np.diff(times)),size=len(time_points))

    cols=['elu-mature','flowthrough-mature','unlabeled-mature']
    rates=['synthesis','degradation']
    if use_precursor: 
        cols+=['elu-precursor','flowthrough-precursor','unlabeled-precursor']
        rates+=['processing']
    if use_ribo:
        cols+=['ribo']
        rates+=['translation']

    nrates=len(rates)
    ndim=len(cols)

    fig=plt.figure(figsize=(15,6))
    fig.clf()
    fig.subplots_adjust(wspace=.5,hspace=.4,left=.05,right=.98)

    ax=[fig.add_subplot(2,ndim,i+1) for i in range(ndim+nrates)]

    # first plot data and fits to data

    for i in range(ndim):
        for r in replicates:
            ax[i].plot(times,TPM[cols[i]].xs(r,level=1),'ko',label='_nolegend_',mfc='none')

    if parameters is not None:
        exp_vals=get_steady_state_values(get_rates(time_points,parameters),T,use_precursor,use_ribo)
        for i in range(ndim):
            ax[i].plot(times,exp_vals[i],'k-',label='theo. mean')

    for level,vals in enumerate(results):
        pred_vals=get_steady_state_values(get_rates(time_points,vals['est_pars']),T,use_precursor,use_ribo)
        for i in range(ndim):
            if level > 1:
                qval=vals['LRT-q']
                ax[i].plot(times,pred_vals[i],linestyle=('-' if qval <= alpha else '--'),\
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
        for i,p in enumerate(get_rates(time_points,vals['est_pars'])):
            qval=vals['LRT-q']
            if level >= 1:
                ax[ndim+i].plot(times,p,linestyle=('-' if qval <= alpha else '--'),label='{0} (q={1:.2g})'.format(vals['model'],qval))
            else:
                ax[ndim+i].plot(times,p,linestyle=':',label='initial')

    for i,r in enumerate(rates):
        ax[ndim+i].set_title(r)
        ax[ndim+i].set_xlabel('time')

    ax[ndim+nrates-1].legend(loc=2,frameon=False,prop={'size':10},bbox_to_anchor=(1.1,1))

    fig.suptitle(title)


def RNAkira (vals, var, NF, T, alpha=0.05, criterion='LRT', min_precursor=1, min_ribo=1, models=None, constant_genes=None, maxlevel=None, priors=None, statsmodel='gaussian'):

    """ main routine in this package: given dataframe of TPM values, variabilities, normalization factors and labeling time T,
        estimates empirical priors and fits models of increasing complexity """

    genes=vals.index
    time_points=np.unique(vals.columns.get_level_values(1))
    ntimes=len(time_points)
    nreps=len(np.unique(vals.columns.get_level_values(2)))

    ndim=7
    TPM=vals*NF

    use_precursor=(TPM['unlabeled-precursor'] > min_precursor).any(axis=1)
    use_ribo=(TPM['ribo'] > min_ribo).any(axis=1)

    # these are the features to use depending on cutoffs
    mature_cols=['elu-mature','flowthrough-mature','unlabeled-mature']
    precursor_cols=['elu-precursor','flowthrough-precursor','unlabeled-precursor']
    ribo_cols=['ribo']

    print >> sys.stderr, '[RNAkira] analyzing {0} MPR genes, {1} MR genes, {2} MP genes, {3} M genes'.format((use_precursor & use_ribo).sum(),(~use_precursor & use_ribo).sum(),(use_precursor & ~use_ribo).sum(),(~use_precursor & ~use_ribo).sum())
    print >> sys.stderr, '[RNAkira] using {0} time points, {1} replicates, {2} model'.format(ntimes,nreps,statsmodel)
    if criterion in ['LRT','AIC']:
        print >> sys.stderr, '[RNAkira] model selection using {0} criterion with alpha={1:.2g}'.format(criterion,alpha)
    elif criterion=='empirical':
        set_new_alpha=False
        if constant_genes is None:
            raise Exception("[RNAkira] cannot use empirical FDR without constant genes!")
        print >> sys.stderr, '[RNAkira] model selection using empirical FDR with alpha={1:.2g} and {2} constant genes'.format(criterion,alpha,len(constant_genes))
    else:
        print >> sys.stderr, '[RNAkira] no model selection'

    #min_args=dict(method='L-BFGS-B',jac=False,options={'disp':False, 'ftol': 1.e-15, 'gtol': 1.e-10})
    min_args=dict(method='L-BFGS-B',jac=True,options={'disp':False, 'ftol': 1.e-15, 'gtol': 1.e-10})

    if maxlevel is not None:
        nlevels=maxlevel+1
    else:
        nlevels=4

    if models is None:
        models=get_model_definitions(nlevels)

    results=defaultdict(dict)
    all_results=defaultdict(dict)
    print >> sys.stderr, '\n[RNAkira] running initial fits,',

    if priors is None:

        # initialize priors with some empirical data (use 45% reasonably expressed genes)

        print >> sys.stderr, 'estimating empirical priors'

        take_mature=((TPM['unlabeled-mature'] > TPM['unlabeled-mature'].quantile(.5)) & \
                     (TPM['unlabeled-mature'] < TPM['unlabeled-mature'].quantile(.95))).any(axis=1)
        take_precursor=take_mature & use_precursor & ((TPM['unlabeled-precursor'] > TPM['unlabeled-precursor'].quantile(.5)) & \
                                                      (TPM['unlabeled-precursor'] < TPM['unlabeled-precursor'].quantile(.95))).any(axis=1)
        take_ribo=take_mature & use_ribo & ((TPM['ribo'] > TPM['ribo'].quantile(.5)) & \
                                            (TPM['ribo'] < TPM['ribo'].quantile(.95))).any(axis=1)

        log_a=np.log((TPM['elu-mature']+TPM['elu-precursor']).divide(T,axis=0))[take_precursor].values.flatten()
        log_b=np.log(np.log(1+TPM['elu-mature']/TPM['flowthrough-mature']).divide(T,axis=0))[take_mature].values.flatten()
        log_c=np.log(np.log(1+TPM['elu-precursor']/TPM['flowthrough-precursor']).divide(T,axis=0))[take_precursor].values.flatten()
        log_d=np.log(TPM['ribo']/TPM['unlabeled-mature'])[take_ribo].values.flatten()

        model_priors=pd.DataFrame([trim_mean_std(x) for x in [log_a,log_b,log_c,log_d]],\
                                  columns=['mu','std'],index=list('abcd'))

        if model_priors.isnull().any().any():
            raise Exception("could not estimate finite model priors!")

    else:
        print >> sys.stderr, 'using given priors'
        model_priors=priors

    niter=0
    level=0
    while True:

        niter+=1

        print >> sys.stderr, '   est. priors for log_a: {0:.2g}/{1:.2g}, log_b: {2:.2g}/{3:.2g}, log_c: {4:.2g}/{5:.2g}, log_d: {6:.2g}/{7:.2g}'.format(*model_priors.values.flatten())

        nfits=0
        # fit full model for each gene
        for gene in genes:

            print >> sys.stderr, '   iter {0}, {1} fits\r'.format(niter,nfits),

            model='AB'
            cols=mature_cols[:]
            if use_precursor[gene]:
                cols+=precursor_cols[:]
                model+='C'
            if use_ribo[gene]:
                cols+=ribo_cols[:]
                model+='D'

            # make numpy arrays out of vals, var and NF for computational efficiency (therefore we need equal number of replicates for each time point!)
            vals_here=vals.loc[gene].unstack(level=0)[cols].stack().values.reshape((ntimes,nreps,len(cols)))
            var_here=var.loc[gene].unstack(level=0)[cols].stack().values.reshape((ntimes,len(cols)))
            nf_here=NF.loc[gene].unstack(level=0)[cols].stack().values.reshape((ntimes,nreps,len(cols)))

            result=fit_model (vals_here, var_here, nf_here, T[gene], time_points, model_priors.loc[list(model.lower())], None, model, statsmodel, min_args)

            results[gene][level]=result
            nfits+=1

        if priors is not None:
            print >> sys.stderr, '   done: {0} fits     \r'.format(nfits)
            break

        new_priors=pd.DataFrame([trim_mean_std(np.array([results[gene][level]['est_pars'][mp] for gene in genes \
                                                         if results[gene][level]['success'] and mp in results[gene][level]['est_pars']]))\
                                 for mp in'abcd'],columns=['mu','std'],index=list('abcd'))

        if new_priors.isnull().any().any():
            raise Exception("could not estimate finite model priors!")

        prior_diff=((model_priors-new_priors)**2).sum().sum()/(new_priors**2).sum().sum()

        print >> sys.stderr, '   iter {0}: {1} fits, prior_diff: {2:.2g}'.format(niter,nfits,prior_diff)

        if prior_diff > 1.e-3 and niter < 10:
            model_priors=new_priors
        else:
            break

    print >> sys.stderr, '\n[RNAkira] fitting model hierarchy'

    # now fit the rates using models of increasing complexity

    for level in range(1,nlevels):

        level_results=defaultdict(dict)
        for model,parent_models in models[level].iteritems():

            model_results=dict()
            nfits=0
            # do this for each gene
            for gene in genes[(use_ribo if 'd' in model.lower() else ~use_ribo) & (use_precursor if 'c' in model.lower() else ~use_precursor)]:

                cols=mature_cols[:]
                if use_precursor[gene]:
                    cols+=precursor_cols[:]
                if use_ribo[gene]:
                    cols+=ribo_cols[:]

                print >> sys.stderr, '   model: {0}, {1} fits\r'.format(model,nfits),

                # use initial estimate from previous level
                if level-1 in results[gene] and results[gene][level-1]['model'] in parent_models:
                    parent=results[gene][level-1]
                else:
                    continue

                # only compute this model if the parent model was significant
                if level > 1 and ((criterion=='LRT' and parent['LRT-q'] > alpha) or \
                                  (criterion=='AIC' and parent['dAIC'] < alpha) or \
                                  (criterion=='empirical' and parent['LRT-p'] > alpha)):
                    continue

                # make numpy arrays out of vals, var and NF for computational efficiency
                vals_here=vals.loc[gene].unstack(level=0)[cols].stack().values.reshape((ntimes,nreps,len(cols)))
                var_here=var.loc[gene].unstack(level=0)[cols].stack().values.reshape((ntimes,len(cols)))
                nf_here=NF.loc[gene].unstack(level=0)[cols].stack().values.reshape((ntimes,nreps,len(cols)))

                result=fit_model (vals_here, var_here, nf_here, T[gene], time_points, model_priors.loc[list(model.lower())], parent, model, statsmodel, min_args)
                model_results[gene]=result
                level_results[gene][model]=result
                nfits+=1

            message='   model: {0}, {1} fits'.format(model,nfits)

            pvals=dict((gene,v['LRT-p']) for gene,v in model_results.iteritems() if 'LRT-p' in v)
            qvals=dict(zip(pvals.keys(),p_adjust_bh(pvals.values())))
            for gene,q in qvals.iteritems():
                model_results[gene]['LRT-q']=q

            if criterion=='LRT':
                nsig=sum(r['LRT-q'] <= alpha for r in model_results.values())
            elif criterion=='AIC':
                nsig=sum(r['dAIC'] >= alpha for r in model_results.values())
            elif criterion=='empirical':
                if level==1 and not set_new_alpha and sum(g in model_results for g in constant_genes) > .5*len(constant_genes):
                    alpha=max(sorted(model_results[g]['LRT-p'] for g in constant_genes if g in model_results)[:int(alpha*len(constant_genes))])
                    print >> sys.stderr, '   setting empirical p-value cutoff to {0:.2g}'.format(alpha)
                    set_new_alpha=True
                nsig=sum(r['LRT-p'] <= alpha for r in model_results.values())

            if criterion in ['LRT','AIC','empirical'] and nfits > 0:
                message +=' ({0} {2} at alpha={1:.2g})'.format(nsig,alpha,'improved' if level > 1 else 'insufficient')

            print >> sys.stderr, message

        print >> sys.stderr, '   selecting best models at level {0}'.format(level)
        for gene in genes:
            all_results[gene][level]=level_results[gene]
            if len(level_results[gene]) > 0:
                results[gene][level]=max(level_results[gene].values(),key=lambda x: x['logL'])

    for gene in genes:
        max_level=max(results[gene].keys())
        if max_level > 0:
            max_fit=results[gene][max_level]
            pval=scipy.stats.chi2.sf(2*np.abs(results[gene][0]['logL']-max_fit['logL']),np.abs(results[gene][0]['npars']-max_fit['npars']))
            results[gene][0]['LRT-p']=(pval if np.isfinite(pval) else 1)
            results[gene][0]['dAIC']=np.abs(results[gene][0]['AIC']-max_fit['AIC'])/np.mean([results[gene][0]['AIC'],max_fit['AIC']])
        else:
            results[gene][0]['LRT-p']=0
            results[gene][0]['dAIC']=0

    pvals=dict((gene,v[0]['LRT-p']) for gene,v in results.iteritems() if 'LRT-p' in v[0])
    qvals=dict(zip(pvals.keys(),p_adjust_bh(pvals.values())))
    for gene,q in qvals.iteritems():
        results[gene][0]['LRT-q']=q

    if criterion=='LRT':
        nsig=sum(results[gene][0]['LRT-q'] <= alpha for gene in genes)
    elif criterion=='AIC':
        nsig=sum(results[gene][0]['dAIC'] >= alpha for gene in genes)
    elif criterion=='empirical':
        nsig=sum(results[gene][0]['LRT-p'] <= alpha for gene in genes)

    if criterion in ['LRT','AIC','empirical']:
        print >> sys.stderr, '   model: AB(CD), {0} improved at alpha={1:.2g}\n'.format(nsig,alpha)

    print >> sys.stderr, '[RNAkira] done'

    return dict((k,v.values()) for k,v in results.iteritems())

def collect_results (results, time_points, alpha=0.01):

    """ helper routine to put RNAkira results into a DataFrame """

    output=dict()
    for gene,res in results.iteritems():

        # first get initial fits
        initial_fit=res[0]
        pars=initial_fit['est_pars']
        tmp=[('initial_synthesis_t{0}'.format(t),np.exp(pars.ix[t,'a'])) for t in time_points]+\
            [('initial_degradation_t{0}'.format(t),np.exp(pars.ix[t,'b'])) for t in time_points]+\
            [('initial_processing_t{0}'.format(t),np.exp(pars.ix[t,'c']) if 'c' in pars.columns else np.nan) for t in time_points]+\
            [('initial_translation_t{0}'.format(t),np.exp(pars.ix[t,'d']) if 'd' in pars.columns else np.nan) for t in time_points]+\
            [('initial_logL',initial_fit['logL']),\
             ('initial_R2_tot',initial_fit['R2_tot']),\
             ('initial_R2_RNA',initial_fit['R2_RNA']),\
             ('initial_R2_ribo',initial_fit['R2_ribo']),\
             ('initial_fit_success',initial_fit['success']),\
             ('initial_dAIC',initial_fit['dAIC']),\
             ('initial_pval',initial_fit['LRT-p'] if 'LRT-p' in initial_fit else np.nan),\
             ('initial_qval',initial_fit['LRT-q'] if 'LRT-q' in initial_fit else np.nan),\
             ('initial_model',initial_fit['model'])]

        # take best significant model or constant otherwise
        best_fit=filter(lambda x: (x['model'].islower() or x['LRT-q'] < alpha),res)[-1]
        pars=best_fit['est_pars']
        tmp+=[('modeled_synthesis_t{0}'.format(t),np.exp(pars.ix[t,'a'])) for t in time_points]+\
            [('modeled_degradation_t{0}'.format(t),np.exp(pars.ix[t,'b'])) for t in time_points]+\
            [('modeled_processing_t{0}'.format(t),np.exp(pars.ix[t,'c']) if 'c' in pars.columns else np.nan) for t in time_points]+\
            [('modeled_translation_t{0}'.format(t),np.exp(pars.ix[t,'d']) if 'd' in pars.columns else np.nan) for t in time_points]+\
            [('modeled_logL',best_fit['logL']),\
             ('modeled_R2_tot',best_fit['R2_tot']),\
             ('modeled_R2_RNA',best_fit['R2_RNA']),\
             ('modeled_R2_ribo',best_fit['R2_ribo']),\
             ('modeled_fit_success',best_fit['success']),\
             ('modeled_dAIC',best_fit['dAIC']),\
             ('modeled_pval',best_fit['LRT-p'] if 'LRT-p' in best_fit else np.nan),\
             ('modeled_qval',best_fit['LRT-q'] if 'LRT-q' in best_fit else np.nan),\
             ('best_model',best_fit['model'])]
        
        output[gene]=OrderedDict(tmp)

    return pd.DataFrame.from_dict(output,orient='index')

def normalize_elu_flowthrough_over_genes (TPM, samples, fig_name=None):

    """ fixes library size normalization of TPM values for each sample separately """

    if fig_name is not None:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        N=np.ceil(np.sqrt(len(samples)))
        M=np.ceil(len(samples)/float(N))
        fig=plt.figure(figsize=(2*N,2*M))
        fig.subplots_adjust(bottom=.1,top=.95,hspace=.4,wspace=.3)

    print >> sys.stderr, '[normalize_elu_flowthrough_over_genes] normalizing by linear regression over genes',

    # collect correction_factors
    CF=pd.Series(1.0,index=TPM.columns)

    told=''
    for n,(t,r) in enumerate(samples):

        if t!=told:
            print >> sys.stderr, '\n   t={0}: {1}'.format(t,r),
            told=t
        else:
            print >> sys.stderr, '{0}'.format(r),

        # select reliable genes with decent mean expression level in mature fractions
        reliable_genes=(TPM['unlabeled-mature',t,r] > TPM['unlabeled-mature',t,r].dropna().quantile(.5)) &\
            (TPM['elu-mature',t,r] > TPM['elu-mature',t,r].dropna().quantile(.5)) &\
            (TPM['flowthrough-mature',t,r] > TPM['flowthrough-mature',t,r].dropna().quantile(.5))

        elu_ratio=(TPM['elu-mature',t,r]/TPM['unlabeled-mature',t,r]).replace([np.inf,-np.inf],np.nan)
        FT_ratio=(TPM['flowthrough-mature',t,r]/TPM['unlabeled-mature',t,r]).replace([np.inf,-np.inf],np.nan)

        ok=np.isfinite(elu_ratio) & np.isfinite(FT_ratio) & reliable_genes
        slope,intercept=odr_regression(elu_ratio[ok],FT_ratio[ok],[-1,1],\
                                       we=elu_ratio[ok].std(),wd=FT_ratio[ok].std())

        if intercept <= 0 or slope >= 0:
            raise Exception('invalid slope ({0:.2g}) or intercept ({1:.2g})'.format(slope,intercept))

        CF['elu-precursor',t,r]=-slope/intercept
        CF['elu-mature',t,r]=-slope/intercept

        CF['flowthrough-precursor',t,r]=1./intercept
        CF['flowthrough-mature',t,r]=1./intercept

        if fig_name is not None:

            ax=fig.add_subplot(N,M,n+1)
            bounds=np.concatenate([np.nanpercentile(elu_ratio[ok],[1,99]),
                                   np.nanpercentile(FT_ratio[ok],[1,99])])
            ax.hexbin(elu_ratio[ok],FT_ratio[ok],bins='log',lw=0,extent=bounds,cmap=plt.cm.Greys,vmin=-1,mincnt=1)
            ax.plot(np.arange(bounds[0],bounds[1],.01),intercept+slope*np.arange(bounds[0],bounds[1],.01),'r-')
            ax.set_xlim(bounds[:2])
            ax.set_ylim(bounds[2:])
            ax.set_title('t={0} ({1})'.format(t,r,ok.sum()),size=10)
            if n >= M*(N-1):
                ax.set_xlabel('elu/unlabeled')
            if n%M==0:
                ax.set_ylabel('FT/unlabeled')

    if fig_name is not None:
        print >> sys.stderr, '\n[normalize_elu_flowthrough_over_genes] saving figure to {0}'.format(fig_name),
        fig.savefig(fig_name)

    print >> sys.stderr, '\n'

    return CF

def normalize_elu_flowthrough_over_samples (TPM, constant_genes, fig_name=None):

    """ fixes library size normalization of total TPM values using constant genes """

    if fig_name is not None:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        fig=plt.figure(figsize=(4,3))

    print >> sys.stderr, '[normalize_elu_flowthrough_over_samples] normalizing using constant genes'


    CF=pd.Series(1.0,index=TPM.columns)
    # normalize TPMs to those of constant genes
    constant_frac=TPM.loc[constant_genes][['unlabeled-mature','unlabeled-precursor']].sum(axis=0).sum(level=[1,2]).mean()
    for c in ['elu','flowthrough','unlabeled']:
        CF[c+'-mature']=constant_frac/TPM.loc[constant_genes][[c+'-mature',c+'-precursor']].sum(axis=0).sum(level=[1,2])
        CF[c+'-precursor']=constant_frac/TPM.loc[constant_genes][[c+'-mature',c+'-precursor']].sum(axis=0).sum(level=[1,2])
    TPM1=TPM.multiply(CF,axis=1)

    slope,intercept=odr_regression(TPM1['elu-mature'].sum(axis=0)/TPM1['unlabeled-mature'].sum(axis=0),\
                                   TPM1['flowthrough-mature'].sum(axis=0)/TPM1['unlabeled-mature'].sum(axis=0))

    if slope > 0 or intercept < 0:
        raise Exception('invalid slope ({0:.2f}) or intercept ({1:.2f}) in normalize_elu_flowthrough_over_samples!'.format(slope,intercept))

    CF['elu-mature'] = -slope*CF['elu-mature']/intercept
    CF['elu-precursor'] = -slope*CF['elu-precursor']/intercept
    CF['flowthrough-mature'] = CF['flowthrough-mature']/intercept
    CF['flowthrough-precursor'] = CF['flowthrough-precursor']/intercept

    if fig_name is not None:

        ax=fig.add_axes([.15,.15,.8,.75])
        for c in TPM['unlabeled-mature'].columns.get_level_values(0).unique():
            ax.plot(TPM1['elu-mature',c].sum(axis=0)/TPM1['unlabeled-mature',c].sum(axis=0),\
                    TPM1['flowthrough-mature',c].sum(axis=0)/TPM1['unlabeled-mature',c].sum(axis=0),'o',label=c)
        xlim=ax.get_xlim()
        ax.plot(np.linspace(xlim[0],xlim[1],20),intercept+slope*np.linspace(xlim[0],xlim[1],20),'k--')
        ax.set_xlabel('elu/total')
        ax.set_ylabel('flowthrough/total')
        ax.legend()
        ax.set_title('normalizing using constant genes',size=10)

        print >> sys.stderr, '[normalize_elu_flowthrough_over_samples] saving figure to {0}'.format(fig_name)
        fig.savefig(fig_name)

    print >> sys.stderr, ''

    return CF

def correct_ubias (TPM, gene_stats, fig_name=None):

    """ do LOWESS regression of log2 elu ratio against log10 ucounts and correct to values for highest 10% of ucounts """

    if fig_name is not None:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        fig=plt.figure(figsize=(4,3))

    print >> sys.stderr, '[correct_ubias] correcting U bias using LOWESS'

    # select reliable genes with decent mean expression level in mature fractions
    reliable_genes=(gene_stats['gene_type']=='protein_coding') & \
        (TPM['unlabeled-mature'] > TPM['unlabeled-mature'].dropna().quantile(.5)).any(axis=1) &\
        (TPM['elu-mature'] > TPM['elu-mature'].dropna().quantile(.5)).any(axis=1)

    # use stacked data frames (= all samples together)
    log2_elu_ratio=np.log2(TPM['elu-mature']/TPM['unlabeled-mature']).stack(level=[0,1],dropna=False).replace([np.inf,-np.inf],np.nan)
    log2_FT_ratio=np.log2(TPM['flowthrough-mature']/TPM['unlabeled-mature']).stack(level=[0,1],dropna=False).replace([np.inf,-np.inf],np.nan)

    stacked_index=log2_elu_ratio.index

    reliable_genes=reliable_genes.loc[stacked_index.get_level_values(0)]
    reliable_genes.index=stacked_index

    ucount=gene_stats['exon_ucount'].loc[stacked_index.get_level_values(0)]
    ucount.index=stacked_index

    ok=np.isfinite(log2_elu_ratio) & reliable_genes & np.isfinite(ucount)
    ucount_range=np.abs(ucount[ok].max()-ucount[ok].min())
    lowess=statsmodels.nonparametric.smoothers_lowess.lowess(log2_elu_ratio[ok],ucount[ok],\
                                                             frac=0.66,it=1,delta=.01*ucount_range).T
    interp_elu=scipy.interpolate.interp1d(lowess[0],lowess[1],bounds_error=False)

    ucorr=interp_elu(ucount)-np.mean(interp_elu(np.nanpercentile(ucount,np.arange(90,99))))
    
    UF=pd.DataFrame(1,index=TPM.index,columns=TPM.columns)
    UF['elu-mature']=pd.Series(2**(-ucorr),index=stacked_index).unstack(level=1).unstack().fillna(1)

    if fig_name is not None:

        ax=fig.add_axes([.15,.15,.8,.8])
        bounds=np.concatenate([np.nanpercentile(ucount[ok],[1,99]),
                               np.nanpercentile(log2_elu_ratio[ok],[1,99])])
        ax.hexbin(ucount[ok],log2_elu_ratio[ok],bins='log',extent=bounds,lw=0,cmap=plt.cm.Greys,vmin=-1,mincnt=1)
        plt.plot(np.linspace(bounds[0],bounds[1],100),interp_elu(np.linspace(bounds[0],bounds[1],100)),'r-')
        ax.set_xlim(bounds[:2])
        ax.set_ylim(bounds[2:])
        ax.set_ylabel('log2 elu/unlabeled')
        ax.set_xlabel('# U residues')

        print >> sys.stderr, '[correct_ubias] saving figure to {0}'.format(fig_name)
        fig.savefig(fig_name)

    print >> sys.stderr, ''

    return UF

def estimate_dispersion (counts, fig_name=None, weight=1.8):

    """ estimates dispersion by a weighted average of the calculated dispersion and a smoothened trend """

    cols=counts.columns.get_level_values(0).unique()
    time_points=counts.columns.get_level_values(1).unique()
    ntimes=len(time_points)
    disp=pd.DataFrame(1.0,index=counts.index,columns=pd.MultiIndex.from_product([cols,time_points]))
    reps=counts.columns.get_level_values(2).unique()
    nreps=len(reps)

    if fig_name is not None:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        fig=plt.figure(figsize=(3*len(cols),3))
        fig.subplots_adjust(left=.05,right=.98,hspace=.4,wspace=.3,bottom=.15)

    print >> sys.stderr, '[estimate_dispersion] averaging samples:'
    for n,c in enumerate(cols):

        print >> sys.stderr, ' '*3+c

        mean_counts_within=counts[c].mean(axis=1,level=0)
        var_counts_within=counts[c].var(axis=1,level=0)
        # get dispersion trend for genes with positive values
        ok=(mean_counts_within.values > 0) &\
            (mean_counts_within.values < var_counts_within.values)
        y=(var_counts_within/mean_counts_within).values[ok]
        X=np.c_[mean_counts_within.values[ok],np.ones(ok.sum())]
        wls=statsmodels.regression.linear_model.WLS(y,X,weights=1./y)
        slope_within,intercept_within=wls.fit().params
        disp_smooth=(intercept_within-1)/mean_counts_within+slope_within
        disp_smooth[disp_smooth < 0]=0
        disp_act_within=((var_counts_within-mean_counts_within)/mean_counts_within**2).replace([np.inf,-np.inf],np.nan)
        # estimated dispersion is weighted average of real and smoothened
        disp_est=(1.-weight/nreps)*disp_act_within+weight*disp_smooth/nreps

        # if estimated dispersion is negative or NaN, use across-sample estimate
        replace=((disp_est < 0) | disp_est.isnull()).any(axis=1)
        if replace.sum() > 0:

            mean_counts_all=counts[c].mean(axis=1)
            var_counts_all=counts[c].var(axis=1)
            # get dispersion trend for genes with positive values
            ok=(mean_counts_all > 0) & (mean_counts_all < var_counts_all)
            y=(var_counts_all/mean_counts_all)[ok]
            X=np.c_[mean_counts_all[ok],np.ones(ok.sum())]
            wls=statsmodels.regression.linear_model.WLS(y,X,weights=1./y)
            slope_all,intercept_all=wls.fit().params
            disp_smooth=(intercept_all-1)/mean_counts_all+slope_all
            disp_act_all=(var_counts_all-mean_counts_all)/mean_counts_all**2
            repl=((1.-weight/nreps)*disp_act_all+weight*disp_smooth/nreps)[replace]
            disp_est[replace]=pd.concat([repl]*ntimes,axis=1,keys=time_points)

        # NaN values only where mean = 0
        disp[c]=disp_est.fillna(1)

        if fig_name is not None:

            ax=fig.add_subplot(1,len(cols),n+1)
            x,y=np.log10(mean_counts_within).values.flatten(),np.log10(disp_act_within).values.flatten()
            bounds=np.concatenate([np.percentile(x[np.isfinite(x)],[1,99]),
                                   np.percentile(y[np.isfinite(y)],[1,99])])
            if not np.all(np.isfinite(bounds)):
                raise Exception("bounds not finite")
            ax.hexbin(x,y,bins='log',lw=0,extent=bounds,cmap=plt.cm.Greys,vmin=-1,mincnt=1)
            # plot estimated values
            ax.plot(np.log10(mean_counts_within).values.flatten()[::10],np.log10(disp_est).values.flatten()[::10],'c.',markersize=2)
            # plot trend
            ax.plot(np.arange(bounds[0],bounds[1],.1),np.log10((intercept_within-1)/10**np.arange(bounds[0],bounds[1],.1)+slope_within),'r-')
            ax.set_xlim(bounds[:2])
            ax.set_ylim(bounds[2:])
            ax.set_title('{0}'.format(c))
            ax.set_xlabel('log10 mean')
            ax.set_ylabel('log10 disp')

    if fig_name is not None:
        print >> sys.stderr, '[estimate_dispersion] saving figure to {0}'.format(fig_name)
        fig.savefig(fig_name)

    print >> sys.stderr, ''

    return disp

def estimate_stddev (TPM, fig_name=None, weight=1.8):

    """ estimates stddev by a weighted average of the calculated std dev and a smoothened std dev from the mean-variance plot """

    cols=TPM.columns.get_level_values(0).unique()
    time_points=TPM.columns.get_level_values(1).unique()
    ntimes=len(time_points)
    stddev=pd.DataFrame(1.0,index=TPM.index,columns=pd.MultiIndex.from_product([cols,time_points]))
    reps=TPM.columns.get_level_values(2).unique()
    nreps=len(reps)

    if fig_name is not None:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        fig=plt.figure(figsize=(3*len(cols),3))
        fig.subplots_adjust(left=.05,right=.98,hspace=.4,wspace=.3,bottom=.15)

    print >> sys.stderr, '[estimate_stddev] averaging samples:'
    for n,c in enumerate(cols):

        print >> sys.stderr, ' '*3+c

        # first do within-group stddev
        log10_std_within=np.log10(TPM[c].std(axis=1,level=0)).replace([np.inf,-np.inf],np.nan)
        # perform lowess regression on log CV vs log mean
        log10_means_within=np.log10(TPM[c].mean(axis=1,level=0)).replace([np.inf,-np.inf],np.nan)
        log10_CV_within=log10_std_within-log10_means_within
        ok=np.isfinite(log10_means_within) & np.isfinite(log10_CV_within)
        log10_mean_range=np.abs(log10_means_within[ok].max().max()-log10_means_within[ok].min().min())
        lowess_within=statsmodels.nonparametric.smoothers_lowess.lowess(log10_CV_within[ok].values.flatten(),\
                                                                        log10_means_within[ok].values.flatten(),\
                                                                        frac=0.6,it=1,delta=.01*log10_mean_range).T
        # this interpolates log10 CV from log10 mean
        interp_within=scipy.interpolate.interp1d(lowess_within[0],lowess_within[1],bounds_error=False)

        # get interpolated std dev within groups
        log10_std_smooth=interp_within(log10_means_within)+log10_means_within
        # estimated std dev is weighted average of real and smoothened
        log10_std_est=(1.-weight/nreps)*log10_std_within+weight*log10_std_smooth/nreps
        replace=log10_std_est.isnull().any(axis=1)

        if replace.sum() > 0:

            # do stddev over all samples
            log10_std_all=np.log10(TPM[c].std(axis=1))
            # perform lowess regression on log CV vs log mean
            log10_means_all=np.log10(TPM[c].mean(axis=1))
            log10_CV_all=log10_std_all-log10_means_all
            ok=np.isfinite(log10_means_all) & np.isfinite(log10_CV_all)
            log10_mean_range=np.abs(log10_means_all[ok].max()-log10_means_all[ok].min())
            lowess_all=statsmodels.nonparametric.smoothers_lowess.lowess(log10_CV_all[ok].values,log10_means_all[ok].values,\
                                                                         frac=0.6,it=1,delta=.01*log10_mean_range).T
            interp_all=scipy.interpolate.interp1d(lowess_all[0],lowess_all[1],bounds_error=False)

            repl=((1.-weight/nreps)*log10_std_all+weight*(interp_all(log10_means_all)+log10_means_all)/nreps)[replace]
            log10_std_est[replace]=pd.concat([repl]*ntimes,axis=1,keys=time_points)

        # add estimated std dev (NaN values only where mean=0)
        stddev[c]=10**log10_std_est.fillna(1)

        if fig_name is not None:

            # get regularized CV
            log10_CV_est=log10_std_est-log10_means_within
            ax=fig.add_subplot(1,len(cols),n+1)
            x,y=log10_means_within.values.flatten(),log10_CV_within.values.flatten()
            bounds=np.concatenate([np.percentile(x[np.isfinite(x)],[1,99]),
                                   np.percentile(y[np.isfinite(y)],[1,99])])
            ax.hexbin(x,y,bins='log',lw=0,extent=bounds,cmap=plt.cm.Greys,vmin=-1,mincnt=1)
            # plot estimated values
            ax.plot(log10_means_within.values.flatten()[::10],log10_CV_est.values.flatten()[::10],'c.',markersize=2)
            ax.plot(np.arange(bounds[0],bounds[1],.1),interp_within(np.arange(bounds[0],bounds[1],.1)),'r-')
            ax.set_xlim(bounds[:2])
            ax.set_ylim(bounds[2:])
            ax.set_title('{0}'.format(c))
            ax.set_xlabel('log10 mean')
            ax.set_ylabel('log10 CV')

    if fig_name is not None:
        print >> sys.stderr, '[estimate_stddev] saving figure to {0}'.format(fig_name)
        fig.savefig(fig_name)

    print >> sys.stderr, ''

    return stddev

def read_featureCounts_output (infiles,samples):

    """ reads in featureCount output file(s), extracts counts and lengths and adds samples as column names """

    counts=[]
    length=[]
    for inf in infiles.split(','):
        fc_table=pd.read_csv(inf,sep='\t',comment='#',index_col=0,header=0)
        cnts=fc_table.ix[:,5:].astype(int)
        cnts.columns=pd.MultiIndex.from_tuples(samples)
        counts.append(cnts)
        length.append(fc_table.ix[:,4]/1.e3)

    return pd.concat(counts,axis=0),pd.concat(length,axis=0)

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
    parser.add_option('-T','--labeling_time',dest='T',help="labeling time (either a number or a csv file)")
    parser.add_option('-o','--out_prefix',dest='out_prefix',default='RNAkira',help="output prefix [RNAkira]")
    parser.add_option('','--alpha',dest='alpha',help="model selection cutoff [0.05]",default=0.05,type=float)
    parser.add_option('','--criterion',dest='criterion',help="model selection criterion [LRT]",default="LRT")
    parser.add_option('','--constant_genes',dest='constant_genes',help="list of constant genes for empirical FDR calculcation")
    parser.add_option('','--maxlevel',dest='maxlevel',help="max level to test [4]",default=4,type=int)
    parser.add_option('','--min_mature',dest='min_mature',help="min TPM for mature [1]",default=1,type=float)
    parser.add_option('','--min_precursor',dest='min_precursor',help="min TPM for precursor [.1]",default=.1,type=float)
    parser.add_option('','--min_ribo',dest='min_ribo',help="min TPM for ribo [1]",default=1,type=float)
    parser.add_option('','--weight',dest='weight',help="weighting parameter for stddev estimation (should be smaller than number of replicates) [1.8]",default=1.8,type=float)
    parser.add_option('','--no_plots',dest='no_plots',help="don't create plots for U-bias correction and normalization",action='store_false')
    parser.add_option('','--save_normalization_factors',dest='save_normalization_factors',action='store_true',default=False,help="""save normalization factors from elu/flowthrough regression [no]""")
    parser.add_option('','--normalize_over_samples',dest='normalize_over_samples',action='store_true',default=False,help="""normalize elu vs. flowthrough over samples using constant genes""")
    parser.add_option('','--save_TPM',dest='save_TPM',action='store_true',default=False,help="""save TPM values [no]""")
    parser.add_option('','--statsmodel',dest='statsmodel',help="statistical model to use (gaussian or nbinom) [nbinom]",default='nbinom')

    # ignore warning about division by zero or over-/underflows
    np.seterr(divide='ignore',over='ignore',under='ignore',invalid='ignore')

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

        # neg binom doesn't work here
        options.statsmodel='gaussian'

        # size factors are 1
        SF=pd.Series(1.0,index=TPM.columns)
        # length factors are 1
        LF=pd.DataFrame(1.0,index=TPM.index,columns=TPM.columns.get_level_values(0).unique())

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

        print >> sys.stderr, "[main] merging count values and normalizing"

        cols=['elu-mature','flowthrough-mature','unlabeled-mature','elu-precursor','flowthrough-precursor','unlabeled-precursor','ribo']

        counts=pd.concat([elu_exons,flowthrough_exons,unlabeled_exons,\
                          elu_introns,flowthrough_introns,unlabeled_introns,\
                          ribo],axis=1,keys=cols).sort_index(axis=1).fillna(0)

        # combine length factors
        LF=pd.concat([elu_exon_length,flowthrough_exon_length,unlabeled_exon_length,\
                      elu_intron_length,flowthrough_intron_length,unlabeled_intron_length,\
                      ribo_length],axis=1,keys=cols).fillna(1)

        # compute RPK
        RPK=counts.divide(LF,axis=0,level=0,fill_value=1)

        # for size factors, add up counts for different fractions (introns and exons), count missing entries as zero
        EF=RPK['elu-mature'].add(RPK['elu-precursor'],fill_value=0).sum(axis=0)/1.e6
        FF=RPK['flowthrough-mature'].add(RPK['flowthrough-precursor'],fill_value=0).sum(axis=0)/1.e6
        UF=RPK['unlabeled-mature'].add(RPK['unlabeled-precursor'],fill_value=0).sum(axis=0)/1.e6
        # for ribo, sum only CDS RPK
        RF=RPK['ribo'].sum(axis=0)/1.e6
        # combine size factors
        SF=pd.concat([EF,FF,UF,\
                      EF,FF,UF,\
                      RF],axis=0,keys=cols)
        
        TPM=RPK.divide(SF,axis=1).fillna(0)

        if options.save_TPM:
            print >> sys.stderr, '[main] saving TPM values to '+options.out_prefix+'_TPM.csv'
            TPM.to_csv(options.out_prefix+'_TPM.csv')

    if options.criterion=='empirical' or options.normalize_over_samples:
        constant_genes=[line.split()[0] for line in open(options.constant_genes)]
        print >> sys.stderr, '[main] using {0} constant genes from {1}'.format(len(constant_genes),options.constant_genes)
    else:
        constant_genes=None

    # correction factors
    print >> sys.stderr, '[main] correcting TPM values using gene stats from '+options.gene_stats
    gene_stats=pd.concat([pd.read_csv(gsfile,index_col=0,header=0) for gsfile in options.gene_stats.split(',')],axis=0).loc[TPM.index]
    UF=correct_ubias(TPM,gene_stats,fig_name=(None if options.no_plots else options.out_prefix+'_ubias_correction.pdf'))

    if options.normalize_over_samples:
        CF=normalize_elu_flowthrough_over_samples (TPM.multiply(UF), constant_genes,\
                                                   fig_name=(None if options.no_plots else options.out_prefix+'_normalization.pdf'))
    else:
        CF=normalize_elu_flowthrough_over_genes (TPM.multiply(UF), samples,\
                                                 fig_name=(None if options.no_plots else options.out_prefix+'_normalization.pdf'))

    # normalization factor combines size factors with TPM correction 
    if options.save_normalization_factors:
        print >> sys.stderr, '[main] saving normalization factors to '+options.out_prefix+'_normalization_factors.csv'
        UF.multiply(CF).divide(SF,axis=1).to_csv(options.out_prefix+'_normalization_factors.csv')

    NF=UF.multiply(CF).divide(LF,axis=0,level=0,fill_value=1).divide(SF,axis=1).fillna(1)
    TPM=TPM.multiply(UF).multiply(CF)

    print >> sys.stderr, '[main] estimating variability'
    if options.statsmodel=='nbinom':
        # estimate dispersion based on library-size-normalized counts but keep scales (divide by geometric mean per assay)
        variability=estimate_dispersion (counts.divide(SF.divide(np.exp(np.log(SF).mean(level=0)),level=0),axis=1), \
                                         fig_name=(None if options.no_plots else options.out_prefix+'_variability.pdf'), weight=options.weight)
    else:
        variability=estimate_stddev (TPM, fig_name=(None if options.no_plots else options.out_prefix+'_variability.pdf'), weight=options.weight)

    # select genes based on TPM cutoffs for mature in any of the time points
    take=(TPM['unlabeled-mature'] > options.min_mature).any(axis=1) 

    try:
        T=float(options.T)
        T=pd.Series(T,index=TPM.index)
    except:
        T=pd.read_csv(options.T,index_col=0,header=None).squeeze()[TPM.index]
        if T.isnull().sum() > 0:
            raise Exception("{0} genes have no labeling time in {1}".format(T.isnull().sum(),options.T))

    if options.input_TPM is None:
        results=RNAkira(counts[take].fillna(0), variability[take], NF[take], T[take], \
                        alpha=options.alpha, criterion=options.criterion, constant_genes=np.intersect1d(constant_genes,TPM[take].index),\
                        min_precursor=options.min_precursor, min_ribo=options.min_ribo,\
                        maxlevel=options.maxlevel, statsmodel=options.statsmodel)
    else:
        # now TPMs already include correction factors, so use NF=1
        results=RNAkira(TPM[take].fillna(0), variability[take], pd.DataFrame(1.0,index=NF.index,columns=NF.columns)[take], T[take], \
                        alpha=options.alpha, criterion=options.criterion, constant_genes=np.intersect1d(constant_genes,TPM[take].index),\
                        min_precursor=options.min_precursor, min_ribo=options.min_ribo,\
                        maxlevel=options.maxlevel, statsmodel=options.statsmodel)

    print >> sys.stderr, '[main] collecting output'
    output=collect_results(results, time_points, alpha=options.alpha)

    print >> sys.stderr, '       writing results to {0}'.format(options.out_prefix+'_results.csv')

    output.to_csv(options.out_prefix+'_results.csv')

