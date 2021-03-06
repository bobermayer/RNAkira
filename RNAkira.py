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

def get_model_definitions(nlevels,take_all=True):

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
            elif level==2 and not take_all:
                for pc in itertools.combinations(pars,1):
                    level_models.update({pars.translate(maketrans(''.join(pc),''.join(pc).upper())): pars})
                for pc in itertools.combinations(pars,len(pars)-1):
                    level_models.update({pars.translate(maketrans(''.join(pc),''.join(pc).upper())): pars})
            elif take_all:
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

def get_steady_state_values (x, T, model, use_flowthrough=True, use_deriv=False):

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
        exp_mean_mature=[a*(b*(1.-np.exp(-c*T))-c*(1.-np.exp(-b*T)))/(b*(b-c))]
        if use_flowthrough:
            exp_mean_mature+=[a*(b*np.exp(-c*T)-c*np.exp(-b*T))/(b*(b-c))]
        exp_mean_mature+=[a/b]

        # add elu-precursor, flowthrough-precursor, unlabeled-precursor
        exp_mean_precursor=[a*(1.-np.exp(-c*T))/c]
        if use_flowthrough:
            exp_mean_precursor+=[a*np.exp(-c*T)/c]
        exp_mean_precursor+=[a/c]

        exp_mean=exp_mean_mature+exp_mean_precursor

    else:

        if use_ribo:
            a,b,d=np.exp(x)
        else:
            a,b=np.exp(x)

        # for elu-mature, flowthrough-mature, unlabeled-mature
        exp_mean_mature=[a*(1.-np.exp(-b*T))/b]
        if use_flowthrough:
            exp_mean_mature+=[a*np.exp(-b*T)/b]
        exp_mean_mature+=[a/b]

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
                                            (2*b-c)*(b*(1.-np.exp(-c*T))-c*(1.-np.exp(-b*T))))/(b*(b-c)**2)]
            if use_flowthrough:
                exp_mean_mature_deriv_log_b+=[a*(b*(b-c)*(T*c*np.exp(-b*T)+np.exp(-c*T))-(2*b-c)*(b*np.exp(-c*T)-c*np.exp(-b*T)))/(b*(b-c)**2)]
            exp_mean_mature_deriv_log_b+=[-a/b]

            # get deriv of mature w.r.t. log(c)
            exp_mean_mature_deriv_log_c=[a*c*((b-c)*(b*T*np.exp(-c*T)-1.+np.exp(-b*T))+\
                                              (b*(1.-np.exp(-c*T))-c*(1.-np.exp(-b*T))))/(b*(b-c)**2)]
            if use_flowthrough:
                exp_mean_mature_deriv_log_c+=[a*c*((b-c)*(-np.exp(-b*T)-T*b*np.exp(-c*T))+(b*np.exp(-c*T)-c*np.exp(-b*T)))/(b*(b-c)**2)]
            exp_mean_mature_deriv_log_c+=[0]

            exp_mean_mature_deriv=[exp_mean_mature_deriv_log_a,\
                                   exp_mean_mature_deriv_log_b,\
                                   exp_mean_mature_deriv_log_c]

            # then get derivatives of precursor w.r.t. log(a),log(b),log(c)
            exp_mean_precursor_deriv=[exp_mean_precursor[:]]
            if use_flowthrough:
                exp_mean_precursor_deriv+=[[0,0,0],\
                                           [a*(np.exp(-c*T)*(1.+c*T)-1)/c,\
                                            -a*np.exp(-c*T)*(1.+c*T)/c,\
                                            -a/c]]
            else:
                exp_mean_precursor_deriv+=[[0,0],\
                                           [a*(np.exp(-c*T)*(1.+c*T)-1)/c,\
                                            -a/c]]

            exp_mean_deriv=[exp_mean_mature_deriv[i]+exp_mean_precursor_deriv[i] for i in range(3)]

            if use_ribo:
                # add derivatives w.r.t. log(d)
                exp_mean_deriv_log_d=[[0]*(4+2*use_flowthrough)]
                exp_mean_deriv+=exp_mean_deriv_log_d
                # add derivatives of ribo values w.r.t. log(a),log(b),log(c),log(d)
                exp_mean_ribo_deriv=[[d*a/b],[-d*a/b],[0],[d*a/b]]
                exp_mean_deriv=[exp_mean_deriv[i]+exp_mean_ribo_deriv[i] for i in range(4)]

        else:

            # get the derivatives of mature vals w.r.t. log(b)
            exp_mean_mature_deriv_log_b=[a*(np.exp(-b*T)*(1.+b*T)-1)/b]
            if use_flowthrough:
                exp_mean_mature_deriv_log_b+=[-a*np.exp(-b*T)*(1.+b*T)/b]
            exp_mean_mature_deriv_log_b+=[-a/b]
        
            exp_mean_deriv=[exp_mean_mature_deriv_log_a,\
                            exp_mean_mature_deriv_log_b]

            if use_ribo:
                # add derivatives w.r.t. log(d)
                exp_mean_mature_deriv_log_d=[0]*(2+use_flowthrough)
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

def get_rates (x, nconds, model):

    """ expands log rates over conditions given rate parameters x, number of conditions, and model """

    nrates=len(model)

    # get instantaneous rates a,b,c,d at each condition by expanding the ones that don't vary 
    log_rates=np.zeros((nrates,nconds))
    k=0
    for n,mp in enumerate(model):
        # uppercase letters in the model definition mean this parameter varies
        if mp.isupper():
            log_rates[n]=x[k:k+nconds]
            k+=nconds
        else:
            log_rates[n]=x[k]
            k+=1
    
    return log_rates.T
        
def steady_state_log_likelihood (x, vals, var, nf, T, conditions, prior_mu, prior_std, prior_weight, model, statsmodel, use_flowthrough, use_deriv):

    """ log-likelihood function for difference between expected and observed values, including all priors """

    nconds=len(conditions)

    fun=0
    log_rates=get_rates(x, nconds, model)
    if use_deriv:
        grad=np.zeros(len(x))
        # index array for gradient vector knows on which parameters vary 
        gg=defaultdict(list)
        k=0
        for n,mp in enumerate(model):
            if mp.isupper():
                for i,cond in enumerate(conditions):
                    gg[cond].append(k+i)
                k+=nconds
            else:
                for cond in conditions:
                    gg[cond].append(k)
                k+=1
        gg=dict((k,np.array(v)) for k,v in gg.iteritems())

    # add up model log-likelihoods for each condition and each replicate
    for i,cond in enumerate(conditions):

        # first add to log-likelihood the priors on rates at each condition (normal distribution with mu and std)
        diff=log_rates[i]-prior_mu
        fun+=prior_weight*np.sum(-.5*(diff/prior_std)**2-.5*np.log(2.*np.pi)-np.log(prior_std))
        
        if use_deriv:
            # add derivatives of rate parameters
            grad[gg[cond]]+=prior_weight*(-diff/prior_std**2)
            
        # get expected mean values (and their derivatives)
        if use_deriv:
            exp_mean,exp_mean_deriv=get_steady_state_values(log_rates[i],T[i],model,use_flowthrough,use_deriv)
        else:
            exp_mean=get_steady_state_values(log_rates[i],T[i],model,use_flowthrough)

        # add gaussian or neg. binomial log-likelihood 
        if statsmodel=='gaussian':
            # here "var" is stddev
            diff=vals[i]-exp_mean/nf[i]
            fun+=np.sum(scipy.stats.norm.logpdf(diff,scale=var/nf[i]))
            if use_deriv:
                grad[gg[cond]]+=np.dot(exp_mean_deriv,np.sum(diff*nf[i]/var**2,axis=0))
                
        elif statsmodel=='nbinom':
            # here "var" is dispersion
            mu=exp_mean/nf[i]
            nn=1./var
            pp=1./(1.+var*mu)
            fun+=np.sum(scipy.stats.nbinom.logpmf(vals[i],nn,pp))
            if use_deriv:
                tmp=(vals[i]-nn*var*mu)/(exp_mean*(1.+var*mu))
                grad[gg[cond]]+=np.dot(exp_mean_deriv,np.sum(tmp,axis=0))

    if fun is np.nan:
        raise Exception("invalid value in steady_state_log_likelihood!")

    if use_deriv:
        # return negative log likelihood and gradient
        return (-fun,-grad)
    else:
        # return negative log likelihood
        return -fun

def steady_state_residuals (x, vals, nf, T, conditions, model, use_flowthrough):

    """ residuals between observed and expected values for each fraction separately"""

    nconds=len(conditions)
    log_rates=get_rates(x, nconds, model)
    ncols=(2+use_flowthrough)+(2+use_flowthrough)*('c' in model.lower())+('d' in model.lower())

    res=np.zeros(ncols)
    # add up model residuals for each condition and each replicate
    for i in range(nconds):
        exp_mean=get_steady_state_values(log_rates[i],T[i],model,use_flowthrough)
        res+=np.sum((vals[i]*nf[i]-exp_mean)**2,axis=0)

    if res is np.nan:
        raise Exception("invalid value in steady_state_residuals")

    return res

def fit_model (vals, var, nf, T, conditions, priors, prior_weight, parent, model, statsmodel, use_flowthrough, min_args):

    """ fits a specific model to data given variance, normalization factors, labeling times, conditions, 
    priors, initial estimate from a parent model etc. """

    nconds=len(conditions)
    nrates=sum(nconds if mp.isupper() else 1 for mp in model)
    ncols=(2+use_flowthrough)+(2+use_flowthrough)*('c' in model.lower())+('d' in model.lower())

    if parent is not None:
        if model.lower()==model:
            initial_estimate=parent['est_pars'].mean(axis=0)
        else:
            initial_estimate=np.concatenate([parent['est_pars'][mp.lower()] if mp.isupper() else [parent['est_pars'][mp.lower()].mean()] for mp in model])
    else:
        # use initial estimate from priors
        initial_estimate=np.repeat(priors['mu'].values,nconds)

    # arguments to minimization
    args=(vals, var, nf, T, conditions, priors['mu'].values, priors['std'].values, prior_weight,  model, statsmodel, use_flowthrough, min_args['jac'])

    # test gradient against numerical difference for debugging 
    if False:

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
    SSres=steady_state_residuals(res.x,vals,nf,T,conditions,model,use_flowthrough)
    mean_vals=np.sum(np.sum(vals*nf,axis=0),axis=0)/np.prod(vals.shape[:2])
    SStot=np.sum(np.sum((vals*nf-mean_vals)**2,axis=0),axis=0)

    # collect result
    rates=get_rates(res.x, nconds, model)
    rate_errs=get_rates(np.sqrt(np.diag(res.hess_inv.todense())), nconds, model)
    result=dict(est_pars=pd.DataFrame(rates,columns=list(model.lower()),index=conditions),\
                est_errs=pd.DataFrame(rate_errs,columns=list(model.lower()),index=conditions),\
                logL=-res.fun,\
                R2_tot=1-SSres.sum()/np.sum((vals*nf-np.mean(vals*nf))**2),\
                R2_frac=(1-SSres/SStot),
                AIC=2*(len(res.x)+1+res.fun),\
                success=res.success,\
                significant=False,\
                message=res.message,\
                npars=len(res.x),
                model=model)
    
    if parent is not None:
        # calculate p-value from LRT test using chi2 distribution
        pval=scipy.stats.chi2.sf(2*np.abs(result['logL']-parent['logL']),np.abs(result['npars']-parent['npars']))
        result['LRT-p']=(pval if np.isfinite(pval) else np.nan)
        # get aggregated log fold change of inferred parameters between these models
        result['tot_LFC']=(result['est_pars']-parent['est_pars']).abs().sum().sum()
    else:
        result['LRT-p']=np.nan
        
    return result

def RNAkira (vals, var, NF, T, alpha=0.05, LFC_cutoff=0, no_model_selection=False, min_precursor=.1, min_ribo=1, \
             use_flowthrough=True, models=None, maxlevel=None, priors=None, prior_weight=1, statsmodel='nbinom'):

    """ main routine in this package: given dataframe of TPM values, variabilities, normalization factors 
    and labeling time T, estimates empirical priors and fits models of increasing complexity """

    genes=vals.index
    conditions=np.unique(vals.columns.get_level_values(1))
    nconds=len(conditions)
    nreps=len(np.unique(vals.columns.get_level_values(2)))
    
    ndim=7
    TPM=vals.multiply(NF)

    # these are the features to use depending on cutoffs
    if use_flowthrough:
        mature_cols=['elu-mature','flowthrough-mature','unlabeled-mature']
        precursor_cols=['elu-precursor','flowthrough-precursor','unlabeled-precursor']
    else:
        mature_cols=['elu-mature','unlabeled-mature']
        precursor_cols=['elu-precursor','unlabeled-precursor']
    ribo_cols=['ribo']

    use_precursor=(TPM['unlabeled-precursor'] > min_precursor).any(axis=1) & ~var[precursor_cols].isnull().any(axis=1)
    use_ribo=(TPM['ribo'] > min_ribo).any(axis=1) & ~var[ribo_cols].isnull().any(axis=1)

    print >> sys.stderr, '[RNAkira] {0:5d} genes with mature+precursor+ribo'.format((use_precursor & use_ribo).sum())
    print >> sys.stderr, '          {0:5d} genes with mature+ribo'.format((~use_precursor & use_ribo).sum())
    print >> sys.stderr, '          {0:5d} genes with mature+precursor'.format((use_precursor & ~use_ribo).sum())
    print >> sys.stderr, '          {0:5d} genes with mature only'.format((~use_precursor & ~use_ribo).sum())
    print >> sys.stderr, '[RNAkira] {0} conditions, {1} replicates, {2} model'.format(nconds,nreps,statsmodel)
    if not no_model_selection:
        print >> sys.stderr, '[RNAkira] model selection: LRT with alpha={0:.2g} and LFC>={1:.2f}'.format(alpha,LFC_cutoff)
    else:
        print >> sys.stderr, '[RNAkira] no model selection'

    min_args=dict(method='L-BFGS-B',jac=True,options={'disp':False, 'ftol': 1.e-15, 'gtol': 1.e-10})

    if models is None:
        if not no_model_selection:
            nlevels=6 if maxlevel is None else maxlevel+1
            models=get_model_definitions(nlevels,take_all=True)
        else:
            nlevels=3
            models=get_model_definitions(nlevels,take_all=False)

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
        log_a=np.log((TPM['elu-mature']+TPM['elu-precursor']).divide(T,axis=0,level=0))[take_precursor].values.flatten()

        if use_flowthrough:
            log_b=np.log(np.log(1+TPM['elu-mature']/TPM['flowthrough-mature']).divide(T,axis=0,level=0))[take_mature].values.flatten()
            log_c=np.log(np.log(1+TPM['elu-precursor']/TPM['flowthrough-precursor']).divide(T,axis=0,level=0))[take_precursor].values.flatten()
        else:
            log_b=np.log(-np.log(TPM['elu-mature']/TPM['unlabeled-mature']).divide(T,axis=0,level=0))[take_mature].values.flatten()
            log_c=np.log(-np.log(TPM['elu-precursor']/TPM['unlabeled-precursor']).divide(T,axis=0,level=0))[take_precursor].values.flatten()

        all_pars='abc'
        prior_msg='   est. priors for log_a: {0:.2g}/{1:.2g}, log_b: {2:.2g}/{3:.2g}, log_c: {4:.2g}/{5:.2g}'
        est_vals=[log_a,log_b,log_c]

        if use_ribo.sum() > 0:
            all_pars+='d'
            prior_msg+=', log_d: {6:.2g}/{7:.2g}'
            take_ribo=take_mature & use_ribo & ((TPM['ribo'] > TPM['ribo'].quantile(.5)) & \
                                                (TPM['ribo'] < TPM['ribo'].quantile(.95))).any(axis=1)
            log_d=np.log(TPM['ribo']/TPM['unlabeled-mature'])[take_ribo].values.flatten()
            est_vals+=[log_d]

        model_priors=pd.DataFrame([trim_mean_std(x) for x in est_vals],\
                                  columns=['mu','std'],index=list(all_pars))

        if model_priors.isnull().any().any():
            raise Exception("could not estimate finite model priors!")

    else:
        print >> sys.stderr, 'using given priors'
        prior_msg='   priors for log_a: {0:.2g}/{1:.2g}, log_b: {2:.2g}/{3:.2g}, log_c: {4:.2g}/{5:.2g}'
        if use_ribo.sum() > 0:
            prior_msg+=', log_d: {6:.2g}/{7:.2g}'
        model_priors=priors

    niter=0
    level=0
    while True:

        niter+=1

        print >> sys.stderr, prior_msg.format(*model_priors.values.flatten())

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

            # make numpy arrays out of vals, var and NF for computational efficiency 
            # (therefore we need equal number of replicates for each condition!)
            vals_here=vals.loc[gene].unstack(level=0)[cols].stack().values.reshape((nconds,nreps,len(cols)))
            var_here=var.loc[gene][cols].values
            nf_here=NF.loc[gene].unstack(level=0)[cols].stack().values.reshape((nconds,nreps,len(cols)))

            result=fit_model (vals_here, var_here, nf_here, T.loc[gene], conditions, model_priors.loc[list(model.lower())], \
                              prior_weight, None, model, statsmodel, use_flowthrough, min_args)

            results[gene][level]=result
            all_results[gene][level]={model: result}
            nfits+=1

        if priors is not None:
            print >> sys.stderr, '   done: {0} fits     \r'.format(nfits)
            break

        new_priors=pd.DataFrame([trim_mean_std(np.array([results[gene][level]['est_pars'][mp] for gene in genes \
                                                         if results[gene][level]['success'] and mp in results[gene][level]['est_pars']]))\
                                 for mp in all_pars],columns=['mu','std'],index=list(all_pars))

        if new_priors.isnull().any().any():
            raise Exception("could not estimate finite model priors!")

        prior_diff=((model_priors['mu']-new_priors['mu'])**2).sum().sum()/(new_priors['mu']**2).sum().sum()

        print >> sys.stderr, '   iter {0}: {1} fits, prior_diff: {2:.2g}'.format(niter,nfits,prior_diff)

        if prior_diff > 1.e-3 and niter < 10:
            model_priors=new_priors
        else:
            break

    print >> sys.stderr, '\n[RNAkira] fitting models'

    # now fit the rates using models of increasing complexity

    for level in range(1,nlevels):

        level_results=defaultdict(dict)
        for model,parent_models in models[level].iteritems():

            model_results=dict()
            nfits=0
            # do this for each gene
            for gene in genes[(use_ribo if 'd' in model.lower() else ~use_ribo) &\
                                  (use_precursor if 'c' in model.lower() else ~use_precursor)]:

                cols=mature_cols[:]
                if use_precursor[gene]:
                    cols+=precursor_cols[:]
                if use_ribo[gene]:
                    cols+=ribo_cols[:]

                print >> sys.stderr, '   model: {0}, {1} fits\r'.format(model,nfits),

                # get best parent model if available for initial estimate and p-value calculation
                if level-1 in results[gene] and results[gene][level-1]['model'] in parent_models: 
                    parent=results[gene][level-1]
                else:
                    continue

                # for model selection: only compute this model if the parent model was significant
                if level > 1 and not no_model_selection and not parent['significant']:
                    continue

                if model.isupper():
                    # this is identical to initial model, so no computation necessary
                    result=results[gene][0].copy()
                    result['significant']=False
                    pval=scipy.stats.chi2.sf(2*np.abs(result['logL']-parent['logL']),np.abs(result['npars']-parent['npars']))
                    result['LRT-p']=(pval if np.isfinite(pval) else np.nan)
                    result['tot_LFC']=(result['est_pars']-parent['est_pars']).abs().sum().sum()
                else:
                    # make numpy arrays out of vals, var and NF for computational efficiency
                    vals_here=vals.loc[gene].unstack(level=0)[cols].stack().values.reshape((nconds,nreps,len(cols)))
                    var_here=var.loc[gene][cols].values
                    nf_here=NF.loc[gene].unstack(level=0)[cols].stack().values.reshape((nconds,nreps,len(cols)))
                    result=fit_model (vals_here, var_here, nf_here, T.loc[gene], conditions, model_priors.loc[list(model.lower())],\
                                      prior_weight, parent, model, statsmodel, use_flowthrough, min_args)

                model_results[gene]=result
                level_results[gene][model]=result
                nfits+=1

            pvals=dict((gene,v['LRT-p']) for gene,v in model_results.iteritems() if 'LRT-p' in v)
            qvals=dict(zip(pvals.keys(),p_adjust_bh(pvals.values())))

            # determine which models are significantly better
            nsig=0
            for gene,q in qvals.iteritems():
                model_results[gene]['LRT-q']=q
                if (not no_model_selection and model_results[gene]['LRT-q'] <= alpha and \
                        model_results[gene]['tot_LFC'] >= LFC_cutoff):
                    model_results[gene]['significant']=True
                    nsig+=1

            # print status 
            message='   model: {0}, {1} fits'.format(model,nfits)
            if not no_model_selection:
                message +=' ({0} {2} at alpha={1:.2g} and LFC>={3:.2g})'.format(nsig,alpha,'improved' if level > 1 else 'insufficient',LFC_cutoff)
            if nfits > 0:
                print >> sys.stderr, message

        for gene in genes:
            all_results[gene][level]=level_results[gene]
            # select best model at this level
            if len(level_results[gene]) > 0:
                results[gene][level]=max(all_results[gene][level].values(),key=lambda x: x['logL'])

    print >> sys.stderr, '[RNAkira] done'

    if not no_model_selection:
        return dict((gene,rr.values()) for gene,rr in results.iteritems())
    else:
        return dict((gene,[r for lv,rr in lr.iteritems() for m,r in rr.iteritems()]) for gene,lr in all_results.iteritems())

def collect_results (results, conditions, select_best=False):

    """ helper routine to put RNAkira results into a DataFrame """

    output=dict()
    for gene,res in results.iteritems():

        if select_best:

            # first get initial fits
            initial_fit=res[0]
            pars=initial_fit['est_pars']
            errs=initial_fit['est_errs']
            tmp=[('initial_synthesis_{0}'.format(cond),pars.loc[cond,'a']) for cond in conditions]+\
                [('initial_synthesis_err_{0}'.format(cond),errs.loc[cond,'a']) for cond in conditions]+\
                [('initial_degradation_{0}'.format(cond),pars.loc[cond,'b']) for cond in conditions]+\
                [('initial_degradation_err_{0}'.format(cond),errs.loc[cond,'b']) for cond in conditions]+\
                [('initial_processing_{0}'.format(cond),pars.loc[cond,'c'] if 'c' in pars.columns else np.nan) for cond in conditions]+\
                [('initial_processing_err_{0}'.format(cond),errs.loc[cond,'c'] if 'c' in pars.columns else np.nan) for cond in conditions]+\
                [('initial_translation_{0}'.format(cond),pars.loc[cond,'d'] if 'd' in pars.columns else np.nan) for cond in conditions]+\
                [('initial_translation_err_{0}'.format(cond),errs.loc[cond,'d'] if 'd' in pars.columns else np.nan) for cond in conditions]+\
                [('initial_logL',initial_fit['logL']),\
                 ('initial_R2_tot',initial_fit['R2_tot']),\
                 ('initial_R2_frac',';'.join(map(str,np.round(initial_fit['R2_frac'],3)))),\
                 ('initial_fit_success',initial_fit['success']),\
                 ('initial_AIC',initial_fit['AIC']),\
                 ('initial_pval',initial_fit['LRT-p'] if 'LRT-p' in initial_fit else np.nan),\
                 ('initial_qval',initial_fit['LRT-q'] if 'LRT-q' in initial_fit else np.nan),\
                 ('initial_model',initial_fit['model'])]

            # take best significant model or constant otherwise
            best_fit=filter(lambda x: (x['model'].islower() or x['significant']),res)[-1]
            pars=best_fit['est_pars']
            errs=best_fit['est_errs']
            tmp+=[('modeled_synthesis_{0}'.format(cond),pars.loc[cond,'a']) for cond in conditions]+\
               [('modeled_synthesis_err_{0}'.format(cond),errs.loc[cond,'a']) for cond in conditions]+\
                [('modeled_degradation_{0}'.format(cond),pars.loc[cond,'b']) for cond in conditions]+\
                [('modeled_degradation_err_{0}'.format(cond),errs.loc[cond,'b']) for cond in conditions]+\
                [('modeled_processing_{0}'.format(cond),pars.loc[cond,'c'] if 'c' in pars.columns else np.nan) for cond in conditions]+\
                [('modeled_processing_err_{0}'.format(cond),errs.loc[cond,'c'] if 'c' in pars.columns else np.nan) for cond in conditions]+\
                [('modeled_translation_{0}'.format(cond),pars.loc[cond,'d'] if 'd' in pars.columns else np.nan) for cond in conditions]+\
                [('modeled_translation_err_{0}'.format(cond),errs.loc[cond,'d'] if 'd' in pars.columns else np.nan) for cond in conditions]+\
                [('modeled_logL',best_fit['logL']),\
                 ('modeled_R2_tot',best_fit['R2_tot']),\
                 ('modeled_R2_frac',';'.join(map(str,np.round(best_fit['R2_frac'],3)))),\
                 ('modeled_fit_success',best_fit['success']),\
                 ('modeled_AIC',best_fit['AIC']),\
                 ('modeled_pval',best_fit['LRT-p'] if 'LRT-p' in best_fit else np.nan),\
                 ('modeled_qval',best_fit['LRT-q'] if 'LRT-q' in best_fit else np.nan),\
                 ('tot_LFC',best_fit['tot_LFC'] if 'tot_LFC' in best_fit else np.nan),\
                 ('best_model',best_fit['model'])]

        else:

            tmp=[]
            for r in res:
                model=r['model']
                pars=r['est_pars']
                errs=r['est_errs']
                tmp+=[(model+'_synthesis_{0}'.format(cond),pars.loc[cond,'a']) for cond in conditions]+\
                [(model+'_synthesis_err_{0}'.format(cond),errs.loc[cond,'a']) for cond in conditions]+\
                [(model+'_degradation_{0}'.format(cond),pars.loc[cond,'b']) for cond in conditions]+\
                [(model+'_degradation_err_{0}'.format(cond),errs.loc[cond,'b']) for cond in conditions]+\
                [(model+'_processing_{0}'.format(cond),pars.loc[cond,'c'] if 'c' in pars.columns else np.nan) for cond in conditions]+\
                [(model+'_processing_err_{0}'.format(cond),errs.loc[cond,'c'] if 'c' in pars.columns else np.nan) for cond in conditions]+\
                [(model+'_translation_{0}'.format(cond),pars.loc[cond,'d'] if 'd' in pars.columns else np.nan) for cond in conditions]+\
                [(model+'_translation_err_{0}'.format(cond),errs.loc[cond,'d'] if 'd' in pars.columns else np.nan) for cond in conditions]+\
                [(model+'_logL',r['logL']),\
                 (model+'_R2_tot',r['R2_tot']),\
                 (model+'_R2_frac',';'.join(map(str,np.round(r['R2_frac'],3)))),\
                 (model+'_fit_success',r['success']),\
                 (model+'_AIC',r['AIC']),\
                 (model+'_pval',r['LRT-p'] if 'LRT-p' in r else np.nan),\
                 (model+'_qval',r['LRT-q'] if 'LRT-q' in r else np.nan)]
        
        output[gene]=OrderedDict(tmp)

    return pd.DataFrame.from_dict(output,orient='index')

def normalize_elu_flowthrough (TPM, samples, gene_stats, balance_normalization_factors=False, fig_name=None):

    """ fixes library size normalization of TPM values for each sample separately """

    if fig_name is not None:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        N=np.ceil(np.sqrt(len(samples)))
        M=np.ceil(len(samples)/float(N))
        fig=plt.figure(figsize=(2*N,2*M))
        fig.subplots_adjust(bottom=.1,top=.95,hspace=.4,wspace=.3)

    print >> sys.stderr, '[normalize_elu_flowthrough] normalizing by linear regression over genes'

    # collect correction_factors
    CF=pd.Series(1.0,index=TPM.columns)


    for n,(c,r) in enumerate(samples):

        elu_ratio=(TPM['elu-mature',c,r]/TPM['unlabeled-mature',c,r]).replace([np.inf,-np.inf],np.nan)
        FT_ratio=(TPM['flowthrough-mature',c,r]/TPM['unlabeled-mature',c,r]).replace([np.inf,-np.inf],np.nan)

        # select protein-coding genes with decent expression level in mature fractions
        reliable_genes=(gene_stats['gene_type']=='protein_coding') & \
            (TPM[['unlabeled-mature','elu-mature']].xs((c,r),axis=1,level=[1,2]) > 1).all(axis=1)

        ok=np.isfinite(elu_ratio) & np.isfinite(FT_ratio) & reliable_genes
        slope,intercept=odr_regression(elu_ratio[ok],FT_ratio[ok],[-1,1])

        if intercept <= 0 or slope >= 0:
            raise Exception('invalid slope ({0:.2g}) or intercept ({1:.2g})'.format(slope,intercept))

        CF['elu-precursor',c,r]=-slope/intercept
        CF['elu-mature',c,r]=-slope/intercept

        CF['flowthrough-precursor',c,r]=1./intercept
        CF['flowthrough-mature',c,r]=1./intercept

        if fig_name is not None:

            ax=fig.add_subplot(N,M,n+1)
            bounds=np.concatenate([np.nanpercentile(elu_ratio[ok],[1,99]),
                                   np.nanpercentile(FT_ratio[ok],[1,99])])
            ax.hexbin(elu_ratio[ok],FT_ratio[ok],bins='log',lw=0,extent=bounds,cmap=plt.cm.Greys,vmin=-1,mincnt=1)
            ax.plot(np.arange(bounds[0],bounds[1],.01),intercept+slope*np.arange(bounds[0],bounds[1],.01),'r-')
            ax.set_xlim(bounds[:2])
            ax.set_ylim(bounds[2:])
            ax.set_title('{0}/{1} (n={2})'.format(c,r,ok.sum()),size=10)
            if n >= M*(N-1):
                ax.set_xlabel('elu/unlabeled')
            if n%M==0:
                ax.set_ylabel('FT/unlabeled')

    if balance_normalization_factors:

        print >> sys.stderr, '[normalize_elu_flowthrough] balancing correction factors over samples'

        reliable_genes=(gene_stats['gene_type']=='protein_coding') & \
            (TPM[['unlabeled-mature','elu-mature']] > 1).all(axis=1)

        newTPM=TPM.multiply(CF)

        elu_ratios=newTPM['elu-mature'].divide(newTPM['unlabeled-mature'],axis=0).loc[reliable_genes].mean(axis=0)
        flowthrough_ratios=newTPM['flowthrough-mature'].divide(newTPM['unlabeled-mature'],axis=0).loc[reliable_genes].mean(axis=0)

        CF['elu-mature']=CF['elu-mature'].divide(elu_ratios/elu_ratios.mean())
        CF['elu-precursor']=CF['elu-precursor'].divide(elu_ratios/elu_ratios.mean())

        CF['flowthrough-mature']=CF['flowthrough-mature'].divide(flowthrough_ratios/flowthrough_ratios.mean())
        CF['flowthrough-precursor']=CF['flowthrough-precursor'].divide(flowthrough_ratios/flowthrough_ratios.mean())

    if fig_name is not None:
        print >> sys.stderr, '[normalize_elu_flowthrough] saving figure to {0}'.format(fig_name)
        fig.savefig(fig_name)

    return CF

def normalize_elu (TPM, gene_stats, constant_genes):

    """ fixes library size normalization of total TPM values using constant genes """

    print >> sys.stderr, '[normalize_elu] normalizing using constant genes'

    CF=pd.Series(1.0,index=TPM.columns)

    # select reliable genes with decent expression level in mature fractions
    # select protein-coding genes with decent expression level in mature fractions
    reliable_genes=(gene_stats['gene_type']=='protein_coding') & \
        (TPM[['unlabeled-mature','elu-mature']] > 1).all(axis=1)

    # normalize elu TPMs to those of constant genes
    cTPM=TPM.loc[constant_genes].sum(axis=0)/TPM.loc[reliable_genes].sum(axis=0)
    elu_ratios=cTPM['elu-mature']/cTPM['unlabeled-mature']

    CF['elu-mature']=elu_ratios.mean()/elu_ratios
    CF['elu-precursor']=elu_ratios.mean()/elu_ratios

    return CF

def correct_ubias (TPM, samples, gene_stats, fig_name=None):

    """ model log2 elu ratio as function of log10 ucounts and correct to asymptotic values """

    if fig_name is not None:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        N=np.ceil(np.sqrt(len(samples)))
        M=np.ceil(len(samples)/float(N))
        fig=plt.figure(figsize=(2*N,2*M))
        fig.subplots_adjust(bottom=.1,top=.95,hspace=.4,wspace=.3)

    print >> sys.stderr, '[correct_ubias] correcting U bias using least squares regression'

    UF=pd.DataFrame(1,index=TPM.index,columns=TPM.columns)

    for n,(c,r) in enumerate(samples):

        # select reliable genes with decent mean expression level in mature fractions
        reliable_genes=(gene_stats['gene_type']=='protein_coding') & \
            (TPM[['unlabeled-mature','elu-mature']].xs((c,r),axis=1,level=[1,2]) > 1).all(axis=1)

        log2_elu_ratio=np.log2(TPM['elu-mature',c,r]/TPM['unlabeled-mature',c,r]).replace([np.inf,-np.inf],np.nan)
        log2_FT_ratio=np.log2(TPM['flowthrough-mature',c,r]/TPM['unlabeled-mature',c,r]).replace([np.inf,-np.inf],np.nan)

        ucount=gene_stats['exon_ucount']

        ok=np.isfinite(log2_elu_ratio) & reliable_genes & np.isfinite(ucount)

        theo_ucorr = lambda p, x, y: p[0]+np.log2(1-p[1]*np.exp(-p[2]*x))-y
        (alpha,beta,gamma),success=scipy.optimize.leastsq(theo_ucorr, [0,.5,.001], args=(ucount[ok],log2_elu_ratio[ok]))
        ucorr=np.log2(1.-beta*np.exp(-gamma*ucount))

        UF['elu-mature',c,r]=2**(-ucorr)

        if fig_name is not None:

            ax=fig.add_subplot(N,M,n+1)
            bounds=np.concatenate([np.nanpercentile(ucount[ok],[1,99]),
                                   np.nanpercentile(log2_elu_ratio[ok],[1,99])])
            ax.hexbin(ucount[ok],log2_elu_ratio[ok],bins='log',extent=bounds,lw=0,cmap=plt.cm.Greys,vmin=-1,mincnt=1)
            plt.plot(np.linspace(bounds[0],bounds[1],100),alpha+np.log2(1.-beta*np.exp(-gamma*np.linspace(bounds[0],bounds[1],100))),'r-')
            ax.set_xlim(bounds[:2])
            ax.set_ylim(bounds[2:])
            ax.set_title('{0}/{1} (n={2})'.format(c,r,ok.sum()),size=10)
            if n >= M*(N-1):
                ax.set_xlabel('# U residues')
            if n%M==0:
                ax.set_ylabel('log2 elu/unlabeled')

    if fig_name is not None:
        print >> sys.stderr, '[correct_ubias] saving figure to {0}'.format(fig_name)
        fig.savefig(fig_name)

    return UF

def estimate_dispersion (counts, weight, fig_name=None):

    """ estimates dispersion by a weighted average of the calculated dispersion and a smoothened trend """

    if fig_name is not None:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        fig=plt.figure(figsize=(4,3))

    print >> sys.stderr, '[estimate_dispersion] fitting dispersion trend'

    # mean counts: averaged first within conditions, then between conditions
    mean_counts=counts.mean(axis=1,level=[0,1]).mean(axis=1,level=0)
    # within-condition variances
    var_counts=counts.var(axis=1,level=[0,1])
    # bare dispersions: geometric average between condition of within-condition variances and mean counts
    log_disp_act=np.log(var_counts.sub(mean_counts,axis=0,level=0).divide(mean_counts**2,axis=0,level=0)).replace([np.inf,-np.inf],np.nan).mean(axis=1,level=0)
    log_mean=np.log(mean_counts).replace([np.inf,-np.inf],np.nan)

    bounds=np.concatenate([np.nanpercentile(log_mean,[1,99]),
                           np.nanpercentile(log_disp_act,[1,99])])
    if not np.all(np.isfinite(bounds)):
        raise Exception("bounds not finite")

    ok=(log_mean.values > bounds[0]) &\
        (log_mean.values < bounds[1]) &\
        (log_disp_act.values > bounds[2]) &\
        (log_disp_act.values < bounds[3])
    # get dispersion trend: theta = alpha/mu + beta, fit in log space
    theo_disp = lambda p, x, y: np.log(p[0]/np.exp(x)+p[1])-y
    (alpha,beta),success=scipy.optimize.leastsq(theo_disp, [1,.02], args=(log_mean.values[ok],log_disp_act.values[ok]))
    if alpha < 0 or beta < 0 or success==0:
        raise Exception("invalid parameters in estimate_dispersion")
    log_disp_smooth=np.log(alpha/np.exp(log_mean)+beta).replace([np.inf,-np.inf],np.nan)
    # estimated dispersion is weighted geometric average of actual dispersion and smoothened trend
    log_disp=(1.-weight)*log_disp_act + weight*log_disp_smooth
    # replace NaN values by smooth estimate
    take=log_disp.isnull()
    log_disp[take]=log_disp_smooth[take]

    if fig_name is not None:

        ax=fig.add_axes([.2,.15,.75,.8])
        ax.hexbin(log_mean.values[ok],log_disp_act.values[ok],bins='log',lw=0,extent=bounds,cmap=plt.cm.Greys,vmin=-1,mincnt=1)
        # plot estimated dispersion
        n=max(1000,ok.sum()/10)
        ax.plot(log_mean.values[ok][:n],log_disp.values[ok][:n],'c.',markersize=1)
        # plot trend
        lmc=np.linspace(bounds[0],bounds[1],100)
        ax.plot(lmc,np.log(alpha/np.exp(lmc)+beta),'r-')
        ax.set_xlim(bounds[:2])
        ax.set_ylim(bounds[2:])
        ax.set_xlabel('log mean')
        ax.set_ylabel('log disp')

    if fig_name is not None:
        print >> sys.stderr, '[estimate_dispersion] saving figure to {0}'.format(fig_name)
        fig.savefig(fig_name)

    return np.exp(log_disp)

def estimate_stddev (TPM, weight, fig_name=None):

    """ estimates stddev by a weighted average of calculated stddev and a smoothened trend """

    if fig_name is not None:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        fig=plt.figure(figsize=(4,3))

    print >> sys.stderr, '[estimate_stddev] fitting stddev trend'

    # do stddev within-condition 
    log_std=np.log(TPM.std(axis=1,level=[0,1])).replace([np.inf,-np.inf],np.nan)

    # perform lowess regression on log CV vs log mean
    log_means=np.log(TPM.mean(axis=1,level=[0,1]).mean(axis=1,level=0))
    log_CV=log_std.sub(log_means,axis=0,level=0).mean(axis=1,level=0)
    ok=np.isfinite(log_means.values) & np.isfinite(log_CV.values)
    log_mean_range=np.abs(log_means.values[ok].max()-log_means.values[ok].min())
    lowess=statsmodels.nonparametric.smoothers_lowess.lowess(log_CV.values[ok],log_means.values[ok],\
                                                             frac=0.6,it=1,delta=.01*log_mean_range).T
    interp=scipy.interpolate.interp1d(lowess[0],lowess[1],bounds_error=False)
    log_std_smooth=(interp(log_means)+log_means).replace([np.inf,-np.inf],np.nan)
    # estimated stddev is weighted average of actual stddev and smoothened trend
    log_std_est=(1.-weight)*(log_CV+log_means)+weight*log_std_smooth
    # replace NaN values by smooth estimate
    take=log_std_est.isnull()
    log_std_est[take]=log_std_smooth[take]

    if fig_name is not None:

        ax=fig.add_axes([.2,.15,.75,.8])
        # get regularized CV
        log_CV_est=log_std_est-log_means
        x,y=log_means.values[ok],log_CV.values[ok]
        bounds=np.concatenate([np.percentile(x[np.isfinite(x)],[1,99]),
                               np.percentile(y[np.isfinite(y)],[1,99])])
        ax.hexbin(x,y,bins='log',lw=0,extent=bounds,cmap=plt.cm.Greys,vmin=-1,mincnt=1)
        # plot estimated values for a subset
        take=np.random.choice(np.arange(len(x)),len(x)/10,replace=False)
        ax.plot(x[take],log_CV_est.values[ok][take],'c.',markersize=1)
        # plot trend
        lmc=np.linspace(bounds[0],bounds[1],100)
        ax.plot(lmc,interp(lmc),'r-')
        ax.set_xlim(bounds[:2])
        ax.set_ylim(bounds[2:])
        ax.set_xlabel('log mean')
        ax.set_ylabel('log CV')

    if fig_name is not None:
        print >> sys.stderr, '[estimate_stddev] saving figure to {0}'.format(fig_name)
        fig.savefig(fig_name)

    return np.exp(log_std_est)

def read_featureCounts_output (infiles,samples):

    """ reads in featureCount output file(s), extracts counts and lengths and adds samples as column names """

    counts=[]
    length=[]
    for inf in infiles.split(','):
        fc_table=pd.read_csv(inf,sep='\t',comment='#',index_col=0,header=0)
        cnts=fc_table.iloc[:,5:].astype(int)
        cnts.columns=pd.MultiIndex.from_tuples(samples)
        counts.append(cnts)
        length.append(fc_table.iloc[:,4]/1.e3)

    return pd.concat(counts,axis=0),pd.concat(length,axis=0)

if __name__ == '__main__':

    parser=OptionParser()
    parser.add_option('-g','--gene_stats',dest='gene_stats',help="gene stats file (created by prepare_annotation.py)")
    parser.add_option('-i','--input_TPM',dest='input_TPM',help='csv file with TPM values')
    parser.add_option('-e','--elu_introns',dest='elu_introns',help="featureCounts output for eluate RNA mapped to introns")
    parser.add_option('-E','--elu_exons',dest='elu_exons',help="featureCounts output for eluate RNA mapped to exons")
    parser.add_option('-f','--flowthrough_introns',dest='flowthrough_introns',help="featureCounts output for flowthrough RNA mapped to introns")
    parser.add_option('-F','--flowthrough_exons',dest='flowthrough_exons',help="featureCounts output for flowthrough RNA mapped to exons")
    parser.add_option('-r','--ribo_CDS',dest='ribo',help="featureCounts output for RPF mapped to CDS")
    parser.add_option('-u','--unlabeled_introns',dest='unlabeled_introns',help="featureCounts output for unlabeled RNA mapped to introns")
    parser.add_option('-U','--unlabeled_exons',dest='unlabeled_exons',help="featureCounts output for unlabeled RNA mapped to exons")
    parser.add_option('-c','--conditions',dest='conditions',help="comma-separated list of conditions (integer or floating-point), MUST correspondg to the data columns in the featureCount outputs and be the same for all fractions")
    parser.add_option('-T','--labeling_time',dest='T',help="labeling time (either a number or a csv file)")
    parser.add_option('-o','--out_prefix',dest='out_prefix',default='RNAkira_',help="output prefix [RNAkira]")
    parser.add_option('','--alpha',dest='alpha',help="model selection cutoff [0.05]",default=0.05,type=float)
    parser.add_option('','--LFC_cutoff',dest='LFC_cutoff',help="model selection LFC cutoff [0]",default=0,type=float)
    parser.add_option('','--maxlevel',dest='maxlevel',help="max level to test [5]",default=5,type=int)
    parser.add_option('','--min_mature',dest='min_mature',help="min TPM for mature [1]",default=1,type=float)
    parser.add_option('','--min_precursor',dest='min_precursor',help="min TPM for precursor [.1]",default=.1,type=float)
    parser.add_option('','--min_ribo',dest='min_ribo',help="min TPM for ribo [1]",default=1,type=float)
    parser.add_option('','--weight',dest='weight',help="weighting parameter for dispersion estimation [1]",default=1.,type=float)
    parser.add_option('','--no_plots',dest='no_plots',help="don't create plots for U-bias correction and normalization",action='store_false')
    parser.add_option('','--no_model_selection',dest='no_model_selection',action='store_true',default=False,help="no model selection")
    parser.add_option('','--save_normalization_factors',dest='save_normalization_factors',action='store_true',default=False,help="""save normalization factors from elu/flowthrough regression [no]""")
    parser.add_option('','--save_variability',dest='save_variability',action='store_true',default=False,help="""save variability estimates [no]""")
    parser.add_option('','--normalize_with_constant_genes',dest='normalize_with_constant_genes',help="""normalize elu using constant genes (provide a list)""")
    parser.add_option('','--save_TPM',dest='save_TPM',action='store_true',default=False,help="""save TPM values [no]""")
    parser.add_option('','--statsmodel',dest='statsmodel',help="statistical model to use (gaussian or nbinom) [nbinom]",default='nbinom')
    parser.add_option('','--prior_weight',dest='prior_weight',help="prior weight in log likelihood [1]",default=1,type=float)
    parser.add_option('','--balance_normalization_factors',dest='balance_normalization_factors',action='store_true',default=False,help="""balance normalization factors across samples [no]""")

    # ignore warning about division by zero or over-/underflows
    np.seterr(divide='ignore',over='ignore',under='ignore',invalid='ignore')

    options,args=parser.parse_args()

    conditions=options.conditions.split(',')

    if len(set(Counter(conditions).values())) > 1:
        raise Exception("unequal number of replicates at different conditions; can't deal with that")

    nreps=Counter(conditions)[conditions[0]]

    samples=[]
    for cond in conditions:
        samples.append((cond,'Rep'+str(sum(x[0]==cond for x in samples)+1)))

    nsamples=len(samples)
    conditions=list(OrderedDict.fromkeys(conditions))

    if options.input_TPM is not None:
        
        print >> sys.stderr, '\n[main] reading TPM values from '+options.input_TPM
        print >> sys.stderr, '       ignoring options -eEfFruU'
        TPM=pd.read_csv(options.input_TPM,index_col=0,header=range(3))
        counts=TPM

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

        print >> sys.stderr, '   unlabeled-introns:\t'+options.unlabeled_introns
        unlabeled_introns,unlabeled_intron_length=read_featureCounts_output(options.unlabeled_introns,samples)

        print >> sys.stderr, '   unlabeled-exons:\t'+options.unlabeled_exons
        unlabeled_exons,unlabeled_exon_length=read_featureCounts_output(options.unlabeled_exons,samples)

        if options.flowthrough_introns is not None:
            print >> sys.stderr, '   flowthrough-introns:\t'+options.flowthrough_introns
            flowthrough_introns,flowthrough_intron_length=read_featureCounts_output(options.flowthrough_introns,samples)
        else:
            print >> sys.stderr, '   flowthrough-introns:\t\tno values given!'
            flowthrough_introns=pd.DataFrame(np.nan,index=unlabeled_introns.index,columns=unlabeled_introns.columns)
            flowthrough_intron_length=pd.Series(1,index=unlabeled_intron_length.index)

        if options.flowthrough_exons is not None:
            print >> sys.stderr, '   flowthrough-exons:\t'+options.flowthrough_exons
            use_flowthrough=True
            flowthrough_exons,flowthrough_exon_length=read_featureCounts_output(options.flowthrough_exons,samples)
        else:
            use_flowthrough=False
            print >> sys.stderr, '   flowthrough-exons:\t\tno values given!'
            flowthrough_exons=pd.DataFrame(np.nan,index=unlabeled_exons.index,columns=unlabeled_exons.columns)
            flowthrough_exon_length=pd.Series(1,index=unlabeled_exon_length.index)

        if options.ribo is not None:
            print >> sys.stderr, '   ribo:\t\t'+options.ribo
            ribo,ribo_length=read_featureCounts_output(options.ribo,samples)
        else:
            print >> sys.stderr, '   ribo:\t\tno values given!'
            ribo=pd.DataFrame(np.nan,index=unlabeled_exons.index,columns=unlabeled_exons.columns)
            ribo_length=pd.Series(1,index=unlabeled_exon_length.index)

        print >> sys.stderr, "[main] merging count values and normalizing"

        mature_cols=['elu-mature','flowthrough-mature','unlabeled-mature']
        precursor_cols=['elu-precursor','flowthrough-precursor','unlabeled-precursor']
        ribo_cols=['ribo']
        cols=mature_cols+precursor_cols+ribo_cols

        counts=pd.concat([elu_exons,flowthrough_exons,unlabeled_exons,\
                          elu_introns,flowthrough_introns,unlabeled_introns,\
                          ribo],axis=1,keys=cols).sort_index(axis=1)

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
        
        TPM=RPK.divide(SF,axis=1)

        if options.save_TPM:
            print >> sys.stderr, '[main] saving TPM values to '+options.out_prefix+'TPM.csv'
            TPM.to_csv(options.out_prefix+'TPM.csv',\
                       header=['.'.join(col) for col in TPM.columns],tupleize_cols=True)

    if not use_flowthrough or options.normalize_with_constant_genes:
        constant_genes=[line.split()[0] for line in open(options.normalize_with_constant_genes)]
        print >> sys.stderr, '[main] using {0} constant genes from {1}'.format(len(constant_genes),options.normalize_with_constant_genes)
    else:
        constant_genes=None

    # correction factors
    print >> sys.stderr, '\n[main] correcting U bias using gene stats from '+options.gene_stats
    gene_stats=pd.concat([pd.read_csv(gsfile,index_col=0,header=0) for gsfile in options.gene_stats.split(',')],axis=0).loc[TPM.index]
    UF=correct_ubias(TPM,samples,gene_stats,fig_name=(None if options.no_plots else options.out_prefix+'ubias_correction.pdf'))

    print >> sys.stderr, '\n[main] normalizing TPMs'
    if not use_flowthrough or options.normalize_with_constant_genes:
        CF=normalize_elu (TPM.multiply(UF), gene_stats, constant_genes)
    else:
        CF=normalize_elu_flowthrough (TPM.multiply(UF), samples, gene_stats,\
                                                 balance_normalization_factors=options.balance_normalization_factors,\
                                                 fig_name=(None if options.no_plots else options.out_prefix+'normalization.pdf'))

    # normalization factor combines size factors with TPM correction 
    NF=UF.multiply(CF).divide(LF,axis=0,level=0,fill_value=1).divide(SF,axis=1).fillna(1)
    TPM=counts.multiply(NF)

    if options.save_normalization_factors:
        print >> sys.stderr, '[main] saving normalization factors to '+options.out_prefix+'normalization_factors.csv'
        UF.multiply(CF).divide(SF,axis=1).fillna(1).to_csv(options.out_prefix+'normalization_factors.csv',\
                                                           header=['.'.join(col) for col in NF.columns.tolist()],tupleize_cols=True)

    print >> sys.stderr, '\n[main] estimating variability'
    if options.statsmodel=='nbinom':
        # estimate dispersion based on library-size-normalized counts but keep scales (divide by geometric mean per assay)
        nf_scaled=NF.divide(np.exp(np.log(NF).mean(axis=1,level=0)),axis=0,level=0)
        variability=estimate_dispersion (counts.multiply(nf_scaled,axis=1), options.weight/float(nreps), \
                                         fig_name=(None if options.no_plots else options.out_prefix+'variability.pdf'))
    else:
        variability=estimate_stddev (TPM, options.weight/float(nreps), \
                                     fig_name=(None if options.no_plots else options.out_prefix+'variability.pdf'))

    if options.save_variability:
        print >> sys.stderr, '[main] saving variability estimates to '+options.out_prefix+'variability.csv'
        variability.to_csv(options.out_prefix+'variability.csv')

    # select genes based on TPM cutoffs for mature in any of the conditions
    take=(TPM['unlabeled-mature'] > options.min_mature).any(axis=1) & ~np.in1d(TPM.index,constant_genes)
    if use_flowthrough:
        take=take & ~variability[['unlabeled-mature','elu-mature','flowthrough-mature']].isnull().any(axis=1)
    else:
        take=take & ~variability[['unlabeled-mature','elu-mature']].isnull().any(axis=1)

    try:
        T=map(float,options.T.split(','))
        if len(T)==1:
            T=pd.DataFrame(T[0],index=TPM.index,columns=conditions)
            print >> sys.stderr, '\n[main] using single value for labeling time: {0}'.format(options.T)
        elif len(T)==len(conditions):
            print >> sys.stderr, '\n[main] using list of values for labeling time: {0}'.format(options.T)
            T=pd.DataFrame([T]*len(TPM.index),index=TPM.index,columns=conditions)
        else:
            raise Exception("cannot parse labeling time {0}".format(options.T))
    except ValueError:
        T=pd.read_csv(options.T,index_col=0,header=None,names=conditions).loc[TPM.index]
        print >> sys.stderr, '\n[main] using file with values for labeling time: {0}'.format(options.T)
        nnull=T.isnull().any(axis=1).sum()
        if nnull > 0:
            raise Exception("{0} genes have no labeling time in {1}".format(nnull,options.T))

    print >> sys.stderr, '\n[main] running RNAkira'
    results=RNAkira(counts[take], variability[take], NF[take], T[take], \
                    alpha=options.alpha, LFC_cutoff=options.LFC_cutoff,\
                    no_model_selection=options.no_model_selection, \
                    min_precursor=options.min_precursor, \
                    min_ribo=options.min_ribo,\
                    use_flowthrough=use_flowthrough,\
                    maxlevel=options.maxlevel, \
                    prior_weight=options.prior_weight, \
                    statsmodel=options.statsmodel)

    print >> sys.stderr, '[main] collecting output'
    output=collect_results(results, conditions, select_best=not options.no_model_selection)

    print >> sys.stderr, '       writing results to {0}'.format(options.out_prefix+'results.csv')
    output.to_csv(options.out_prefix+'results.csv')

