import os
import sys
import numpy as np
import pandas as pd
import scipy.stats
from collections import defaultdict,OrderedDict
import itertools 
import RNAkira
from matplotlib import pyplot as plt
from optparse import OptionParser

plt.ion()
np.random.seed(0)

# ignore warning about division by zero or over-/underflows
np.seterr(divide='ignore',over='ignore',under='ignore',invalid='ignore')

# change here if you want to test other scenarios
# these are prior estimates similar to what we observe in our data
true_priors=pd.DataFrame(dict(mu=np.array([5,-1.5,.6,0,.02,.02,.02,.02]),\
                              std=np.array([2,1,.5,.5,.01,.005,.005,.005])),\
                         index=['log_a0','log_b0','log_c0','log_d0','alpha','beta','gamma','delta'])

# distribute models over genes, make sure most genes don't change for multiple testing correction to work (other model types than MPR don't really work here)
true_gene_class=['MPR_0']*1259+\
    ['MPR_a']*250+\
    ['MPR_b']*250+\
    ['MPR_c']*250+\
    ['MPR_d']*250+\
    ['MPR_ab']*33+\
    ['MPR_ac']*33+\
    ['MPR_ad']*33+\
    ['MPR_bc']*33+\
    ['MPR_bd']*33+\
    ['MPR_cd']*33+\
    ['MPR_abc']*9+\
    ['MPR_acd']*9+\
    ['MPR_abd']*9+\
    ['MPR_bcd']*9+\
    ['MPR_abcd']*7

true_gene_class=['MPR_0']*1500+['MPR_all']*500
#true_gene_class=['MPR_0']*100

# define time points
time_points=['0','12','24','36','48']
times=map(float,time_points)
# define number of replicates
nreps=10
replicates=map(str,range(nreps))
samples=list(itertools.product(time_points,replicates))
# labeling time
T=1
# parameters of dispersion curve (set intercept=1 and slope=0 for Poisson model)
slope,intercept=0.01,2
#slope,intercept=0,1

# some flags to control behavior
use_length_library_bias=False
used_true_priors=True
do_direct_fits=True

# model definition (same as in RNAkira)

mature_pars=['log_a0','log_b0']
precursor_pars=['log_c0']
ribo_pars=['log_d0']
nlevels=6
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

nlevels=len(models)
model_pars=dict((m,v[0]) for l in range(nlevels) for m,v in models[l].iteritems())

min_args=dict(method='L-BFGS-B',jac=True,options={'disp':False, 'ftol': 1.e-15, 'gtol': 1.e-10})

parser=OptionParser()
parser.add_option('','--maxlevel',dest='maxlevel',help="max level to test [5]",default=5,type=int)
parser.add_option('','--alpha',dest='alpha',help="FDR cutoff [0.05]",default=0.05)

options,args=parser.parse_args()

sig_level=float(options.alpha)

cols=['elu-mature','flowthrough-mature','unlabeled-mature','elu-precursor','flowthrough-precursor','unlabeled-precursor','ribo']

nGenes=len(true_gene_class)
genes=np.array(map(lambda x: '_'.join(x), zip(['gene']*nGenes,map(str,range(nGenes)),true_gene_class)))

# random lengths and ucounts for genes
gene_stats=pd.DataFrame(dict(exon_length=10**scipy.stats.norm.rvs(3.0,scale=.56,size=nGenes),\
                             exon_ucount=10**scipy.stats.norm.rvs(2.4,scale=.52,size=nGenes),
                             gene_type=['protein_coding']*nGenes),index=genes)

# this is used to calculate size factors for elu + flowthrough
sf=np.exp(-np.exp(true_priors.ix['log_b0','mu'])*T)

true_gene_class=pd.Series(true_gene_class,index=genes)

parameters_known=True
parameters={}
counts={}
disp={}
stddev={}

print >> sys.stderr, '\n[test_RNAkira] drawing parameters and observations for {0} genes ({1} time points, {2} replicates)'.format(nGenes,len(time_points),nreps)
print >> sys.stderr, '[test_RNAkira] true priors for log_a: {0:.2g}/{1:.2g}, log_b: {2:.2g}/{3:.2g}, log_c: {4:.2g}/{5:.2g}, log_d: {6:.2g}/{7:.2g}'.format(*true_priors.ix[:4].values.flatten())

for ng,gene in enumerate(genes):

    print >> sys.stderr, '[test_RNAkira] {0} genes initialized\r'.format(ng+1),

    lf=1.-np.exp(-gene_stats.ix[gene,'exon_length']/1.)

    model=true_gene_class[gene]
    if model=='MPR_all':
        rates=[scipy.stats.norm.rvs(scipy.stats.norm.rvs(true_priors.ix[v,'mu'],true_priors.ix[v,'std']),\
                                    .5*true_priors.ix[v,'std'],size=len(time_points)) for v in model_pars[model]]
        pars=pd.DataFrame(rates,columns=time_points,index=model_pars[model]).T
    else:
        pars=pd.Series(np.array([scipy.stats.norm.rvs(true_priors.ix[v,'mu'],true_priors.ix[v,'std']) for v in model_pars[model]]),index=model_pars[model])
        # add random up- or down-movement and subtract linear trend from baseline
        if 'alpha' in pars:
            pars['alpha']=np.random.choice([+1,-1])*pars['alpha']
            pars['log_a0']-=pars['alpha']*np.mean(times)
        if 'beta' in pars:
            pars['beta']=np.random.choice([+1,-1])*pars['beta']
            pars['log_b0']-=pars['beta']*np.mean(times)
        if 'gamma' in pars:
            pars['gamma']=np.random.choice([+1,-1])*pars['gamma']
            pars['log_c0']-=pars['gamma']*np.mean(times)
        if 'delta' in pars:
            pars['delta']=np.random.choice([+1,-1])*pars['delta']
            pars['log_d0']-=pars['delta']*np.mean(times)
        rates=RNAkira.get_rates(time_points,pars)

    # now get random values for observations according to these rate parameters
    cnts=pd.Series(index=pd.MultiIndex.from_product([cols,time_points,replicates]))
    std=pd.Series(index=pd.MultiIndex.from_product([cols,time_points]))
    dsp=pd.Series(index=pd.MultiIndex.from_product([cols,time_points]))

    for i,t in enumerate(time_points):

        mu=RNAkira.get_steady_state_values([r[i] for r in rates],T,use_precursor='P' in model, use_ribo='R' in model)
        # multiply by gene length, introduce length-dependent 4sU incorporation bias and library size effect
        if use_length_library_bias:
            mu_eff=mu*gene_stats.ix[gene,'exon_length']/1.e3/np.array([1.-sf,sf,1.,1.-sf,sf,1.,1.])/np.array([lf,1,1,1,1,1,1])
        else:
            mu_eff=mu

        for n in range(len(mu)):
            d=(intercept-1)/mu_eff[n]+slope
            if d < 1.e-8:
                cnts[cols[n],t]=scipy.stats.poisson.rvs(mu_eff[n],size=nreps)
            else:
                cnts[cols[n],t]=scipy.stats.nbinom.rvs(1./d,1./(1.+d*mu_eff[n]),size=nreps)
            std[cols[n],t]=np.sqrt(mu_eff[n]*(1.+d*mu_eff[n]))
            dsp[cols[n],t]=d

    parameters[gene]=pars
    counts[gene]=cnts
    stddev[gene]=std
    disp[gene]=dsp

    if do_direct_fits:

        vals_here=cnts.unstack(level=0)[cols].stack().values.reshape((len(time_points),nreps,len(cols)))
        std_here=std.unstack(level=0)[cols].stack().values.reshape((len(time_points),len(cols)))
        disp_here=dsp.unstack(level=0)[cols].stack().values.reshape((len(time_points),len(cols)))
        nf_here=np.ones_like(vals_here)

        resg={}
        resg['all']=RNAkira.fit_model(vals_here,std_here,nf_here,T,time_points,true_priors,None,'MPR_all',model_pars['MPR_all'],'gaussian',min_args)
        resg['0']=RNAkira.fit_model(vals_here,std_here,nf_here,T,time_points,true_priors,resg['all'],'MPR_0',model_pars['MPR_0'],'gaussian',min_args)
        resg['a']=RNAkira.fit_model(vals_here,std_here,nf_here,T,time_points,true_priors,resg['0'],'MPR_a',model_pars['MPR_a'],'gaussian',min_args)
        resg['ab']=RNAkira.fit_model(vals_here,std_here,nf_here,T,time_points,true_priors,resg['a'],'MPR_ab',model_pars['MPR_ab'],'gaussian',min_args)
        resg['abc']=RNAkira.fit_model(vals_here,std_here,nf_here,T,time_points,true_priors,resg['ab'],'MPR_abc',model_pars['MPR_abc'],'gaussian',min_args)
        resg['abcd']=RNAkira.fit_model(vals_here,std_here,nf_here,T,time_points,true_priors,resg['abc'],'MPR_abcd',model_pars['MPR_abcd'],'gaussian',min_args)

        resn={}
        resn['all']=RNAkira.fit_model(vals_here,disp_here,nf_here,T,time_points,true_priors,None,'MPR_all',model_pars['MPR_all'],'nbinom',min_args)
        resn['0']=RNAkira.fit_model(vals_here,disp_here,nf_here,T,time_points,true_priors,resn['all'],'MPR_0',model_pars['MPR_0'],'nbinom',min_args)
        resn['a']=RNAkira.fit_model(vals_here,disp_here,nf_here,T,time_points,true_priors,resn['0'],'MPR_a',model_pars['MPR_a'],'nbinom',min_args)
        resn['ab']=RNAkira.fit_model(vals_here,disp_here,nf_here,T,time_points,true_priors,resn['a'],'MPR_ab',model_pars['MPR_ab'],'nbinom',min_args)
        resn['abc']=RNAkira.fit_model(vals_here,disp_here,nf_here,T,time_points,true_priors,resn['ab'],'MPR_abc',model_pars['MPR_abc'],'nbinom',min_args)
        resn['abcd']=RNAkira.fit_model(vals_here,disp_here,nf_here,T,time_points,true_priors,resn['abc'],'MPR_abcd',model_pars['MPR_abcd'],'nbinom',min_args)

        raise Exception('stop')

print >> sys.stderr, ''

counts=pd.DataFrame.from_dict(counts,orient='index').loc[genes]
disp=pd.DataFrame.from_dict(disp,orient='index').loc[genes]
stddev=pd.DataFrame.from_dict(stddev,orient='index').loc[genes]

if use_length_library_bias:
    LF=gene_stats['exon_length']/1.e3
    # normalize by "sequencing depth"
    elu_factor=(counts['elu-mature'].add(counts['elu-precursor'],fill_value=0).sum(axis=0))/1.e6
    flowthrough_factor=(counts['flowthrough-mature'].add(counts['flowthrough-precursor'],fill_value=0).sum(axis=0))/1.e6
    unlabeled_factor=counts['unlabeled-mature'].add(counts['unlabeled-precursor'],fill_value=0).sum(axis=0)/1.e6
    ribo_factor=counts['ribo'].sum(axis=0)/1.e6
    SF=pd.concat([elu_factor,flowthrough_factor,unlabeled_factor,\
                  elu_factor,flowthrough_factor,unlabeled_factor,\
                  ribo_factor],axis=0,keys=cols)
    CF=RNAkira.normalize_elu_flowthrough(counts.divide(LF,axis=0).divide(SF,axis=1).fillna(1),samples,gene_stats,fig_name='test_TPM_correction.pdf')
    NF=CF.divide(LF,axis=0).divide(SF,axis=1).fillna(1)
else:
    LF=pd.Series(1,index=LF.index)
    SF=pd.Series(1,index=SF.index)
    NF=pd.DataFrame(1,index=counts.index,columns=counts.columns)

TPM=counts.multiply(NF)
if use_length_library_bias:
    stddev=RNAkira.estimate_stddev (TPM,fig_name='test_variability_stddev.pdf')
    disp=RNAkira.estimate_dispersion (counts.divide(SF.divide(np.exp(np.log(SF).mean(level=0)),level=0),axis=1),fig_name='test_variability_disp.pdf')

results_gaussian=RNAkira.RNAkira(counts, stddev, NF, T, sig_level=sig_level, min_ribo=.1, min_precursor=.1, maxlevel=options.maxlevel, statsmodel='gaussian', priors=true_priors if use_true_priors else None)
results_nbinom=RNAkira.RNAkira(counts, disp, NF, T, sig_level=sig_level, min_ribo=.1, min_precursor=.1, maxlevel=options.maxlevel, statsmodel='nbinom', priors=true_priors if use_true_priors else None)

output_gaussian=RNAkira.collect_results(results_gaussian, time_points, sig_level=sig_level)
output_nbinom=RNAkira.collect_results(results_nbinom, time_points, sig_level=sig_level)

output_true=pd.DataFrame([pd.DataFrame(RNAkira.get_rates(time_points,parameters[gene]),columns=time_points,index=['initial_synthesis','initial_degradation','initial_processing','initial_translation']).stack() for gene in genes],index=genes).loc[output_gaussian.index]
output_true['initial_synthesis']-=np.log(SF['unlabeled-mature'].mean())
output_true.columns=[c[0]+'_t'+c[1] for c in output_true.columns.tolist()]

#counts.to_csv('test_counts.csv')
#output_gaussian.to_csv('test_gaussian_output.csv')
#output_nbinom.to_csv('test_nbinom_output.csv')
#output_true.to_csv('test_true_output.csv')

print >> sys.stderr, '\n[test_RNAkira] evaluating performance'

#for output,results,statsmodel in zip([output_gaussian,output_nbinom],[results_gaussian,results_nbinom],['gaussian','nbinom']):
for output,statsmodel in zip([output_gaussian,output_nbinom],['gaussian','nbinom']):

    inferred_gene_class=output.ix[genes,'best_model']
    inferred_gene_class[output.ix[genes,'initial_qval'] < options.alpha]=output.ix[genes,'initial_model'][output.ix[genes,'initial_qval'] < options.alpha]

    # use this if you want to plot specific examples
    if False:

        genes_to_plot=genes[np.where(inferred_gene_class==true_gene_class)[0]]
        np.random.shuffle(genes_to_plot)

        for k,gene in enumerate(genes_to_plot[:min(5,len(genes_to_plot))]):
            pcorr=pd.Series(dict(log_a0=np.log(SF['unlabeled-mature'].mean()*LF.mean()),log_b0=0,log_c0=0,log_d0=np.log(SF['ribo'].mean()*LF.mean())))
            RNAkira.plot_data_rates_fits(time_points,replicates,TPM.ix[gene],T,\
                                         parameters[gene]-pcorr if parameters_known else None,\
                                         results[gene],\
                                         'P' in true_gene_class[gene],\
                                         'R' in true_gene_class[gene],\
                                         title='{0} (true: {1}, inferred: {2})'.format(gene,true_gene_class[gene],inferred_gene_class[gene]),\
                                         priors=None,sig_level=options.alpha)

    tgc=np.array([m.split('_')[1] for m in true_gene_class])
    igc=np.array([m.split('_')[1] for m in inferred_gene_class])
    mods=[s for lev in [1,2,3,4,5,0] for s in set([m.split('_')[1] for m in models[lev]])]
    matches=np.array([[np.sum((tgc==m1) & (igc==m2)) for m2 in mods] for m1 in mods])

    nexact=np.sum(np.diag(matches))
    nover=np.sum(np.triu(matches,1))
    nlim=sum(tgc!='all')
    nunder=np.sum(np.tril(matches,-1))
    ntarget=sum(tgc!='0')

    stats='{0} exact hits ({1:.1f}%)\n{2} over-classifications ({3:.1f}%)\n{4} under-classifications ({5:.1f}%)'.format(nexact,100*nexact/float(nGenes),nover,100*nover/float(nlim),nunder,100*nunder/float(ntarget))
    title='{0} genes, {1} time points, {2} replicates, {3} model\n{4}'.format(nGenes,len(time_points),nreps,statsmodel,stats)
    print >> sys.stderr, stats

    fig=plt.figure(figsize=(7,7))
    fig.clf()

    ax=fig.add_axes([.1,.1,.85,.8])
    ax.imshow(np.log2(1+matches),origin='lower',cmap=plt.cm.Blues,vmin=0,vmax=np.log2(nGenes))
    ax.set_xticks(range(len(mods)))
    ax.set_xticklabels(mods,rotation=90,va='top',ha='center')
    ax.set_xlabel('inferred model')
    ax.set_ylabel('true model')
    ax.set_yticks(range(len(mods)))
    ax.set_yticklabels(mods)
    for i in range(len(mods)):
        for j in range(len(mods)):
            if matches[i,j] > 0:
                ax.text(j,i,matches[i,j],size=8,ha='center',va='center',color='k' if i==j else 'r')

    ax.set_title(title,size=10)

    #fig.savefig('test_confusion_matrix_{0}.pdf'.format(statsmodel))

    diff=np.abs(np.log(output.ix[:,:len(time_points)*4])-output_true)

    fig=plt.figure()
    fig.clf()
    fig.subplots_adjust(hspace=.4,wspace=.4)

    for n,r in enumerate(['synthesis','degradation','processing','translation']):
        ax=fig.add_subplot(2,2,n+1)
        y,x=np.log(output_nbinom.ix[:,5*n:5*(n+1)]).values.flatten(),output_true.ix[:,5*n:5*(n+1)].values.flatten()
        ok=np.isfinite(x) & np.isfinite(y)
        xr=np.percentile(x[ok],[1,99])
        yr=np.percentile(y[ok],[1,99])
        ax.hexbin(x[ok],y[ok],extent=(xr[0],xr[1],yr[0],yr[1]),bins='log',mincnt=1,vmin=-1)
        ax.plot(np.linspace(xr[0],xr[1],100),np.linspace(xr[0],xr[1],100),'r-',lw=.5)
        ax.set_title('{0}: r={1:.2f}'.format(r,scipy.stats.spearmanr(x,y)[0]),size=10)
        if n > 1:
            ax.set_xlabel('log true value')
        if n%2==0:
            ax.set_ylabel('log fitted value'.format(statsmodel))

    fig.suptitle('{0} genes, {1} time points, {2} replicates, {3} model'.format(nGenes,len(time_points),nreps,statsmodel),size=10)
