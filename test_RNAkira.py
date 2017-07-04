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

parser=OptionParser()
parser.add_option('','--maxlevel',dest='maxlevel',help="max level to test [4]",default=4,type=int)
parser.add_option('','--alpha',dest='alpha',help="FDR cutoff [0.05]",default=0.05)

options,args=parser.parse_args()
sig_level=float(options.alpha)

########################################################################
#### set parameters here                                            ####
########################################################################

# these are prior estimates on rates a,b,c,d similar to what we observe in our data
true_priors=pd.DataFrame(dict(mu=np.array([5,-1.5,.6,0]),\
                              std=np.array([2,1,.5,.5,])),\
                         index=list("abcd"))

# distribute models over genes, make sure most genes don't change for multiple testing correction to work
true_gene_class=['abcd']*759+\
    ['Abcd']*250+\
    ['aBcd']*250+\
    ['abCd']*250+\
    ['abcD']*250+\
    ['ABcd']*33+\
    ['AbCd']*33+\
    ['AbcD']*33+\
    ['aBCd']*33+\
    ['aBcD']*33+\
    ['abCD']*33+\
    ['ABCd']*9+\
    ['ABcD']*9+\
    ['AbCD']*9+\
    ['aBCD']*9+\
    ['ABCD']*7

#true_gene_class=['abcd']*150+['ABCD']*50
#true_gene_class=['ABCD']*1

# define time points
time_points=['0','12','24','36','48']
ntimes=len(time_points)
# define number of replicates
nreps=10
replicates=map(str,range(nreps))
samples=list(itertools.product(time_points,replicates))
# labeling time
T=1
# parameters of dispersion curve (set intercept=1 and slope=0 for Poisson model, otherwise neg binomial)
slope,intercept=0.01,2
#slope,intercept=0,1

# some flags to control behavior
use_length_library_bias=False
use_true_priors=False
do_direct_fits=False

# model definition (same as in RNAkira)
nlevels=5
models=RNAkira.get_model_definitions(nlevels)

min_args=dict(method='L-BFGS-B',jac=True,options={'disp':False, 'ftol': 1.e-15, 'gtol': 1.e-10})

cols=['elu-mature','flowthrough-mature','unlabeled-mature','elu-precursor','flowthrough-precursor','unlabeled-precursor','ribo']

########################################################################
#### simulate data                                                  ####
########################################################################

nGenes=len(true_gene_class)
genes=np.array(map(lambda x: '_'.join(x), zip(['gene']*nGenes,map(str,range(nGenes)),true_gene_class)))

# random lengths and ucounts for genes
gene_stats=pd.DataFrame(dict(exon_length=10**scipy.stats.norm.rvs(3.0,scale=.56,size=nGenes),\
                             exon_ucount=10**scipy.stats.norm.rvs(2.4,scale=.52,size=nGenes),
                             gene_type=['protein_coding']*nGenes),index=genes)

# this is used to calculate size factors for elu + flowthrough
sf=np.exp(-np.exp(true_priors.ix['b','mu'])*T)

true_gene_class=pd.Series(true_gene_class,index=genes)

parameters={}
counts={}
disp={}
stddev={}

print >> sys.stderr, '\n[test_RNAkira] drawing parameters and observations for {0} genes ({1} time points, {2} replicates)'.format(nGenes,ntimes,nreps)
print >> sys.stderr, '[test_RNAkira] true priors for log_a: {0:.2g}/{1:.2g}, log_b: {2:.2g}/{3:.2g}, log_c: {4:.2g}/{5:.2g}, log_d: {6:.2g}/{7:.2g}'.format(*true_priors.ix[:4].values.flatten())

for ng,gene in enumerate(genes):

    print >> sys.stderr, '[test_RNAkira] {0} genes initialized\r'.format(ng+1),

    lf=1.-np.exp(-gene_stats.ix[gene,'exon_length']/1.)

    model=true_gene_class[gene]
    # first draw constant baselines for each parameter
    pars=[scipy.stats.norm.rvs(true_priors.ix[mp,'mu'],true_priors.ix[mp,'std']) for mp in model.lower()]
    # then expand this over time (add randomness of ~2fold per time point for variable parameters)
    rates=np.array([p*np.ones(ntimes) if mp.islower() else \
                    scipy.stats.norm.rvs(p,np.log(2),size=ntimes) for mp,p in zip(model,pars)])

    # now get random values for observations according to these rate parameters
    cnts=pd.Series(index=pd.MultiIndex.from_product([cols,time_points,replicates]))
    std=pd.Series(index=pd.MultiIndex.from_product([cols,time_points]))
    dsp=pd.Series(index=pd.MultiIndex.from_product([cols,time_points]))

    for i,t in enumerate(time_points):

        mu=RNAkira.get_steady_state_values(rates[:,i],T,model)
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

    parameters[gene]=rates
    counts[gene]=cnts
    stddev[gene]=std
    disp[gene]=dsp

    if do_direct_fits:

        vals_here=cnts.unstack(level=0)[cols].stack().values.reshape((len(time_points),nreps,len(cols)))
        std_here=std.unstack(level=0)[cols].stack().values.reshape((len(time_points),len(cols)))
        disp_here=dsp.unstack(level=0)[cols].stack().values.reshape((len(time_points),len(cols)))
        nf_here=np.ones_like(vals_here)

        resg={}
        resg['ABCD']=RNAkira.fit_model(vals_here,std_here,nf_here,T,time_points,true_priors,None,'ABCD','gaussian',min_args)
        resg['abcd']=RNAkira.fit_model(vals_here,std_here,nf_here,T,time_points,true_priors,resg['ABCD'],'abcd','gaussian',min_args)
        resg['Abcd']=RNAkira.fit_model(vals_here,std_here,nf_here,T,time_points,true_priors,resg['abcd'],'Abcd','gaussian',min_args)
        resg['ABcd']=RNAkira.fit_model(vals_here,std_here,nf_here,T,time_points,true_priors,resg['Abcd'],'ABcd','gaussian',min_args)
        resg['ABCd']=RNAkira.fit_model(vals_here,std_here,nf_here,T,time_points,true_priors,resg['ABcd'],'ABCd','gaussian',min_args)

        resn={}
        resn['ABCD']=RNAkira.fit_model(vals_here,std_here,nf_here,T,time_points,true_priors,None,'ABCD','nbinom',min_args)
        resn['abcd']=RNAkira.fit_model(vals_here,std_here,nf_here,T,time_points,true_priors,resn['ABCD'],'abcd','nbinom',min_args)
        resn['Abcd']=RNAkira.fit_model(vals_here,std_here,nf_here,T,time_points,true_priors,resn['abcd'],'Abcd','nbinom',min_args)
        resn['ABcd']=RNAkira.fit_model(vals_here,std_here,nf_here,T,time_points,true_priors,resn['Abcd'],'ABcd','nbinom',min_args)
        resn['ABCd']=RNAkira.fit_model(vals_here,std_here,nf_here,T,time_points,true_priors,resn['ABcd'],'ABCd','nbinom',min_args)

        raise Exception('stop')

print >> sys.stderr, '\n'

########################################################################
#### normalization, 4sU bias correction                             ####
########################################################################

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
    CF=RNAkira.normalize_elu_flowthrough(counts.divide(LF,axis=0).divide(SF,axis=1).fillna(1),samples,gene_stats)#,fig_name='test_TPM_correction.pdf')
    NF=CF.divide(LF,axis=0).divide(SF,axis=1).fillna(1)
else:
    LF=pd.Series(1,index=gene_stats.index)
    SF=pd.Series(1,index=cols)
    NF=pd.DataFrame(1,index=counts.index,columns=counts.columns)

TPM=counts.multiply(NF)
if use_length_library_bias:
    stddev=RNAkira.estimate_stddev (TPM)#,fig_name='test_variability_stddev.pdf')
    disp=RNAkira.estimate_dispersion (counts.divide(SF.divide(np.exp(np.log(SF).mean(level=0)),level=0),axis=1))#,fig_name='test_variability_disp.pdf')

########################################################################
#### RNAkira results                                                ####
########################################################################

results_gaussian=RNAkira.RNAkira(counts, stddev, NF, T, sig_level=sig_level, min_ribo=1, min_precursor=1, \
                                 maxlevel=options.maxlevel, statsmodel='gaussian', priors=true_priors if use_true_priors else None)
results_nbinom=RNAkira.RNAkira(counts, disp, NF, T, sig_level=sig_level, min_ribo=1, min_precursor=1, \
                               maxlevel=options.maxlevel, statsmodel='nbinom', priors=true_priors if use_true_priors else None)

output_gaussian=RNAkira.collect_results(results_gaussian, time_points, sig_level=sig_level)
output_nbinom=RNAkira.collect_results(results_nbinom, time_points, sig_level=sig_level)

output_true=pd.DataFrame([pd.DataFrame(parameters[gene],columns=time_points,\
                                       index=['initial_synthesis','initial_degradation','initial_processing','initial_translation']).stack() for gene in genes],index=genes).loc[output_gaussian.index]
if use_length_library_bias:
    output_true['initial_synthesis']-=np.log(SF['unlabeled-mature'].mean())
output_true.columns=[c[0]+'_t'+c[1] for c in output_true.columns.tolist()]

#counts.to_csv('test_counts.csv')
#output_gaussian.to_csv('test_gaussian_output.csv')
#output_nbinom.to_csv('test_nbinom_output.csv')
#output_true.to_csv('test_true_output.csv')

########################################################################
#### evaluate performance                                           ####
########################################################################

for output,statsmodel in zip([output_gaussian,output_nbinom],['gaussian','nbinom']):

    inferred_gene_class=output.ix[genes,'best_model']
    inferred_gene_class[output.ix[genes,'initial_qval'] < options.alpha]=output.ix[genes,'initial_model'][output.ix[genes,'initial_qval'] < options.alpha]

    # use this if you want to plot specific examples (plot_data_rates_fits needs fixing!)
    if False:

        genes_to_plot=genes[np.where(inferred_gene_class==true_gene_class)[0]]
        np.random.shuffle(genes_to_plot)

        for k,gene in enumerate(genes_to_plot[:min(5,len(genes_to_plot))]):
            pcorr=pd.Series(dict(log_a0=np.log(SF['unlabeled-mature'].mean()*LF.mean()),log_b0=0,log_c0=0,log_d0=np.log(SF['ribo'].mean()*LF.mean())))
            RNAkira.plot_data_rates_fits(time_points,replicates,TPM.ix[gene],T,\
                                         parameters[gene]-pcorr,\
                                         results[gene],\
                                         'P' in true_gene_class[gene],\
                                         'R' in true_gene_class[gene],\
                                         title='{0} (true: {1}, inferred: {2})'.format(gene,true_gene_class[gene],inferred_gene_class[gene]),\
                                         priors=None,sig_level=options.alpha)

    tgc=true_gene_class
    igc=inferred_gene_class
    mods=[s for lev in [1,2,3,4,0] for s in set(models[lev]) if len(s)==4]
    matches=np.array([[np.sum((tgc==m1) & (igc==m2)) for m2 in mods] for m1 in mods])

    nexact=np.sum(np.diag(matches))
    nover=np.sum(np.triu(matches,1))
    nlim=sum(tgc!='all')
    nunder=np.sum(np.tril(matches,-1))
    ntarget=sum(tgc!='0')

    stats='{0} exact hits ({1:.1f}%)\n{2} over-classifications ({3:.1f}%)\n{4} under-classifications ({5:.1f}%)'.format(nexact,100*nexact/float(nGenes),nover,100*nover/float(nlim),nunder,100*nunder/float(ntarget))
    title='{0} genes, {1} time points, {2} replicates, {3} model\n{4}'.format(nGenes,len(time_points),nreps,statsmodel,stats)
    print >> sys.stderr, stats

    fig=plt.figure(figsize=(5,5.5))
    fig.clf()

    ax=fig.add_axes([.15,.1,.8,.8])
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
        y,x=np.log(output.ix[:,5*n:5*(n+1)]).values.flatten(),output_true.ix[:,5*n:5*(n+1)].values.flatten()
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

    initial_R2=output[['initial_R2_RNA','initial_R2_ribo']]
    modeled_R2=output[['modeled_R2_RNA','modeled_R2_ribo']]

    fig=plt.figure(figsize=(6,3))
    fig.clf()
    fig.subplots_adjust(hspace=.4,wspace=.4,bottom=.15)

    ax=fig.add_subplot(1,2,1)
    ax.hist(initial_R2.dropna().values,bins=np.arange(-.2,1.2,.01),histtype='step',label=['RNA','RPF'])
    ax.set_ylabel('counts')
    ax.set_xlabel('initial R2')

    ax=fig.add_subplot(1,2,2)
    ax.hist(modeled_R2.dropna().values/initial_R2.dropna().values,bins=np.arange(-.2,1.2,.01),histtype='step',label=['RNA','RPF'])
    ax.set_ylabel('counts')
    ax.set_xlabel('modeled eff. R2')
    
