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
parser.add_option('','--alpha',dest='alpha',help="model selection cutoff [0.05]",default=0.05,type=float)
parser.add_option('','--criterion',dest='criterion',help="model selection criterion [LRT]",default='LRT')
parser.add_option('','--use_length_library_bias',dest='use_length_library_bias',action='store_true',default=False)
parser.add_option('','--estimate_variability',dest='estimate_variability',action='store_true',default=False)
parser.add_option('','--use_true_priors',dest='use_true_priors',action='store_true',default=False)
parser.add_option('','--do_direct_fits',dest='do_direct_fits',action='store_true',default=False)
parser.add_option('','--statsmodel',dest='statsmodel',default='nbinom')
parser.add_option('','--save_figures',dest='save_figures',action='store_true',default=False)
parser.add_option('','--out_prefix',dest='out_prefix',default='test')

options,args=parser.parse_args()

########################################################################
#### set parameters here                                            ####
########################################################################

# these are prior estimates on rates a,b,c,d similar to what we observe in our data
true_priors=pd.DataFrame(dict(mu=np.array([6,-1.5,.6,-1]),\
                              std=np.array([2,1,.5,.5])),\
                         index=list("abcd"))

# distribute models over genes
true_gene_class=['abcd']*4559+\
    ['Abcd']*50+\
    ['aBcd']*50+\
    ['abCd']*50+\
    ['abcD']*50+\
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
#true_gene_class=['abcd']*1000
#true_gene_class=['abcd']*500+['Abcd']*125+['ABcd']*125+['ABCd']*125+['ABCD']*125

# define models to test (otherwise RNAkira will test all combinations)
models=OrderedDict([(0,OrderedDict([('ABCD',[])])),\
                    (1,OrderedDict([('abcd',['ABCD'])])),\
                    (2,OrderedDict([('Abcd',['abcd'])])),\
                    (3,OrderedDict([('ABcd',['Abcd'])])),\
                    (4,OrderedDict([('ABCd',['ABcd'])]))])

# define time points
time_points=['0','20','40','60','80','100']
ntimes=len(time_points)
# define number of replicates
nreps=5
replicates=map(str,range(nreps))
samples=list(itertools.product(time_points,replicates))
# labeling time
T=1
# parameters of dispersion curve (set intercept=1 and slope=0 for Poisson model, otherwise neg binomial)
slope,intercept=0.01,2
#slope,intercept=0,1
# average fold change between time points
AVE_FC=2

min_args=dict(method='L-BFGS-B',jac=True,options={'disp':False, 'ftol': 1.e-15, 'gtol': 1.e-10})

cols=['elu-mature','flowthrough-mature','unlabeled-mature','elu-precursor','flowthrough-precursor','unlabeled-precursor','ribo']

########################################################################
#### simulate data                                                  ####
########################################################################

nGenes=len(true_gene_class)
genes=np.array(map(lambda x: '_'.join(x), zip(['gene']*nGenes,map(str,range(nGenes)),true_gene_class)))
T=pd.Series(1,index=genes)

# random lengths and ucounts for genes
gene_stats=pd.DataFrame(dict(exon_length=(10**scipy.stats.norm.rvs(3.0,scale=.56,size=nGenes)).astype(int),\
                             gene_type=['protein_coding']*nGenes),index=genes)
gene_stats['exon_ucount']=1+(.25*gene_stats['exon_length']).astype(int)

true_gene_class=pd.Series(true_gene_class,index=genes)
if options.criterion=='empirical':
    constant_genes=true_gene_class.index[(true_gene_class=='abcd')][:1000]
else:
    constant_genes=None

parameters={}

print >> sys.stderr, '\n[test_RNAkira] drawing parameters for {0} genes ({1} time points, {2} replicates)'.format(nGenes,ntimes,nreps)
print >> sys.stderr, '[test_RNAkira] true priors for log_a: {0:.2g}/{1:.2g}, log_b: {2:.2g}/{3:.2g}, log_c: {4:.2g}/{5:.2g}, log_d: {6:.2g}/{7:.2g}'.format(*true_priors.ix[:4].values.flatten())

for ng,gene in enumerate(genes):

    model=true_gene_class[gene]
    # first draw constant baselines for each parameter
    pars=[scipy.stats.norm.rvs(true_priors.ix[mp,'mu'],true_priors.ix[mp,'std']) for mp in model.lower()]
    # then expand this over time (add randomness of AVE_FC per time point for variable parameters)
    rates=np.array([p*np.ones(ntimes) if mp.islower() else \
                    scipy.stats.norm.rvs(p,np.log(AVE_FC),size=ntimes) for mp,p in zip(model,pars)])

    parameters[gene]=rates

counts={}
disp={}
stddev={}

# this is used to calculate size factors for elu + flowthrough
UF=np.mean([np.exp(parameters[gene].mean(1)[0]-parameters[gene].mean(1)[1]) for gene in genes])
EF=np.mean([np.exp(parameters[gene].mean(1)[0]-parameters[gene].mean(1)[1])*\
            (1-np.exp(-np.exp(parameters[gene].mean(1)[1])*T)) for gene in genes])
FF=np.mean([np.exp(parameters[gene].mean(1)[0]-parameters[gene].mean(1)[1])*\
            (np.exp(-np.exp(parameters[gene].mean(1)[1])*T)) for gene in genes])
size_factor=np.array([UF/EF,UF/FF,1,UF/EF,UF/FF,1,1])

for ng,gene in enumerate(genes):

    print >> sys.stderr, '[test_RNAkira] {0} genes initialized\r'.format(ng+1),

    # now get random values for observations according to these rate parameters
    cnts=pd.Series(index=pd.MultiIndex.from_product([cols,time_points,replicates]))
    std=pd.Series(index=pd.MultiIndex.from_product([cols,time_points]))
    dsp=pd.Series(index=pd.MultiIndex.from_product([cols,time_points]))

    # this is used to model U-pulldown bias
    ubias=1.-.5*np.exp(-gene_stats.ix[gene,'exon_ucount']/500.)

    for i,t in enumerate(time_points):

        # get expected values given these rates
        mu=RNAkira.get_steady_state_values(parameters[gene][:,i],T[gene],model)
        if options.use_length_library_bias:
            # multiply by gene length and library size factors and introduce U-bias 
            mu_eff=mu*(gene_stats.ix[gene,'exon_length']/1.e3)*size_factor
            mu_eff[0]=mu_eff[0]*ubias
        else:
            mu_eff=mu

        # get counts based on these expected values
        for n in range(len(mu)):
            d=(intercept-1)/mu_eff[n]+slope
            if d < 1.e-8:
                cnts[cols[n],t]=scipy.stats.poisson.rvs(mu_eff[n],size=nreps)
            else:
                cnts[cols[n],t]=scipy.stats.nbinom.rvs(1./d,1./(1.+d*mu_eff[n]),size=nreps)
            std[cols[n],t]=np.sqrt(mu_eff[n]*(1.+d*mu_eff[n]))
            dsp[cols[n],t]=d

    counts[gene]=cnts
    stddev[gene]=std
    disp[gene]=dsp

    if options.do_direct_fits:

        vals_here=cnts.unstack(level=0)[cols].stack().values.reshape((len(time_points),nreps,len(cols)))
        std_here=std.unstack(level=0)[cols].stack().values.reshape((len(time_points),len(cols)))
        disp_here=dsp.unstack(level=0)[cols].stack().values.reshape((len(time_points),len(cols)))
        nf_here=np.ones_like(vals_here)

        resg={}
        resg['ABCD']=RNAkira.fit_model(vals_here,std_here,nf_here,T[gene],time_points,true_priors,None,'ABCD','gaussian',min_args)
        resg['abcd']=RNAkira.fit_model(vals_here,std_here,nf_here,T[gene],time_points,true_priors,resg['ABCD'],'abcd','gaussian',min_args)
        resg['Abcd']=RNAkira.fit_model(vals_here,std_here,nf_here,T[gene],time_points,true_priors,resg['abcd'],'Abcd','gaussian',min_args)
        resg['ABcd']=RNAkira.fit_model(vals_here,std_here,nf_here,T[gene],time_points,true_priors,resg['Abcd'],'ABcd','gaussian',min_args)
        resg['ABCd']=RNAkira.fit_model(vals_here,std_here,nf_here,T[gene],time_points,true_priors,resg['ABcd'],'ABCd','gaussian',min_args)

        resn={}
        resn['ABCD']=RNAkira.fit_model(vals_here,std_here,nf_here,T[gene],time_points,true_priors,None,'ABCD','nbinom',min_args)
        resn['abcd']=RNAkira.fit_model(vals_here,std_here,nf_here,T[gene],time_points,true_priors,resn['ABCD'],'abcd','nbinom',min_args)
        resn['Abcd']=RNAkira.fit_model(vals_here,std_here,nf_here,T[gene],time_points,true_priors,resn['abcd'],'Abcd','nbinom',min_args)
        resn['ABcd']=RNAkira.fit_model(vals_here,std_here,nf_here,T[gene],time_points,true_priors,resn['Abcd'],'ABcd','nbinom',min_args)
        resn['ABCd']=RNAkira.fit_model(vals_here,std_here,nf_here,T[gene],time_points,true_priors,resn['ABcd'],'ABCd','nbinom',min_args)

        raise Exception('stop')

print >> sys.stderr, '\n'

counts=pd.DataFrame.from_dict(counts,orient='index').loc[genes]
disp=pd.DataFrame.from_dict(disp,orient='index').loc[genes]
stddev=pd.DataFrame.from_dict(stddev,orient='index').loc[genes]

if options.criterion=='empirical':
    counts.ix[constant_genes,'ribo']=0
    counts.ix[constant_genes,'unlabeled-precursor']=0
    counts.ix[constant_genes,'elu-precursor']=0
    counts.ix[constant_genes,'flowthrough-precursor']=0

########################################################################
#### normalization, U-bias correction                               ####
########################################################################

if options.use_length_library_bias:
    LF=gene_stats['exon_length']/1.e3
    # normalize by "sequencing depth"
    RPK=counts.divide(LF,axis=0)
    EF=(RPK['elu-mature'].add(RPK['elu-precursor'],fill_value=0).sum(axis=0))/1.e6
    FF=(RPK['flowthrough-mature'].add(RPK['flowthrough-precursor'],fill_value=0).sum(axis=0))/1.e6
    UF=RPK['unlabeled-mature'].add(RPK['unlabeled-precursor'],fill_value=0).sum(axis=0)/1.e6
    RF=RPK['ribo'].sum(axis=0)/1.e6
    SF=pd.concat([EF,FF,UF,\
                  EF,FF,UF,\
                  RF],axis=0,keys=cols)
    TPM=RPK.divide(SF,axis=1).fillna(1)
    UF=RNAkira.correct_ubias(TPM,gene_stats,fig_name=options.out_prefix+'_ubias_correction.pdf' if options.save_figures else None)
    CF=RNAkira.normalize_elu_flowthrough_over_genes(TPM.multiply(UF),samples,fig_name=options.out_prefix+'_TPM_correction_1.pdf' if options.save_figures else None)

else:
    print >> sys.stderr, '[test_RNAkira] run without library size normalization and U-bias correction\n'
    LF=pd.Series(1,index=gene_stats.index)
    SF=pd.Series(1,index=pd.MultiIndex.from_product([cols,time_points,replicates]))
    UF=pd.DataFrame(1,index=counts.index,columns=counts.columns)
    CF=pd.Series(1,index=counts.columns)

NF=UF.multiply(CF).divide(LF,axis=0).divide(SF,axis=1).fillna(1)
TPM=counts.multiply(NF)

if options.estimate_variability:
    if options.statsmodel=='gaussian':
        var=RNAkira.estimate_stddev (TPM,fig_name=options.out_prefix+'_variability_stddev.pdf' if options.save_figures else None)
    else:
        var=RNAkira.estimate_dispersion (counts.divide(SF.divide(np.exp(np.log(SF).mean(level=0)),level=0),axis=1),\
                                         fig_name=options.out_prefix+'_variability_disp.pdf' if options.save_figures else None)
else:
    print >> sys.stderr, '[test_RNAkira] run without estimation of variability\n'
    if options.statsmodel=='gaussian':
        var=stddev
    else:
        var=disp

########################################################################
#### RNAkira results                                                ####
########################################################################

results=RNAkira.RNAkira(counts, var, NF, T, alpha=options.alpha, criterion=options.criterion, min_ribo=.1, min_precursor=.1, \
                        models=None, constant_genes=constant_genes, maxlevel=options.maxlevel, statsmodel=options.statsmodel, \
                        priors=true_priors if options.use_true_priors else None)

output=RNAkira.collect_results(results, time_points).loc[genes]

output_true=pd.DataFrame([pd.DataFrame(parameters[gene],columns=time_points,\
                                       index=['initial_synthesis','initial_degradation','initial_processing','initial_translation']).stack() for gene in genes],index=genes)

if options.use_length_library_bias:
    output_true['initial_synthesis']-=np.log(SF['unlabeled-mature'].mean())
    output_true['initial_translation']+=np.log(SF['unlabeled-mature'].mean())-np.log(SF['ribo'].mean())
output_true.columns=[c[0]+'_t'+c[1] for c in output_true.columns.tolist()]

########################################################################
#### evaluate performance                                           ####
########################################################################

inferred_gene_class=output['best_model']

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
                                     priors=None,alpha=options.alpha)

tgc=true_gene_class.apply(lambda x: ''.join(m for m in x if m.isupper()))
igc=inferred_gene_class.apply(lambda x: ''.join(m for m in x if m.isupper()))
mods=sorted(np.union1d(tgc.unique(),igc.unique()),\
            key=lambda x: (len(x),x))
matches=np.array([[np.sum((tgc==m1) & (igc==m2)) for m2 in mods] for m1 in mods])

nexact=np.sum(np.diag(matches))
nover=np.sum(np.triu(matches,1))
nlim=sum(tgc!='ABCD')
nunder=np.sum(np.tril(matches,-1))
ntarget=sum(tgc!='')

stats='{0} exact hits ({1:.1f}%)\n{2} over-classifications ({3:.1f}%)\n{4} under-classifications ({5:.1f}%)'.format(nexact,100*nexact/float(nGenes),nover,100*nover/float(nlim),nunder,100*nunder/float(ntarget))
title='{0} genes, {1} time points, {2} replicates, {3} model\n{4}'.format(nGenes,len(time_points),nreps,options.statsmodel,stats)
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

if options.save_figures:
    fig.savefig(options.out_prefix+'_confusion_matrix.pdf')

if False:

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
            ax.set_ylabel('log fitted value'.format(options.statsmodel))

    fig.suptitle('{0} genes, {1} time points, {2} replicates, {3} model'.format(nGenes,len(time_points),nreps,options.statsmodel),size=10)

if False:
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

if False:

    mods=['abcd','Abcd','ABcd','ABCd','ABCD']

    R2_tot=pd.DataFrame.from_dict(dict((gene,pd.Series([results[gene][l]['R2_tot'] for l in [1,2,3,4,0]],index=mods)) for gene in genes),orient='index').loc[genes]
    R2_RNA=pd.DataFrame.from_dict(dict((gene,pd.Series([results[gene][l]['R2_RNA'] for l in [1,2,3,4,0]],index=mods)) for gene in genes),orient='index').loc[genes]
    R2_RPF=pd.DataFrame.from_dict(dict((gene,pd.Series([results[gene][l]['R2_ribo'] for l in [1,2,3,4,0]],index=mods)) for gene in genes),orient='index').loc[genes]

    R2=pd.concat([R2_tot,R2_RNA,R2_RPF],axis=1,keys=['tot','RNA','RPF'])

    use=R2['tot','ABCD'] > .5

    fig=plt.figure(figsize=(5,6))
    fig.clf()

    for j,l in enumerate(['tot','RNA','RPF']):
        ax=fig.add_axes([.15,.75-.28*j,.8,.22])
        for i,mod in enumerate(mods):
            for k,m in enumerate(mods):
                ax.bar(i+.15*k,R2[l,m][use & (true_gene_class==mod)].mean(),\
                       color='rgbcymk'[k],width=.15,lw=0,label=(mods[k] if i==0 else '_nolegend_'))
        ax.set_ylim([0,1])
        ax.set_xticks(np.arange(len(mods))+2.5*.15)
        ax.set_xticklabels([])
        ax.set_ylabel('R2 '+l)
        if j==2:
            ax.set_xticklabels(mods)
            ax.set_xlabel('true model')
            ax.legend(loc=3,bbox_to_anchor=(-.2,-.8),ncol=5,title='fitted model')





