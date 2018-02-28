import os
import sys
import numpy as np
import pandas as pd
import scipy.stats
from collections import defaultdict,OrderedDict
import itertools 
import RNAkira
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from optparse import OptionParser

np.random.seed(0)

# ignore warning about division by zero or over-/underflows
np.seterr(divide='ignore',over='ignore',under='ignore',invalid='ignore')

parser=OptionParser()
parser.add_option('','--maxlevel',dest='maxlevel',help="max level to test [5]",default=5,type=int)
parser.add_option('','--alpha',dest='alpha',help="model selection q-value cutoff [0.05]",default=0.05,type=float)
parser.add_option('','--LFC_cutoff',dest='LFC_cutoff',help="model selection LFC cutoff [0]",default=0,type=float)
parser.add_option('','--nreps',dest='nreps',help="number of replicates [5]",default=5,type=int)
parser.add_option('','--nconds',dest='nconds',help="number of conditions [6]",default=6,type=int)
parser.add_option('','--weight',dest='weight',help="weight for variability estimation [1]",default=1,type=float)
parser.add_option('','--model_selection',dest='model_selection',help="use model selection (LRT or empirical)")
parser.add_option('','--no_length_library_bias',dest='no_length_library_bias',action='store_true',default=False)
parser.add_option('','--no_ribo',dest='no_ribo',action='store_true',default=False)
parser.add_option('','--use_true_variability',dest='use_true_variability',action='store_true',default=False)
parser.add_option('','--use_true_normalization',dest='use_true_normalization',action='store_true',default=False)
parser.add_option('','--use_true_priors',dest='use_true_priors',action='store_true',default=False)
parser.add_option('','--do_direct_fits',dest='do_direct_fits',action='store_true',default=False)
parser.add_option('','--statsmodel',dest='statsmodel',default='nbinom')
parser.add_option('','--prior_weight',dest='prior_weight',default=1,type=float)
parser.add_option('','--out_prefix',dest='out_prefix',default='test_')
parser.add_option('','--save_input',dest='save_input',action='store_true',default=False)
parser.add_option('','--save_normalization_factors',dest='save_normalization_factors',action='store_true',default=False)
parser.add_option('','--save_variability',dest='save_variability',action='store_true',default=False)
parser.add_option('','--save_figures',dest='save_figures',action='store_true',default=False)
parser.add_option('','--save_results',dest='save_results',action='store_true',default=False)

options,args=parser.parse_args()

########################################################################
#### set parameters here                                            ####
########################################################################

# these are prior estimates on rates a,b,c,d similar to what we observe in our data
true_priors=pd.DataFrame(dict(mu=np.array([4.5,-1.5,.6,-2]),\
                              std=np.array([2,1,.5,.5])),\
                         index=list("abcd"))

full_model='ABCD'
rate_types=['synthesis','degradation','processing','translation']

# distribute models over genes
if options.no_ribo:
    full_model='ABC'
    rate_types=['synthesis','degradation','processing']

    true_gene_class=['abc']*9275+\
        ['Abc']*150+\
        ['aBc']*150+\
        ['abC']*150+\
        ['ABc']*75+\
        ['AbC']*75+\
        ['aBC']*75+\
        ['ABC']*50

else:

    true_gene_class=['abcd']*9210+\
        ['Abcd']*100+\
        ['aBcd']*100+\
        ['abCd']*100+\
        ['abcD']*100+\
        ['ABcd']*50+\
        ['AbCd']*50+\
        ['AbcD']*50+\
        ['aBCd']*50+\
        ['aBcD']*50+\
        ['abCD']*50+\
        ['ABCd']*20+\
        ['ABcD']*20+\
        ['AbCD']*20+\
        ['aBCD']*20+\
        ['ABCD']*10

# or use other designs for testing
if False:

    true_gene_class=['abcd']*1835+\
        ['Abcd']*20+\
        ['aBcd']*20+\
        ['abCd']*20+\
        ['abcD']*20+\
        ['ABcd']*10+\
        ['AbCd']*10+\
        ['AbcD']*10+\
        ['aBCd']*10+\
        ['aBcD']*10+\
        ['abCD']*10+\
        ['ABCd']*5+\
        ['ABcD']*5+\
        ['AbCD']*5+\
        ['aBCD']*5+\
        ['ABCD']*5

    #true_gene_class=['abcd']*50+['Abcd']*10+['aBcd']*10+['abCd']*10+['abcD']*10+['ABCD']*10
    true_gene_class=['abcd']*100

    full_model='ABC'
    rate_types=['synthesis','degradation','processing']
    true_gene_class=['abc']*2000

    true_gene_class=['abc']*1905+\
        ['Abc']*20+\
        ['aBc']*20+\
        ['abC']*20+\
        ['ABc']*10+\
        ['AbC']*10+\
        ['aBC']*10+\
        ['ABC']*5

cols=['elu-mature','flowthrough-mature','unlabeled-mature','elu-precursor','flowthrough-precursor','unlabeled-precursor','ribo']

# define conditions
nconds=options.nconds
conditions=list('ABCDEFGHIKLMNOPQRST')[:nconds]
# define number of replicates
nreps=options.nreps
replicates=map(str,range(nreps))
samples=list(itertools.product(conditions,replicates))
# labeling time
T=1
# parameters of dispersion curve (set intercept=1 and slope=0 for Poisson model, otherwise neg binomial)
slope,intercept=0.01,2
#slope,intercept=0,1
# average fold change between conditions
AVE_FC=2
# arguments for minimization 
min_args=dict(method='L-BFGS-B',jac=True,options={'disp':False, 'ftol': 1.e-15, 'gtol': 1.e-10})

########################################################################
#### simulate data                                                  ####
########################################################################

nGenes=len(true_gene_class)
genes=np.array(map(lambda x: '_'.join(x), zip(['gene']*nGenes,map(str,range(nGenes)),true_gene_class)))
#T=pd.DataFrame([[1,2]]*len(genes),index=genes,columns=conditions)
T=pd.DataFrame(T,index=genes,columns=conditions)

if options.no_length_library_bias:
    gene_stats=pd.DataFrame(dict(exon_length=1000,gene_type='protein_coding',exon_ucount=250),index=genes)
else:
    # random lengths and ucounts for genes
    gene_stats=pd.DataFrame(dict(exon_length=(10**scipy.stats.norm.rvs(3.0,scale=.56,size=nGenes)).astype(int),\
                                 gene_type=['protein_coding']*nGenes),index=genes)
    gene_stats['exon_ucount']=1+(.25*gene_stats['exon_length']).astype(int)

true_gene_class=pd.Series(true_gene_class,index=genes)

print >> sys.stderr, '[test_RNAkira] drawing parameters for {0} genes ({1} conditions, {2} replicates)'.format(nGenes,nconds,nreps)
print >> sys.stderr, '[test_RNAkira] true priors for log_a: {0:.2g}/{1:.2g}, log_b: {2:.2g}/{3:.2g}, log_c: {4:.2g}/{5:.2g}, log_d: {6:.2g}/{7:.2g}'.format(*true_priors.iloc[:4].values.flatten())

parameters={}

for ng,gene in enumerate(genes):

    model=true_gene_class[gene]
    # first draw constant baselines for each parameter
    pars=[scipy.stats.norm.rvs(true_priors.loc[mp,'mu'],true_priors.loc[mp,'std']) for mp in model.lower()]
    # give DE genes a bit more reads (make synthesis rate 1 log higher)
    if model!=model.lower():
        pars[0]+=1
    # then expand this over conditions (add randomness of AVE_FC per condition for variable parameters)
    parameters[gene]=np.array([p*np.ones(nconds) if mp.islower() else \
                               scipy.stats.norm.rvs(p,np.log(AVE_FC),size=nconds) for mp,p in zip(model,pars)])

parameters=pd.DataFrame([pd.DataFrame(parameters[gene],columns=conditions,\
                                      index=rate_types).stack() for gene in genes],index=genes)

counts={}
disp={}
stddev={}

# this is used to calculate size factors for elu + flowthrough
if options.no_length_library_bias:
    size_factor=pd.Series(1,index=cols)
else:
    UF=np.exp(parameters['synthesis']-parameters['degradation']).mean(axis=0).mean()
    EF=(np.exp(parameters['synthesis']-parameters['degradation'])*\
        (1-np.exp(-np.exp(parameters['degradation']).multiply(T,axis=0)))).mean(axis=0).mean()
    FF=(np.exp(parameters['synthesis']-parameters['degradation'])*\
        (np.exp(-np.exp(parameters['degradation']).multiply(T,axis=0)))).mean(axis=0).mean()
    size_factor=pd.Series([UF/EF,UF/FF,1,UF/EF,UF/FF,1,1],index=cols)

for ng,gene in enumerate(genes):

    print >> sys.stderr, '[test_RNAkira] {0} genes initialized\r'.format(ng+1),

    # now get random values for observations according to these rate parameters
    pars=parameters.loc[gene]
    cnts=pd.Series(index=pd.MultiIndex.from_product([cols,conditions,replicates]))
    std=pd.Series(index=pd.MultiIndex.from_product([cols,conditions]))
    dsp=pd.Series(index=pd.MultiIndex.from_product([cols,conditions]))

    # this is used to model U-pulldown bias
    ubias=1.-.5*np.exp(-gene_stats.loc[gene,'exon_ucount']/500.)

    model=true_gene_class[gene]
    cols_here=['elu-mature','flowthrough-mature','unlabeled-mature']
    if 'c' in model.lower():
        cols_here+=['elu-precursor','flowthrough-precursor','unlabeled-precursor']
    if 'd' in model.lower():
        cols_here+=['ribo']

    for cond in conditions:

        # get expected values given these rates
        mu=RNAkira.get_steady_state_values(pars.xs(cond,level=1),T.loc[gene,cond],model)
        if options.no_length_library_bias:
            mu_eff=mu
        else:
            # multiply by gene length and library size factors and introduce U-bias 
            mu_eff=mu*(gene_stats.loc[gene,'exon_length']/1.e3)*size_factor[cols_here]
            mu_eff[0]=mu_eff[0]*ubias

        # get counts based on these expected values
        for m,col in zip(mu_eff,cols_here):
            d=(intercept-1)/m+slope
            if d < 1.e-8:
                cnts[col,cond]=scipy.stats.poisson.rvs(m,size=nreps)
            else:
                cnts[col,cond]=scipy.stats.nbinom.rvs(1./d,1./(1.+d*m),size=nreps)
            std[col,cond]=np.sqrt(m*(1.+d*m))
            dsp[col,cond]=d

    counts[gene]=cnts
    stddev[gene]=std
    disp[gene]=dsp

    # use this to fit models directly and abort after one gene
    if options.do_direct_fits:

        vals_here=cnts.unstack(level=0)[cols_here].stack().values.reshape((len(conditions),nreps,len(cols_here)))
        std_here=std.mean(level=0)[cols_here].values
        disp_here=dsp.mean(level=0)[cols_here].values
        if options.no_length_library_bias:
            nf_here=np.ones_like(vals_here)
        else:
            nf_here=1./(size_factor[cols_here].values*gene_stats.loc[gene,'exon_length']/1.e3)
            nf_here[0]=nf_here[0]*ubias
            nf_here=np.tile(nf_here,(nconds,nreps)).reshape(vals_here.shape)

        if options.statsmodel=='gaussian':
            res={}
            if options.no_ribo:
                priors=true_priors.loc[list('abc')]
                res['ABC']=RNAkira.fit_model(vals_here,std_here,nf_here,T.loc[gene],conditions,priors,options.prior_weight,None,'ABC','gaussian',min_args)
                res['abc']=RNAkira.fit_model(vals_here,std_here,nf_here,T.loc[gene],conditions,priors,options.prior_weight,res['ABC'],'abc','gaussian',min_args)
                res['Abc']=RNAkira.fit_model(vals_here,std_here,nf_here,T.loc[gene],conditions,priors,options.prior_weight,res['abc'],'Abc','gaussian',min_args)
                res['ABc']=RNAkira.fit_model(vals_here,std_here,nf_here,T.loc[gene],conditions,priors,options.prior_weight,res['Abc'],'ABc','gaussian',min_args)
            else:
                priors=true_priors.loc[list('abcd')]
                res['ABCD']=RNAkira.fit_model(vals_here,std_here,nf_here,T.loc[gene],conditions,priors,options.prior_weight,None,'ABCD','gaussian',min_args)
                res['abcd']=RNAkira.fit_model(vals_here,std_here,nf_here,T.loc[gene],conditions,priors,options.prior_weight,res['ABCD'],'abcd','gaussian',min_args)
                res['Abcd']=RNAkira.fit_model(vals_here,std_here,nf_here,T.loc[gene],conditions,priors,options.prior_weight,res['abcd'],'Abcd','gaussian',min_args)
                res['ABcd']=RNAkira.fit_model(vals_here,std_here,nf_here,T.loc[gene],conditions,priors,options.prior_weight,res['Abcd'],'ABcd','gaussian',min_args)
                res['ABCd']=RNAkira.fit_model(vals_here,std_here,nf_here,T.loc[gene],conditions,priors,options.prior_weight,res['ABcd'],'ABCd','gaussian',min_args)
        else:
            res={}
            if options.no_ribo:
                priors=true_priors.loc[list('abc')]
                res['ABC']=RNAkira.fit_model(vals_here,disp_here,nf_here,T.loc[gene],conditions,priors,options.prior_weight,None,'ABC','nbinom',min_args)
                res['abc']=RNAkira.fit_model(vals_here,disp_here,nf_here,T.loc[gene],conditions,priors,options.prior_weight,res['ABC'],'abc','nbinom',min_args)
                res['Abc']=RNAkira.fit_model(vals_here,disp_here,nf_here,T.loc[gene],conditions,priors,options.prior_weight,res['abc'],'Abc','nbinom',min_args)
                res['ABc']=RNAkira.fit_model(vals_here,disp_here,nf_here,T.loc[gene],conditions,priors,options.prior_weight,res['Abc'],'ABc','nbinom',min_args)
            else:
                priors=true_priors.loc[list('abcd')]
                res['ABCD']=RNAkira.fit_model(vals_here,disp_here,nf_here,T.loc[gene],conditions,priors,options.prior_weight,None,'ABCD','nbinom',min_args)
                res['abcd']=RNAkira.fit_model(vals_here,disp_here,nf_here,T.loc[gene],conditions,priors,options.prior_weight,res['ABCD'],'abcd','nbinom',min_args)
                res['Abcd']=RNAkira.fit_model(vals_here,disp_here,nf_here,T.loc[gene],conditions,priors,options.prior_weight,res['abcd'],'Abcd','nbinom',min_args)
                res['ABcd']=RNAkira.fit_model(vals_here,disp_here,nf_here,T.loc[gene],conditions,priors,options.prior_weight,res['Abcd'],'ABcd','nbinom',min_args)
                res['ABCd']=RNAkira.fit_model(vals_here,disp_here,nf_here,T.loc[gene],conditions,priors,options.prior_weight,res['ABcd'],'ABCd','nbinom',min_args)

        raise Exception('stop')

counts=pd.DataFrame.from_dict(counts,orient='index').loc[genes]
disp=pd.DataFrame.from_dict(disp,orient='index').loc[genes]
stddev=pd.DataFrame.from_dict(stddev,orient='index').loc[genes]

print >> sys.stderr, ''

if options.model_selection=='empirical':
    # constant genes have no intronic or ribo coverage
    constant_genes=true_gene_class.index[(true_gene_class=='abcd')][:400]
    counts.loc[constant_genes,'ribo']=np.nan
    counts.loc[constant_genes,'unlabeled-precursor']=np.nan
    counts.loc[constant_genes,'elu-precursor']=np.nan
    counts.loc[constant_genes,'flowthrough-precursor']=np.nan
else:
    constant_genes=None

if options.save_input:
    # save input data
    print >> sys.stderr, '[test_RNAkira] saving input'
    dummy=pd.DataFrame('NA',columns=['dummy'+str(k) for k in range(4)],index=genes)
    parameters.to_csv(options.out_prefix+'parameters.csv',\
                       header=[c[0]+'_t'+c[1] for c in parameters.columns.tolist()],tupleize_cols=True)
    gene_stats.to_csv(options.out_prefix+'gene_stats.csv')
    counts.to_csv(options.out_prefix+'counts.csv',header=['.'.join(c) for c in counts.columns.tolist()],tupleize_cols=True)
    for col in cols:
        tmp=counts[col].fillna(0).astype(int)
        tmp.columns=['.'.join(c) for c in tmp.columns.tolist()]
        pd.concat([dummy,gene_stats['exon_length'],tmp],axis=1).to_csv(options.out_prefix+col+'.tsv',sep='\t')

########################################################################
#### normalization, U-bias correction                               ####
########################################################################

if options.use_true_normalization:

    print >> sys.stderr, '[test_RNAkira] use true normalization factors'
    LF=gene_stats['exon_length']/1.e3
    SF=pd.Series(size_factor.repeat(len(conditions)*nreps).values,index=counts.columns)
    UF=pd.DataFrame(1,index=counts.index,columns=counts.columns)
    if not options.no_length_library_bias:
        UF['elu-mature']=UF['elu-mature'].divide(1.-.5*np.exp(-gene_stats['exon_ucount']/500.),axis=0)
    CF=pd.Series(1,index=counts.columns)
    NF=UF.multiply(CF).divide(LF,axis=0).divide(SF,axis=1).fillna(1)

else:

    # normalize by "sequencing depth" and length (so here TPM=RPKM)
    LF=gene_stats['exon_length']/1.e3
    RPK=counts.divide(LF,axis=0)
    EF=(RPK['elu-mature'].add(RPK['elu-precursor'],fill_value=0).sum(axis=0))/1.e6
    FF=(RPK['flowthrough-mature'].add(RPK['flowthrough-precursor'],fill_value=0).sum(axis=0))/1.e6
    UF=RPK['unlabeled-mature'].add(RPK['unlabeled-precursor'],fill_value=0).sum(axis=0)/1.e6
    RF=RPK['ribo'].sum(axis=0)/1.e6
    SF=pd.concat([EF,FF,UF,\
                  EF,FF,UF,\
                  RF],axis=0,keys=cols).fillna(1)
    TPM=RPK.divide(SF,axis=1)

    if options.no_length_library_bias:
        UF=pd.DataFrame(1,index=counts.index,columns=counts.columns)
    else:
        UF=RNAkira.correct_ubias(TPM,samples,gene_stats,fig_name=options.out_prefix+'ubias_correction.pdf' if options.save_figures else None)
    CF=RNAkira.normalize_elu_flowthrough_over_genes(TPM.multiply(UF),samples,fig_name=options.out_prefix+'TPM_correction.pdf' if options.save_figures else None)

    NF=UF.multiply(CF).divide(LF,axis=0).divide(SF,axis=1).fillna(1)

TPM=counts.multiply(NF)

if options.save_normalization_factors:
    print >> sys.stderr, '[test_RNAkira] saving normalization factors'
    UF.multiply(CF).divide(SF,axis=1).fillna(1).to_csv(options.out_prefix+'normalization_factors.csv',\
                                                       header=['.'.join(c) for c in NF.columns.tolist()],tupleize_cols=True)

if options.use_true_variability:
    print >> sys.stderr, '[test_RNAkira] use true variability'
    if options.statsmodel=='gaussian':
        var=stddev.mean(axis=1,level=0)
    else:
        var=disp.mean(axis=1,level=0)
else:
    if options.statsmodel=='gaussian':
        var=RNAkira.estimate_stddev (TPM, options.weight/float(nreps),\
                                     fig_name=options.out_prefix+'variability_stddev.pdf' if options.save_figures else None)
    else:
        nf_scaled=NF.divide(np.exp(np.log(NF).mean(axis=1,level=0)),axis=0,level=0)
        var=RNAkira.estimate_dispersion (counts.divide(nf_scaled,axis=1),\
                                         options.weight/float(nreps),\
                                         fig_name=options.out_prefix+'variability_disp.pdf' if options.save_figures else None)

if options.save_variability:
    print >> sys.stderr, '[test_RNAkira] saving variability estimates'
    var.to_csv(options.out_prefix+'variability.csv')

print >> sys.stderr, ''

########################################################################
#### RNAkira results                                                ####
########################################################################

take=(TPM['unlabeled-mature'] > 1).any(axis=1) & \
    ~var[['unlabeled-mature','elu-mature','flowthrough-mature']].isnull().any(axis=1)

results=RNAkira.RNAkira(counts[take], var[take], NF[take], T[take], \
                        alpha=options.alpha, LFC_cutoff=options.LFC_cutoff, \
                        model_selection=options.model_selection, \
                        constant_genes=np.intersect1d(constant_genes,TPM[take].index), \
                        maxlevel=options.maxlevel, statsmodel=options.statsmodel, \
                        priors=true_priors if options.use_true_priors else None, \
                        prior_weight=options.prior_weight)

output=RNAkira.collect_results(results, conditions, select_best=(options.model_selection is not None)).loc[genes][take]

if options.save_results:
    print >> sys.stderr, '[test_RNAkira] saving results'
    output.to_csv(options.out_prefix+'results.csv')

########################################################################
#### evaluate performance                                           ####
########################################################################

tgc=true_gene_class[take].apply(lambda x: '0' if x.islower() else ''.join(m for m in x if m.isupper()))

if not options.no_length_library_bias:
    # synthesis rate is measured in different units (TPM/h instead of counts)
    parameters['synthesis']-=np.log(SF['unlabeled-mature'].mean())
    if not options.no_ribo:
        # translation efficiencies cannot measure global shift
        parameters['translation']+=np.log(SF['unlabeled-mature'].mean())-np.log(SF['ribo'].mean())
parameters.columns=[c[0]+'_t'+c[1] for c in parameters.columns.tolist()]
parameters=parameters[take]
genes=genes[take]
nGenes=len(genes)

if options.model_selection is not None:

    igc=output['best_model'].apply(lambda x: '0' if x.islower() else ''.join(m for m in x if m.isupper()))

    mods=sorted(np.union1d(tgc.unique(),igc.unique()),\
                key=lambda x: (len(x),x))
    matches=np.array([[np.sum((tgc==m1) & (igc==m2)) for m2 in mods] for m1 in mods])

    nexact=np.sum(np.diag(matches))
    nover=np.sum(np.triu(matches,1))
    nlim=sum(tgc!=full_model)
    nunder=np.sum(np.tril(matches,-1))
    ntarget=sum(tgc!='0')

    stats='{0} exact hits ({1:.1f}%)\n{2} over-classifications ({3:.1f}%)\n{4} under-classifications ({5:.1f}%)'.format(nexact,100*nexact/float(nGenes),nover,100*nover/float(nlim),nunder,100*nunder/float(ntarget))
    title='{0} genes, {1} conditions, {2} replicates, {3} model\n{4}'.format(nGenes,nconds,nreps,options.statsmodel,stats)
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
        fig.savefig(options.out_prefix+'confusion_matrix.pdf')

if True: # compare fitted values directly to true rate parameters

    fig=plt.figure()
    fig.clf()
    fig.subplots_adjust(hspace=.4,wspace=.4)

    for n,r in enumerate(rate_types):
        if options.model_selection is None:
            output_cols=['{0}_{1}_{2}'.format(full_model,r,cond) for cond in conditions]
        else:
            output_cols=['initial_{0}_{1}'.format(r,cond) for cond in conditions]
        par_cols=['{0}_t{1}'.format(r,cond) for cond in conditions]
        ax=fig.add_subplot(2,2,n+1)
        y,x=output[output_cols].values.flatten(),parameters[par_cols].values.flatten()
        ok=np.isfinite(x) & np.isfinite(y)
        xr=np.percentile(x[ok],[1,99])
        yr=np.percentile(y[ok],[1,99])
        ax.hexbin(x[ok],y[ok],extent=(xr[0],xr[1],yr[0],yr[1]),bins='log',mincnt=1,vmin=-1)
        ax.plot(np.linspace(xr[0],xr[1],100),np.linspace(xr[0],xr[1],100),'r-',lw=.5)
        ax.set_title('{0}'.format(r),size=10)
        good = np.sum(np.abs(np.log2(y/x)[ok]) < 2)
        ax.set_xlim(xr)
        ax.set_ylim(yr)
        ax.text(xr[0]+.05*(xr[1]-xr[0]),yr[1]-.05*(yr[1]-yr[0]),'{0:.0f}% within 2fold\nr={1:.2f}\nrho={2:.2f}\nn={3}'.format(100*good/float(ok.sum()),scipy.stats.pearsonr(x[ok],y[ok])[0],scipy.stats.spearmanr(x[ok],y[ok])[0],ok.sum()),size=6,va='top',ha='left')
        if n > 1:
            ax.set_xlabel('log true value')
        if n%2==0:
            ax.set_ylabel('log fitted value'.format(options.statsmodel))

    fig.suptitle('{0} genes, {1} conditions, {2} replicates, {3} model'.format(nGenes,nconds,nreps,options.statsmodel),size=10)

    if options.save_figures:
        fig.savefig(options.out_prefix+'parameter_fits.pdf')

if options.model_selection is None:

    output.columns=pd.MultiIndex.from_tuples([(c.split('_')[0],'_'.join(c.split('_')[1:])) for c in output.columns])
    if options.no_ribo:
        tested_models1=['aBC','AbC','ABc']
        tested_models2=['Abc','aBc','abC']
    else:
        tested_models1=['aBCD','AbCD','ABcD','ABCd']
        tested_models2=['Abcd','aBcd','abCd','abcD']
    true_models=tgc.unique()

    R2=output.xs("R2_tot",axis=1,level=1)[tested_models1+tested_models2+[full_model]]

    use=R2[full_model] > .5

    import statsmodels.graphics.boxplots as sgb
    import matplotlib.patches as mpatches

    fig=plt.figure(figsize=(12,6))
    fig.clf()

    for j,tmod in enumerate([tested_models1,tested_models2]):
        ax=fig.add_axes([.12,.65-.45*j,.8,.3])
        for i,mod in enumerate(true_models):
            for k,m in enumerate(tmod):
                color='rgbcymk'[k]
                vals=(R2[full_model]-R2[m])[use & (tgc==mod)].dropna()
                if len(vals) > 3:
                    sgb.violinplot([vals],ax=ax,positions=[i+.2*k],show_boxplot=False,\
                                   plot_opts=dict(violin_fc=color,violin_ec=color,violin_alpha=.5,violin_width=.15,cutoff=True))
                bp=ax.boxplot([vals],positions=[i+.2*k],widths=.1,sym='',notch=False)
                plt.setp(bp['boxes'],color='k',linewidth=1)
                plt.setp(bp['whiskers'],color='k',linestyle='solid',linewidth=.5)
                plt.setp(bp['caps'],color='k')
                plt.setp(bp['medians'],color='r')
        patches=[mpatches.Patch(color='rgbcymk'[k],alpha=.5) for k in range(len(tmod))]
        #ax.set_ylim([0,1])
        ax.set_yscale('log')
        ax.set_xticks(np.arange(len(true_models))+2*.2-.1)
        ax.set_xticklabels([])
        ax.set_xlim([-.2,len(true_models)-.2])
        ax.set_ylabel('unexplained variance')
        ax.set_xticklabels(true_models)
        ax.set_xlabel('true model')
        if j==0:
            ax.set_title('one parameter constant, others vary',size=10)
        else:
            ax.set_title('one parameter varies, others constant',size=10)
            leg=ax.legend(patches,list(full_model),loc=3,ncol=4,bbox_to_anchor=(-.1,-.5),title='parameter')

    if options.save_figures:
        fig.savefig(options.out_prefix+'R2_stats.pdf')
