import os
import sys
import numpy as np
import pandas as pd
import scipy.stats
from collections import defaultdict,OrderedDict
import RNAkira
from matplotlib import pyplot as plt
from optparse import OptionParser

plt.ion()
np.random.seed(0)

# change here if you want to test other scenarios
# these are prior estimates similar to what we observe in our data
true_priors=pd.DataFrame(dict(mu=np.array([.6,-2.3,1.2,0.1,.01,.01,.01,.01]),std=np.array([1.3,1.,1.5,.6,.01,.005,.005,.005])),\
						 index=['log_a0','log_b0','log_c0','log_d0','alpha','beta','gamma','delta'])

# distribute models over genes, make sure most genes don't change for multiple testing correction to work
true_gene_class=['ribo_0']*2000+\
	['ribo_a']*50+\
	['ribo_b']*50+\
	['ribo_c']*50+\
	['ribo_d']*50+\
	['ribo_ab']*20+\
	['ribo_ac']*20+\
	['ribo_ad']*20+\
	['ribo_bc']*20+\
	['ribo_bd']*20+\
	['ribo_cd']*20+\
	['ribo_abc']*15+\
	['ribo_acd']*15+\
	['ribo_abd']*15+\
	['ribo_bcd']*15+\
	['ribo_abcd']*20+\
	['ribo_all']*100


# define time points
time_points=['0','12','24','36','48']
times=map(float,time_points)
# define number of replicates
replicates=map(str,range(10))
# labeling time
T=1
# noise level (decrease for testing)
sigma_f=1
# parameters of mean-variance relationship: std = sqrt(mu + 2**(-ma)*mu**(2-mb))
ma,mb=5.,0.

# model definition (same as in RNAkira)

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

nlevels=len(models)
model_pars=dict((m,v[0]) for l in range(nlevels) for m,v in models[l].iteritems())

parser=OptionParser()
parser.add_option('-i','--values',dest='values',help="file with simulated TPM (created by test_RNAkira.py)")
parser.add_option('-o','--outf',dest='outf',help="save simulated TPM to this file")
parser.add_option('-T','--labeling_time',dest='T',help="labeling time (default: 1)", default=1)
parser.add_option('','--maxlevel',dest='maxlevel',help="max level to test (default: 5)",default=5,type=int)
parser.add_option('','--alpha',dest='alpha',help="FDR cutoff (default: 0.05)",default=0.05)

options,args=parser.parse_args()

sig_level=float(options.alpha)
T=float(options.T)

if options.values is not None:

	print >> sys.stderr, '\n[test_RNAkira] reading simulated data from '+options.values
	values=pd.read_csv(options.values,index_col=0,header=range(4))
	values.columns=pd.MultiIndex.from_tuples([(c[0],c[1],c[2],int(c[3])) for c in values.columns.tolist()])

	time_points=np.unique(zip(*values.columns.tolist())[2])
	replicates=np.unique(zip(*values.columns.tolist())[3])
	cols=np.unique(zip(*values.columns.tolist())[1])
	genes=values.index.values
	nGenes=len(genes)
	true_gene_class=pd.Series(np.array(['_'.join(g.split('_')[2:]) for g in genes]),index=genes)
	parameters_known=False

else:
	
	cols=['elu-precursor','elu-total','flowthrough-precursor','flowthrough-total','ribo','unlabeled-precursor','unlabeled-total']

	nGenes=len(true_gene_class)
	genes=np.array(map(lambda x: '_'.join(x), zip(['gene']*nGenes,map(str,range(nGenes)),true_gene_class)))

	true_gene_class=pd.Series(true_gene_class,index=genes)

	parameters_known=True
	parameters={}
	values={}

	print >> sys.stderr, '\n[test_RNAkira] drawing parameters and observations for {0} genes ({1} time points, {2} replicates)'.format(nGenes,len(time_points),len(replicates))
	print >> sys.stderr, '[test_RNAkira] true priors for log_a: {0:.2g}/{1:.2g}, log_b: {2:.2g}/{3:.2g}, log_c: {4:.2g}/{5:.2g}, log_d: {6:.2g}/{7:.2g}'.format(*true_priors.ix[:4].values.flatten())

	for gene in genes:

		model=true_gene_class[gene]
		if model=='ribo_all':
			log_a=scipy.stats.norm.rvs(scipy.stats.norm.rvs(true_priors.ix['log_a0','mu'],true_priors.ix['log_a0','std']),\
									   .5*true_priors.ix['log_a0','std'],size=len(time_points))
			log_b=scipy.stats.norm.rvs(scipy.stats.norm.rvs(true_priors.ix['log_b0','mu'],true_priors.ix['log_b0','std']),\
									   .5*true_priors.ix['log_b0','std'],size=len(time_points))
			log_c=scipy.stats.norm.rvs(scipy.stats.norm.rvs(true_priors.ix['log_c0','mu'],true_priors.ix['log_c0','std']),\
									   .5*true_priors.ix['log_c0','std'],size=len(time_points))
			log_d=scipy.stats.norm.rvs(scipy.stats.norm.rvs(true_priors.ix['log_d0','mu'],true_priors.ix['log_d0','std']),\
									   .5*true_priors.ix['log_d0','std'],size=len(time_points))
			pars=pd.DataFrame([log_a,log_b,log_c,log_d],columns=time_points,index=['log_a0','log_b0','log_c0','log_d0']).T
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
			log_a,log_b,log_c,log_d=RNAkira.get_rates(time_points,pars)
			
		# now get random values for observations according to these rate parameters
		vals=[]

		for i,t in enumerate(time_points):

			mu=RNAkira.get_steady_state_values([log_a[i],log_b[i],log_c[i],log_d[i]],T,use_ribo=True)

			# use overdispersed gamma distribution (here: std = sqrt(mu + 2^(-ma) mu^(2-mb)))
			std=sigma_f*np.sqrt(mu+2**(-ma)*mu**(2-mb))

			vals.append([scipy.stats.gamma.rvs((mu[n]/std[n])**2,scale=std[n]**2/mu[n],size=len(replicates)) for n in range(len(mu))])
			vals.append([np.ones(len(replicates))*std[n] for n in range(len(std))])

		parameters[gene]=pars
		values[gene]=np.array(vals).flatten()

	values=pd.DataFrame.from_dict(values,orient='index')

	values.columns=pd.MultiIndex.from_tuples([(t,stat,c,r) for t in time_points for stat in ['mean','std'] for c in cols for r in replicates])
	values=values.reorder_levels([1,2,0,3],axis=1).sort_index(axis=1)

	if (values < 0).any().any():
		raise Exception('invalid random values!')

	if options.outf is not None:
		print >> sys.stderr, '[test_RNAkira] saving to '+options.outf
		values.to_csv(options.outf)

results=RNAkira.RNAkira(values, T, sig_level=sig_level, min_TPM_ribo=0.01, maxlevel=options.maxlevel)

output=RNAkira.collect_results(results, time_points, sig_level=sig_level)

print >> sys.stderr, '\n[test_RNAkira] evaluating performance'

inferred_gene_class=output.ix[genes,'best_model']
inferred_gene_class[output.ix[genes,'initial_qval'] < .05]='ribo_all'

# use this if you want to plot specific examples
if False:

	genes_to_plot=genes[np.where(inferred_gene_class!=true_gene_class)[0]]
	np.random.shuffle(genes_to_plot)

	for k,gene in enumerate(genes_to_plot[:min(5,len(genes_to_plot))]):
		RNAkira.plot_data_rates_fits(time_points,replicates,values.ix[gene,'mean'],T,\
									 parameters[gene] if parameters_known else None,\
									 results[gene],True,\
									 title='{0} (true: {1}, inferred: {2})'.format(gene,true_gene_class[gene],inferred_gene_class[gene]),\
									 priors=None,sig_level=sig_level)

mods=[m for lev in [1,2,3,4,5,0] for m in models[lev] if 'rna' not in m]
matches=np.array([[np.sum((true_gene_class==m1) & (inferred_gene_class==m2)) for m2 in mods] for m1 in mods])

nexact=np.sum(np.diag(matches))
nover=np.sum(np.triu(matches,1))
nlim=sum(true_gene_class!='all')
nunder=np.sum(np.tril(matches,-1))
ntarget=sum(true_gene_class!='0')

stats='{0} exact hits ({1:.1f}%)\n{2} over-classifications ({3:.1f}%)\n{4} under-classifications ({5:.1f}%)'.format(nexact,100*nexact/float(nGenes),nover,100*nover/float(nlim),nunder,100*nunder/float(ntarget))
title='{0} genes, {1} time points, {2} replicates\n{3}'.format(nGenes,len(time_points),len(replicates),stats)
print >> sys.stderr, stats
		
fig=plt.figure(figsize=(5,5.5))
fig.clf()

ax=fig.add_axes([.2,.2,.75,.65])
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

