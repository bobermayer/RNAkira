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

# model definition (same as in RNAkira)

models=OrderedDict([(0,OrderedDict([('all',(['log_a0','log_b0','log_c0','log_d0'],[]))])),\
					(1,OrderedDict([('0',(['log_a0','log_b0','log_c0','log_d0'],['all']))])),\
					(2,OrderedDict([('a',(['log_a0','log_b0','log_c0','log_d0','alpha'],['0'])),\
									('b',(['log_a0','log_b0','log_c0','log_d0','beta'],['0'])),\
									('c',(['log_a0','log_b0','log_c0','log_d0','gamma'],['0'])),\
									('d',(['log_a0','log_b0','log_c0','log_d0','delta'],['0']))])),\
					(3,OrderedDict([('ab',(['log_a0','log_b0','log_c0','log_d0','alpha','beta'],['a','b'])),\
									('ac',(['log_a0','log_b0','log_c0','log_d0','alpha','gamma'],['a','c'])),\
									('ad',(['log_a0','log_b0','log_c0','log_d0','alpha','delta'],['a','d'])),\
									('bc',(['log_a0','log_b0','log_c0','log_d0','beta','gamma'],['b','c'])),\
									('bd',(['log_a0','log_b0','log_c0','log_d0','beta','delta'],['b','d'])),\
									('cd',(['log_a0','log_b0','log_c0','log_d0','gamma','delta'],['c','d']))])),\
					(4,OrderedDict([('abc',(['log_a0','log_b0','log_c0','log_d0','alpha','beta','gamma'],['ab','ac','bc'])),\
									('abd',(['log_a0','log_b0','log_c0','log_d0','alpha','beta','delta'],['ab','ad','bd'])),\
									('acd',(['log_a0','log_b0','log_c0','log_d0','alpha','gamma','delta'],['ac','ad','cd'])),\
									('bcd',(['log_a0','log_b0','log_c0','log_d0','beta','gamma','delta'],['bc','bd','cd']))])),\
					(5,OrderedDict([('abcd',(['log_a0','log_b0','log_c0','log_d0','alpha','beta','gamma','delta'],['abc','abd','acd','bcd']))]))])

nlevels=len(models)
model_pars=dict((m,v[0]) for l in range(nlevels) for m,v in models[l].iteritems())

# change here if you want to test other scenarios

true_priors=pd.DataFrame(dict(mu=np.array([2,-2.5,2,1,.01,.005,.005,.005]),std=np.array([1,.5,1,1.3,.002,.001,.001,.001])),\
						 index=['log_a0','log_b0','log_c0','log_d0','alpha','beta','gamma','delta'])

true_gene_class=['a']*100+\
	['b']*100+\
	['c']*100+\
	['d']*100+\
	['ab']*20+\
	['ac']*20+\
	['ad']*20+\
	['bc']*20+\
	['bd']*20+\
	['cd']*20+\
	['abc']*15+\
	['acd']*15+\
	['abd']*15+\
	['bcd']*15+\
	['abcd']*10+\
	['all']*10+\
	['0']*1400


parser=OptionParser()
parser.add_option('-i','--values',dest='values',help="file with simulated TPM (created by test_RNAkira.py)")
parser.add_option('-o','--outf',dest='outf',help="save simulated TPM to this file")
parser.add_option('-T','--labeling_time',dest='T',help="labeling time (default: 1)", default=1)
parser.add_option('','--maxlevel',dest='maxlevel',help="max level to test (default: 5)",default=5,type=int)
parser.add_option('','--alpha',dest='alpha',default=0.05,help="FDR cutoff (default: 0.05)",default=0.05)

options,args=parser.parse_args()

sig_level=float(options.alpha)
T=float(options.T)

if options.values is not None:

	print >> sys.stderr, '[test_RNAkira] reading simulated data from '+options.values
	values=pd.read_csv(options.values,index_col=0,header=range(4))

	time_points=np.unique(map(int,zip(*values.columns.tolist())[2]))
	replicates=np.unique(zip(*values.columns.tolist())[3])
	cols=np.unique(zip(*values.columns.tolist())[1])
	ndim=len(cols)
	genes=values.index.values
	nGenes=len(genes)
	true_gene_class=pd.Series(np.array([g.split('_')[-1] for g in genes]),index=genes)

else:

	time_points=np.array([0,12,24,36,48])
	replicates=range(4)
	T=1
	sigma_f=1
	ma,mb=9.8,-0.26

	ndim=7
	cols=['elu-precursor','elu-total','flowthrough-precursor','flowthrough-total','ribo','unlabeled-precursor','unlabeled-total']

	nGenes=len(true_gene_class)
	genes=np.array(map(lambda x: '_'.join(x), zip(['gene']*nGenes,map(str,range(nGenes)),true_gene_class)))

	true_gene_class=pd.Series(true_gene_class,index=genes)

	parameters={}
	values={}

	print >> sys.stderr, '\n[test_RNAkira] drawing parameters and observations for {0} genes ({1} time points, {2} replicates)'.format(nGenes,len(time_points),len(replicates))
	print >> sys.stderr, '[test_RNAkira] true priors for log_a: {0:.2g}/{1:.2g}, log_b: {2:.2g}/{3:.2g}, log_c: {4:.2g}/{5:.2g}, log_d: {6:.2g}/{7:.2g}'.format(*true_priors.ix[:4].values.flatten())

	for gene in genes:

		model=true_gene_class[gene]
		if model=='all':
			log_a=scipy.stats.norm.rvs(scipy.stats.norm.rvs(true_priors.ix['log_a0','mu'],true_priors.ix['log_a0','std']),.5*true_priors.ix['log_a0','std'],size=len(time_points))
			log_b=scipy.stats.norm.rvs(scipy.stats.norm.rvs(true_priors.ix['log_b0','mu'],true_priors.ix['log_b0','std']),.5*true_priors.ix['log_b0','std'],size=len(time_points))
			log_c=scipy.stats.norm.rvs(scipy.stats.norm.rvs(true_priors.ix['log_c0','mu'],true_priors.ix['log_c0','std']),.5*true_priors.ix['log_c0','std'],size=len(time_points))
			log_d=scipy.stats.norm.rvs(scipy.stats.norm.rvs(true_priors.ix['log_d0','mu'],true_priors.ix['log_d0','std']),.5*true_priors.ix['log_d0','std'],size=len(time_points))
			pars=pd.DataFrame([log_a,log_b,log_c,log_d],columns=time_points,index=['log_a0','log_b0','log_c0','log_d0']).T
		else:
			pars=pd.Series(np.array([scipy.stats.norm.rvs(true_priors.ix[v,'mu'],true_priors.ix[v,'std']) for v in model_pars[model]]),index=model_pars[model])
			# add random up- or down-movement and subtract linear trend from baseline
			if 'alpha' in pars:
				pars['alpha']=np.random.choice([+1,-1])*pars['alpha']
				pars['log_a0']-=pars['alpha']*np.mean(time_points)
			if 'beta' in pars:
				pars['beta']=np.random.choice([+1,-1])*pars['beta']
				pars['log_b0']-=pars['beta']*np.mean(time_points)
			if 'gamma' in pars:
				pars['gamma']=np.random.choice([+1,-1])*pars['gamma']
				pars['log_c0']-=pars['gamma']*np.mean(time_points)
			if 'delta' in pars:
				pars['delta']=np.random.choice([+1,-1])*pars['delta']
				pars['log_d0']-=pars['delta']*np.mean(time_points)
			log_a,log_b,log_c,log_d=RNAkira.get_rates(time_points,pars)

		# now get random values for observations according to these rate parameters
		vals=[]

		for i,t in enumerate(time_points):

			mu=RNAkira.get_steady_state_values([log_a[i],log_b[i],log_c[i],log_d[i]],T,use_ribo=True)
			# use overdispersed gamma distribution (here: std = mu + 2^(-a) mu^(2-b))
			std=sigma_f*np.sqrt(mu+2**(-ma)*mu**(2-mb))

			vals.append([scipy.stats.gamma.rvs((mu[n]/std[n])**2,scale=std[n]**2/mu[n],size=len(replicates)) for n in range(ndim)])
			vals.append([np.ones(len(replicates))*std[n] for n in range(ndim)])

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

print >> sys.stderr, '[test_RNAkira] running RNAkira on simulated data'

results=RNAkira.RNAkira(values, T, sig_level=sig_level, min_TPM_ribo=0, maxlevel=options.maxlevel)

output=RNAkira.collect_results(results, time_points, sig_level=sig_level)

inferred_gene_class=output.ix[genes,'best_model']

genes_to_plot=genes[np.where(inferred_gene_class!=true_gene_class)[0]]
np.random.shuffle(genes_to_plot)

for k,gene in enumerate(genes_to_plot[:min(5,len(genes_to_plot))]):
	RNAkira.plot_data_rates_fits(time_points,replicates,values.ix[gene,'mean'],T,parameters[gene],results[gene],\
							   title='{0} (true: {1}, inferred: {2})'.format(gene,true_gene_class[gene],inferred_gene_class[gene]),\
							   priors=None,sig_level=sig_level)

mods=['0','a','b','c','ab','ac','ad','bc','bd','abc','abd','acd','bcd','abcd','all']
matches=np.array([[np.sum((true_gene_class==m1) & (inferred_gene_class==m2)) for m2 in mods] for m1 in mods])

nexact=np.sum(np.diag(matches))
nover=np.sum(np.triu(matches,1))
nlim=sum(true_gene_class!='all')
nunder=np.sum(np.tril(matches,-1))
ntarget=sum(true_gene_class!='0')

stats='[test_RNAkira] {0} exact hits ({1:.1f}%)\n{2} over-classifications ({3:.1f}%)\n{4} under-classifications ({5:.1f}%)'.format(nexact,100*nexact/float(nGenes),nover,100*nover/float(nlim),nunder,100*nunder/float(ntarget))
title='{0} genes, {1} time points, {2} replicates\n{3}'.format(nGenes,len(time_points),len(replicates),stats)
print >> sys.stderr, stats
		
fig=plt.figure(figsize=(5,5.5))
plt.imshow(np.log2(1+matches),origin='lower',cmap=plt.cm.Blues,vmin=0,vmax=np.log2(nGenes))
plt.xticks(range(len(mods)),mods)
plt.xlabel('inferred behavior')
plt.ylabel('true behavior')
plt.yticks(range(len(mods)),mods)
for i in range(len(mods)):
	for j in range(len(mods)):
		if matches[i,j] > 0:
			plt.text(j,i,matches[i,j],size=10,ha='center',va='center',color='k' if i==j else 'r')

plt.title(title,size=10)

