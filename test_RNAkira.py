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
true_priors=pd.DataFrame(dict(mu=np.array([2,-4,-.7,.0,.02,.02,.02,.02]),\
							  std=np.array([2,1.6,1,.7,.01,.005,.005,.005])),\
						 index=['log_a0','log_b0','log_c0','log_d0','alpha','beta','gamma','delta'])

# distribute models over genes, make sure most genes don't change for multiple testing correction to work (other model types than MPR don't really work here)
true_gene_class=['MPR_0']*2000+\
	['MPR_a']*50+\
	['MPR_b']*50+\
	['MPR_c']*50+\
	['MPR_d']*50+\
	['MPR_ab']*20+\
	['MPR_ac']*20+\
	['MPR_ad']*20+\
	['MPR_bc']*20+\
	['MPR_bd']*20+\
	['MPR_cd']*20+\
	['MPR_abc']*15+\
	['MPR_acd']*15+\
	['MPR_abd']*15+\
	['MPR_bcd']*15+\
	['MPR_abcd']*20+\
	['MPR_all']*100

# define time points
time_points=['0','12','24','36','48']
times=map(float,time_points)
# define number of replicates
replicates=map(str,range(10))
# labeling time
T=1
# parameters of dispersion curve
slope,intercept=0.01,5

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
parser.add_option('-i','--values',dest='values',help="file with simulated counts (created by test_RNAkira.py)")
parser.add_option('-o','--outf',dest='outf',help="save simulated counts to this file")
parser.add_option('-T','--labeling_time',dest='T',help="labeling time [1]", default=1)
parser.add_option('','--maxlevel',dest='maxlevel',help="max level to test [5]",default=5,type=int)
parser.add_option('','--alpha',dest='alpha',help="FDR cutoff [0.05]",default=0.05)
parser.add_option('','--statsmodel',dest='statsmodel',help="statsmodel [gaussian]",default='gaussian')

options,args=parser.parse_args()

sig_level=float(options.alpha)
T=float(options.T)

if options.values is not None:

	print >> sys.stderr, '\n[test_RNAkira] reading simulated data from '+options.values
	counts=pd.read_csv(options.values,index_col=0,header=range(3))
	counts.columns=pd.MultiIndex.from_tuples([(c[0],c[1],int(c[2])) for c in values.columns.tolist()])

	time_points=np.unique(zip(*counts.columns.tolist())[2])
	replicates=np.unique(zip(*counts.columns.tolist())[3])
	cols=np.unique(zip(*counts.columns.tolist())[1])
	genes=counts.index.counts
	nGenes=len(genes)
	true_gene_class=pd.Series(np.array(['_'.join(g.split('_')[2:]) for g in genes]),index=genes)
	parameters_known=False

else:
	
	cols=['elu-mature','flowthrough-mature','unlabeled-mature','elu-precursor','flowthrough-precursor','unlabeled-precursor','ribo']

	nGenes=len(true_gene_class)
	genes=np.array(map(lambda x: '_'.join(x), zip(['gene']*nGenes,map(str,range(nGenes)),true_gene_class)))

	true_gene_class=pd.Series(true_gene_class,index=genes)

	parameters_known=True
	parameters={}
	counts={}
	#diff={}

	print >> sys.stderr, '\n[test_RNAkira] drawing parameters and observations for {0} genes ({1} time points, {2} replicates)'.format(nGenes,len(time_points),len(replicates))
	print >> sys.stderr, '[test_RNAkira] true priors for log_a: {0:.2g}/{1:.2g}, log_b: {2:.2g}/{3:.2g}, log_c: {4:.2g}/{5:.2g}, log_d: {6:.2g}/{7:.2g}'.format(*true_priors.ix[:4].values.flatten())

	for ng,gene in enumerate(genes):

		print >> sys.stderr, '[test_RNAkira] {0} genes initialized\r'.format(ng+1),

		model=true_gene_class[gene]
		if model=='MPR_all':
			rates=[scipy.stats.norm.rvs(scipy.stats.norm.rvs(true_priors.ix[v,'mu'],true_priors.ix[v,'std']),\
										.5*true_priors.ix[v,'std'],size=len(time_points)) for v in model_pars[model]]
			pars=pd.DataFrame(rates,columns=time_points,index=['log_a0','log_b0','log_c0','log_d0']).T
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
		#stddev=pd.Series(index=pd.MultiIndex.from_product([cols,time_points]))
		#disp=pd.Series(index=pd.MultiIndex.from_product([cols,time_points]))

		for i,t in enumerate(time_points):

			mu=1.e-8+RNAkira.get_steady_state_values([r[i] for r in rates],T,use_precursor='P' in model, use_ribo='R' in model)

			d=(intercept-1)/mu+slope

			for n in range(len(mu)):
				cnts[cols[n],t]=scipy.stats.nbinom.rvs(1./d[n],1./(1.+d[n]*mu[n]),size=len(replicates))
				#disp[cols[n],t]=d[n]
				#stddev[cols[n],t]=np.sqrt(mu[n]*(1.+d[n]*mu[n]))

		parameters[gene]=pars
		counts[gene]=cnts

		if False:

			cnts_here=cnts.unstack(level=0)[cols].stack().values.reshape((len(time_points),len(replicates),len(cols)))
			stddev_here=stddev.unstack(level=0)[cols].stack().values.reshape((len(time_points),len(cols)))
			disp_here=disp.unstack(level=0)[cols].stack().values.reshape((len(time_points),len(cols)))
			nf_here=np.ones_like(cnts_here)

			res_true=pd.DataFrame(rates,index=['log_a0','log_b0','log_c0','log_d0'],columns=time_points).T
			res_gaussian=RNAkira.fit_model(cnts_here,stddev_here,nf_here,T,time_points,true_priors,None,'MPR_all',model_pars['MPR_all'],'gaussian',min_args)
			res_nbinom=RNAkira.fit_model(cnts_here,disp_here,nf_here,T,time_points,true_priors,None,'MPR_all',model_pars['MPR_all'],'nbinom',min_args)

			diff[gene]=dict(gaussian=((res_true-res_gaussian['est_pars'])**2).mean().mean(),nbinom=((res_true-res_nbinom['est_pars'])**2).mean().mean())

	print >> sys.stderr, ''

	counts=pd.DataFrame.from_dict(counts,orient='index')
	#diff=pd.DataFrame.from_dict(diff,orient='index')
	
	if options.outf is not None:
		print >> sys.stderr, '[test_RNAkira] saving to '+options.outf
		counts.to_csv(options.outf)

# all genes have the same length
LF=pd.Series(1,index=counts.index)
# normalize by "sequencing depth"
elu_flowthrough_factor=(counts['elu-mature'].add(counts['elu-precursor'],fill_value=0).sum(axis=0)+\
						counts['flowthrough-mature'].add(counts['flowthrough-precursor'],fill_value=0).sum(axis=0))/1.e6
unlabeled_factor=counts['unlabeled-mature'].add(counts['unlabeled-precursor'],fill_value=0).sum(axis=0)/1.e6
ribo_factor=counts['ribo'].sum(axis=0)/1.e6
# one global size factor to strictly keep relative proportions
SF=pd.concat([elu_flowthrough_factor,elu_flowthrough_factor,unlabeled_factor,\
			  elu_flowthrough_factor,elu_flowthrough_factor,unlabeled_factor,\
			  ribo_factor],axis=0,keys=cols)

CF=pd.DataFrame(1,index=counts.index,columns=counts.columns)
NF=CF.divide(LF,axis=0,level=0).divide(SF,axis=1).fillna(1)
NF1=pd.DataFrame(1,index=NF.index,columns=NF.columns)

if options.statsmodel=='gaussian':
	TPM=counts.multiply(NF)
	stddev=RNAkira.estimate_stddev (TPM)
	#results=RNAkira.RNAkira(counts, stddev, NF, T, sig_level=sig_level, min_ribo=.1, min_precursor=.1, maxlevel=options.maxlevel, statsmodel=options.statsmodel)
	results=RNAkira.RNAkira(TPM, stddev, NF1, T, sig_level=sig_level, min_ribo=.1, min_precursor=.1, maxlevel=options.maxlevel, statsmodel=options.statsmodel)
else:
	disp=RNAkira.estimate_dispersion (counts*NF.divide(np.exp(np.log(NF).mean(axis=1)),axis=0))
	results=RNAkira.RNAkira(counts, disp, NF, T, sig_level=sig_level, min_ribo=.1, min_precursor=.1, maxlevel=options.maxlevel, statsmodel=options.statsmodel)

output=RNAkira.collect_results(results, time_points, sig_level=sig_level)

print >> sys.stderr, '\n[test_RNAkira] evaluating performance'

inferred_gene_class=output.ix[genes,'best_model']
inferred_gene_class[output.ix[genes,'initial_qval'] < .05]=output.ix[genes,'initial_model'][output.ix[genes,'initial_qval'] < .05]

# use this if you want to plot specific examples
if False:

	genes_to_plot=genes[np.where(inferred_gene_class!=true_gene_class)[0]]
	np.random.shuffle(genes_to_plot)

	for k,gene in enumerate(genes_to_plot[:min(5,len(genes_to_plot))]):
		pcorr=pd.Series(dict(log_a0=np.log(SF.mean()*LF.mean()),log_b0=0,log_c0=0,log_d0=0))
		RNAkira.plot_data_rates_fits(time_points,replicates,TPM.ix[gene],T,\
									 parameters[gene]-pcorr if parameters_known else None,\
									 results[gene],\
									 'P' in inferred_gene_class[gene],\
									 'R' in inferred_gene_class[gene],\
									 title='{0} (true: {1}, inferred: {2})'.format(gene,true_gene_class[gene],inferred_gene_class[gene]),\
									 priors=None,sig_level=sig_level)

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

pdf=pd.DataFrame([pd.DataFrame(RNAkira.get_rates(time_points,parameters[gene]),columns=time_points,index=['initial_synthesis','initial_degradation','initial_processing','initial_translation']).stack() for gene in genes],index=genes)
pdf['initial_synthesis']-=np.log(SF.mean()*LF.mean())
pdf.columns=[c[0]+'_t'+c[1] for c in pdf.columns.tolist()]
diff=np.abs(np.log(output.ix[:,:len(time_points)*4])-pdf)

fig=plt.figure(figsize=(5,4))
fig.clf()

ax=fig.add_axes([.15,.15,.75,.75])
ax.hist([np.log10(diff.ix[:,n*len(time_points):(n+1)*len(time_points)].dropna().values.flatten()) for n in range(4)],bins=np.arange(-5,2,.1),histtype='step',log=True,label=['synthesis','degradation','processing','translation'])
ax.set_xlabel('log10 rel. error')
ax.set_ylabel('counts')
ax.legend(loc=2)
ax.set_title('{0} genes, {1} time points, {2} replicates'.format(nGenes,len(time_points),len(replicates)),size=10)
