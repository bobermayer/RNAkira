# RNAkira

RNAkira (RNA Kinetic Rate Analysis) is a tool to estimate synthesis, degradation, processing and translation rates 
using data from high-throughput sequencing of 4sU-labeled RNA (4sU-seq) and ribosome protected fragments (RPFs from Ribo-seq). 
It is conceptually related to other tools such as  [DRiLL](http://dx.doi.org/10.1016/j.cell.2014.11.015) or [INSPEcT](http://dx.doi.org/10.1093/bioinformatics/btv288), but key differences are the inclusion of flowthrough for normalization, ribo-seq data for translation rate estimates, and the assumption of steady-state kinetics.

## Prerequisites
RNAkira runs on Python 2.7 and requires numpy, scipy, statsmodels and pandas (+ twobitreader if prepare_annotation.py is used). Read counts for exonic and intronic regions are expected in [featureCounts](http://bioinf.wehi.edu.au/featureCounts/) output format, but normalized TPM values can be supplied as well.

## Description
The tool assumes standard RNA kinetics: precursor RNA *P* is born with synthesis rate *a* and destroyed with processing rate *c*, mature RNA *M* is produced by processing a precursor, translated to ribo *R* with rate *d* and destroyed with degradation rate *b*. 

Precursor RNA is estimated from intronic RNA read counts, mature from exonic RNA read counts, and ribo from CDS RPF counts. For RNA, reads come in three fractions: newly synthesized (=elu), pre-existing (=flowthrough) and total (=unlabeled). TPM values are calculated for each sample, elu values are corrected for 4sU incorporation efficiency, and elu and flowthrough samples normalized using linear regression (see, e.g., [Schwannhäuser et al. Nature 2011](http://dx.doi.org/10.1038/nature10098)). RNAkira initially fits the steady-state solutions to the above kinetics using maximum likelihood with empirical Bayes priors estimated across genes at each time point separately, and then performs model selection, starting with constant rates for the different time points and successively allowing additional linear changes to the (log) rates (= log fold changes) in a hierarchy of models (similar to INSPEcT). Models at different levels are compared and best models are selected using an FDR cutoff of alpha (default: 5%).

## Usage

### 1. prepare annotation
The script ``prepare_annotation.py`` takes a Gencode gtf file and adds features for introns. It also counts T occurrences in exonic regions (genome must be supplied as 2bit file) to correct for 4sU incorporation bias, and outputs a csv file with gene stats (length of various regions, gene types, exonic and intronic T counts)
```
prepare_annotation.py -i annotation.gtf.gz -o annotation_with_introns.gtf -s gene_stats.csv -g genome.2bit
```
### 2. run featureCounts on your data
RNAkira assumes that you have 4 datasets for pre-existing (flowthrough), newly synthesized (elu), unlabeled RNA (unlabeled), and ribosome protected fragments (ribo) for each timepoint. The associated bam files will be quantified over exonic and intronic regions one fraction at a time:
```
featureCounts -t exon -g gene_id -a annotation_with_introns.gtf -o elu_counts_exons.txt elu_bam_t1_rep1.bam elu_bam_t1_rep2.bam elu_bam_t2_rep1.bam elu_bam_t2_rep2.bam
```
and similar for the flowthrough, unlabeled and ribo fractions (use ``-t intron`` for intronic reads and ``-t CDS`` for ribo-seq data; use ``-s`` and ``-p`` appropriately if you have stranded or paired-end data). 
**Note**: use the same ordering of bam files for all fractions!

### 3. run RNAkira
assuming a labeling time T (which should be much smaller than the difference between any two time points for the steady-state assumption to hold) and time points t1,t2,t3 in duplicates, the tool is called as follows
```
python RNAkira.py 
    -T T 
    -t t1,t1,t2,t2,t3,t3 
    -o RNAkira_output.csv 
    -a 0.05
    -g gene_stats.csv 
    -e elu_counts_introns.txt 
    -E elu_counts_exons.txt 
    -f flowthrough_counts_introns.txt 
    -F flowthrough_counts_exons.txt 
    -r ribo_counts_CDS.txt 
    -u unlabeled_counts_introns.txt 
    -U unlabeled_counts_exons.txt  
```
**Note**: for n time points in k replicates, the last n\*k columns of **each** of the featureCounts output files have to correspond exactly to the n\*k time points given as arguments to ``-t``

Alternatively, if you have TPM values corrected for 4sU incorporation bias and with elu and flowthrough fractions properly normalized (e.g., the ``corrected_TPM.csv`` output of a previous RNAkira run on the same data), you can use
```
python RNAkira.py 
    -T T 
    -t t1,t1,t2,t2,t3,t3 
    -a 0.05
    -o RNAkira_output.csv 
    -i corrected_TPM.csv 
```

## Output
* TPM.csv -- a csv file with raw TPM values for each fraction and each sample
* corrected_TPM.csv -- a csv file with TPM values corrected for 4sU incorporation bias and with elu and flowthrough fractions normalized 
* RNAkira_output.csv -- a csv file with fit results for each gene: synthesis, degradation, processing and translation rates (+ error estimates) for each time point from the **initial fit**, together with the log-likelihood of this model, fit success (boolean), and a p- and q-value from comparing to the best model; then rates + errors for each time point for the **best model**, followed by estimated log2 fold changes, the resulting log-likelihood and the fit success, and finally p- and q-value from comparing the best to the next-best model
