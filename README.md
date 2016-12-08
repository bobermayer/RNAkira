# RNAkira

RNAkira (RNA Kinetic Rate Analysis) is a tool to estimate synthesis, degradation, processing and translation rates 
using data from high-throughput sequencing of 4sU-labeled RNA (4sU-seq) and ribosome protected fragments (Ribo-seq). 
While it is conceptually related to other tools such as  [DRiLL](http://dx.doi.org/10.1016/j.cell.2014.11.015) or [INSPEcT](http://dx.doi.org/10.1093/bioinformatics/btv288), key differences are the inclusion of flowthrough data for normalization and the assumption of steady-state kinetics.

## Prerequisites
RNAkira runs on Python 2.7 and requires numpy, scipy, statsmodels and pandas (+ twobitreader if prepare_annotation.py is used). Read counts for exonic and intronic regions are expected in [featureCounts](http://bioinf.wehi.edu.au/featureCounts/) output format, but normalized TPM values can be supplied as well.

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
    -o RNAkira_output.csv 
    -i corrected_TPM.csv 
```

## Output
* TPM.csv -- a csv file with raw TPM values for each fraction and each sample
* corrected_TPM.csv -- a csv file with TPM values corrected for 4sU incorporation bias and with elu and flowthrough fractions normalized by linear regression (see, e.g., [Schwannhäuser et al. Nature 2011](http://dx.doi.org/10.1038/nature10098))
* RNAkira_output.csv -- a csv file with fit results and the following columns
  * gene_id
  * initial_synthesis_t1 -- synthesis rate from initial fit
  * initial_synthesis_err_t1 -- with associated error
  * ... for each time point
  * initial_degradation_t1 -- degradation rate
  * ... + error for each time point
  * initial_processing_t1 -- processing rate
  * ... + error for each time point
  * initial_translation_t1 -- translation rate
  * ... + error for each time point
  * initial_logL -- log likelihood of initial fit
  * initial_fit_success -- status of minimization routine
  * initial_pval -- chi-squared p-value comparing initial fit to best model
  * initial_qval -- BH-corrected p-value
