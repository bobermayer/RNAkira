import os
import sys
import gzip
import re
import pandas as pd
from string import maketrans
from collections import defaultdict
from optparse import OptionParser

def RC (s):
    """ reverse complement a sequence """
    rc_tab=maketrans('ACGTUNacgtun','TGCAANtgcaan')
    return s.translate(rc_tab)[::-1]

def merge_intervals (intervals):

    """ interval merging function from here: http://codereview.stackexchange.com/questions/69242/merging-overlapping-intervals """

    sorted_by_lower_bound = sorted(intervals, key=lambda tup: tup[0])
    merged = []

    for higher in sorted_by_lower_bound:
        if not merged:
            merged.append(higher)
        else:
            lower = merged[-1]
            # test for intersection between lower and higher:
            # we know via sorting that lower[0] <= higher[0]
            if higher[0] <= lower[1]:
                upper_bound = max(lower[1], higher[1])
                merged[-1] = (lower[0], upper_bound)  # replace by merged interval
            else:
                merged.append(higher)

    return merged

def fix_lines  (gene, lines, outf, genome=None):

    """ take all GTF lines for a specific gene, fix UTR annotation, add intron coordinates, \
    and calculate length and number of U's in merged exonic and intronic sequences """

    genes=set()
    chroms=set()
    strands=set()
    exon_coords=[]
    exon_lines=[]
    CDS_coords=[]
    UTR_lines=[]

    for line in lines:
        
        ls=line.strip().split('\t')
        info=dict((x.split()[0].strip(),x.split()[1].strip().strip('"')) for x in ls[8].strip(';').split(";"))

        genes.add(info['gene_id'])
        chroms.add(ls[0])
        strands.add(ls[6])

        if ls[2] in ['gene','transcript']:
            outf.write(line)
        elif ls[2]=='exon' and not "retained_intron" in ls[8]:
            # print exon lines and remember coordinates
            exon_lines.append(ls)
            exon_coords.append((int(ls[3])-1,int(ls[4])))
            outf.write(line)
        elif ls[2]=='CDS':
            # print CDS lines and remember coordinates
            CDS_coords.append((int(ls[3])-1,int(ls[4])))
            outf.write(line)
        elif ls[2] in ['UTR','three_prime_utr','five_prime_utr']:
            # remember UTR lines but do not print yet
            UTR_lines.append((int(ls[3])-1,int(ls[4]),line))
        else: #if not "retained_intron" in ls[8]:
            # print everything else (except retained introns?)
            outf.write(line)

    if len(exon_lines)==0:
        return dict(gene_type=info['gene_type' if 'gene_type' in info else 'gene_biotype'],\
                        gene_name=info['gene_name'],\
                        exon_length=0,\
                        intron_length=0,\
                        CDS_length=0,\
                        UTR3_length=0,\
                        UTR5_length=0)

    if len(genes) > 1 or len(strands) > 1 or len(chroms) > 1:
        raise Exception("more than one gene, strand or chrom in this bunch of lines -- GTF not sorted?")
    
    gene=genes.pop()
    chrom=chroms.pop()
    strand=strands.pop()

    UTR3_length=0
    UTR5_length=0
    CDS_length=0

    if len(CDS_coords) > 0:

        # merge CDS coords to determine start and stop

        merged_CDS_coords=merge_intervals (CDS_coords)
        min_CDS=min(start for start,_ in merged_CDS_coords)
        max_CDS=max(end for _,end in merged_CDS_coords)

        CDS_length=sum(end-start for start,end in merged_CDS_coords)

        UTR3_coords=[]
        UTR5_coords=[]

        # now print UTR lines and fix 5' and 3' annotation based on position relative to CDS
        for start,end,ll in UTR_lines:
            if end <= min_CDS:
                if strand=='+':
                    UTR5_coords.append((start,end))
                else:
                    UTR3_coords.append((start,end))
                outf.write(re.sub('UTR','UTR5' if strand=='+' else 'UTR3',ll))
            elif start >= max_CDS:
                if strand=='+':
                    UTR3_coords.append((start,end))
                else:
                    UTR5_coords.append((start,end))
                outf.write(re.sub('UTR','UTR3' if strand=='+' else 'UTR5',ll))
            else:
                outf.write(ll)

        # calculate total UTR3/UTR5 length based on merged exons
        if len(UTR3_coords) > 0:
            UTR3_length=sum(end-start for start,end in merge_intervals(UTR3_coords))
        if len(UTR5_coords) > 0:
            UTR5_length=sum(end-start for start,end in merge_intervals(UTR5_coords))

    # merge exon coords and add rest of GTF info from other lines
    merged_exon_coords=merge_intervals (exon_coords)
    merged_exon_info=exon_lines[0][0:8]+[';'.join(set.intersection(*[set(ls[8].strip(';').split(';')) for ls in exon_lines])).strip()]

    exon_length=sum(end-start for start,end in merged_exon_coords)

    # get sequence for exons and count U'
    if genome is not None and chrom in genome:
        exon_seq=''.join(genome[chrom][start:end] for start,end in merged_exon_coords)
        if strand=='-':
            exon_seq=RC(exon_seq)
        exon_ucount=exon_seq.count('T')
        if len(exon_seq) != exon_length:
            raise Exception("this shouldn't happen!")

    # print lines for introns and get their sequence
    nintrons=len(merged_exon_coords)-1
    intron_seq=''
    intron_length=0
    for n in range(nintrons):
        intron_start=merged_exon_coords[n][1]
        intron_end=merged_exon_coords[n+1][0]
        if intron_end > intron_start:
            ls=merged_exon_info
            outf.write('{0}\t{1}\tintron\t{2}\t{3}\t'.format(ls[0],ls[1],intron_start+1,intron_end)+\
                           '\t'.join(ls[5:])+'; intron_number {0};\n'.format(n+1))
            intron_length+=intron_end-intron_start
            if genome is not None and chrom in genome:
                intron_seq+=genome[chrom][intron_start:intron_end]

    if genome is not None and chrom in genome:
        if strand=='-':
            intron_seq=RC(intron_seq)
        intron_ucount=intron_seq.count('T')

    stats=dict(gene_type=info['gene_type' if 'gene_type' in info else 'gene_biotype'],\
                   gene_name=info['gene_name'],\
                   exon_length=exon_length,\
                   intron_length=intron_length,\
                   CDS_length=CDS_length,\
                   UTR3_length=UTR3_length,\
                   UTR5_length=UTR5_length)

    if genome is not None and chrom in genome:
        stats['exon_ucount']=exon_ucount
        stats['intron_ucount']=intron_ucount

    # collect info for this gene in a dictionary
    return stats

if __name__ == '__main__':

    parser=OptionParser()
    parser.add_option('-i','--infile',dest='infile',help="input GTF (default: stdin), should be sorted so that all lines for one gene are together")
    parser.add_option('-o','--outfile',dest='outfile',help="output GTF (default: stdout)")
    parser.add_option('-s','--stats',dest='stats',help="if given, write gene stats to here")
    parser.add_option('-g','--genome',dest='genome',help="if exonic and intronic U's should be counted, give genome in 2bit format")

    options,args=parser.parse_args()

    if options.infile is None:
        print >> sys.stderr, 'reading from stdin'
        inf=sys.stdin
    else:
        print >> sys.stderr, 'reading from '+options.infile
        if options.infile.endswith('.gz'):
            inf=gzip.open(options.infile)
        else:
            inf=open(options.infile)

    gene_lines=defaultdict(list)
    for line in inf:

        if line.startswith('#'):
            continue

        ls=line.strip().split("\t")
        info=dict((x.split()[0].strip(),x.split()[1].strip().strip('"')) for x in ls[8].strip(';').split(";"))
        name=info['gene_id']
        gene_lines[name].append(line)

    if options.outfile is None:
        print >> sys.stderr, 'writing to stdout'
        outf=sys.stdout
    else:
        print >> sys.stderr, 'writing to '+options.outfile
        if options.outfile.endswith('.gz'):
            outf=gzip.open(options.outfile,'wb')
        else:
            outf=open(options.outfile,'w')

    if options.genome is not None:
        print >> sys.stderr, 'using genome '+options.genome
        if options.genome.endswith('.2bit'):
            from twobitreader import TwoBitFile
            genome=TwoBitFile(options.genome)
        else:
            from pysam import FastaFile
            genome=FastaFile(options.genome)
    else:
        genome=None

    gene_stats={}
    for gene,lines in gene_lines.iteritems():
        gene_stats[gene]=fix_lines(gene,lines,outf,genome=genome)

    outf.close()

    print >> sys.stderr, 'saving gene stats to '+options.stats
    gene_stats=pd.DataFrame.from_dict(gene_stats,orient='index')
    gene_stats.to_csv(options.stats)
