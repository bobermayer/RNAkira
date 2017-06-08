import os
import sys
import gzip
import re
import pandas as pd
from string import maketrans
from optparse import OptionParser
from twobitreader import TwoBitFile

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

def write_gene_lines (lines, outf, genome=None):

	""" take all non-transcript GTF lines for a specific gene, fix UTR annotation, add intron coordinates, and calculate length and number of U's in merged exonic and intronic sequences """

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
			# these lines should have been printed before
			raise Exception("this shouldn't happen!")
		elif ls[2]=='exon' and not "retained_intron" in ls[8]:
			# print exon lines and remember coordinates
			exon_lines.append(ls)
			exon_coords.append((int(ls[3]),int(ls[4])))
			outf.write(line)
		elif ls[2]=='CDS':
			# print CDS lines and remember coordinates
			CDS_coords.append((int(ls[3]),int(ls[4])))
			outf.write(line)
		elif ls[2]=='UTR':
			# remember UTR lines but do not print yet
			UTR_lines.append((int(ls[3]),int(ls[4]),ls[6],line))
		else: #if not "retained_intron" in ls[8]:
			# print everything else (except retained introns?)
			outf.write(line)

	if len(exon_lines)==0:
		return None

	if len(genes) > 1:
		raise Exception("more than one gene in this bunch of lines -- GTF not sorted?")
	if len(strands) > 1:
		raise Exception("more than one strand in this bunch of lines -- GTF not sorted?")
	if len(chroms) > 1:
		raise Exception("more than one chrom in this bunch of lines -- GTF not sorted?")
	
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

		# now print UTR lines and fix 5' and 3' annotation based on position relative to CDS
		for start,end,st,ll in UTR_lines:
			if st==strand and end <= min_CDS:
				UTR5_length+=end-start
				outf.write(re.sub('UTR','UTR5' if strand=='+' else 'UTR3',ll))
			elif st==strand and start >= max_CDS:
				UTR3_length+=end-start
				outf.write(re.sub('UTR','UTR3' if strand=='+' else 'UTR5',ll))
			else:
				outf.write(ll)

	# merge exon coords and add rest of GTF info from other lines
	merged_exon_coords=merge_intervals (exon_coords)
	merged_exon_info=exon_lines[0][0:8]+[';'.join(set.intersection(*[set(ls[8].strip(';').split(';')) for ls in exon_lines])).strip()]

	exon_length=sum(end-start for start,end in merged_exon_coords)

	# get sequence for exons and count U'
	if genome is not None:
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
		intron_start=merged_exon_coords[n][1]+1
		intron_end=merged_exon_coords[n+1][0]-1
		if intron_end > intron_start:
			ls=merged_exon_info
			outf.write('{0}\t{1}\tintron\t{2}\t{3}\t'.format(ls[0],ls[1],intron_start,intron_end)+'\t'.join(ls[5:])+'; intron_number {0};\n'.format(n+1))
			intron_length+=intron_end-intron_start
			if genome is not None:
				intron_seq+=genome[chrom][intron_start:intron_end]

	if genome is not None:
		if strand=='-':
			intron_seq=RC(intron_seq)
		intron_ucount=intron_seq.count('T')

	res=dict(gene_type=info['gene_type'],\
			 gene_name=info['gene_name'],\
			 exon_length=exon_length,\
			 intron_length=intron_length,\
			 CDS_length=CDS_length,\
			 UTR3_length=UTR3_length,\
			 UTR5_length=UTR5_length)

	if genome is not None:
		res['exon_ucount']=exon_ucount
		res['intron_ucount']=intron_ucount

	# collect info for this gene in a dictionary
	return (gene,res)
	

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
		genome=TwoBitFile(options.genome)

	gene_lines=[]
	gene_stats={}

	for line in inf:

		# simply print comment lines
		if line.startswith('#'):
			outf.write(line)
			continue

		ls=line.strip().split("\t")

		# if this line defines a gene, print lines for preceding gene and print this one
		if ls[2]=='gene':

			try:
				gene,stats=write_gene_lines(gene_lines, outf, genome if options.genome is not None else None)
				if gene not in gene_stats:
					gene_stats[gene]=stats
				else:
					raise Exception("more than one bunch of lines for "+gene)
			except:
				pass

			outf.write(line)
			gene_lines=[]

		elif ls[2]=='transcript':

			# if line defines a transcript, print it
			outf.write(line)

		else:

			# if line is not gene or transcript, add it to gene_lines
			gene_lines.append(line)

	try:
		gene,stats=write_gene_lines(gene_lines, outf, genome if options.genome is not None else None)
		if gene not in gene_stats:
			gene_stats[gene]=stats
		else:
			raise Exception("more than one bunch of lines for "+gene)
	except:
		pass

	outf.close()

	try:
		print >> sys.stderr, 'saving gene stats to '+options.stats
		gene_stats=pd.DataFrame.from_dict(gene_stats,orient='index')
		gene_stats.to_csv(options.stats)
	except:
		pass
