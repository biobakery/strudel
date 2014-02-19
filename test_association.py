#!/usr/bin/env python 

import strudel, halla, numpy 
import sys 
import multiprocessing 
import argparse
import subprocess


#try:
#	multiprocessing.Pool
#	bPP = True 
#except Exception:
#	bPP = False 

bPP = False

##global parameters 
c_num_cores = 8 

def _main( strFile, iRow, iCol, strMethod, iIter, fSparsity, fNoise, strSpike, strBase, bParam, iPval ):

		strTitle = strMethod + "_" + strBase + "_" + strSpike + "_" + str(iRow) + "x" + str(iCol) + "_" + ("parametric" if bParam else "nonparametric") + "_" + ("pval" if iPval ==1 else "association")
		strFile = strFile or (strTitle + ".png") 
	
	try:
		s = strudel.Strudel()
		s.set_base(strBase)
		s.set_noise(fNoise)

		##Generate synthetic data 
		X = s.randmat( shape = (iRow, iCol) )
		Y,A = s.spike_synthetic_data( X, sparsity = fSparsity, spike_method = strSpike )

		##Run iIterations of pipeline 
		A_emp = [] 

		if bPP:
			pool = multiprocessing.Pool(processes = iIter )
			result = pool.map( lambda x: s.association(x[0],x[1],strMethod = x[2]), [[X,Y,"halla"]]*3 )

		else:
			for i in range(iIter):
			    print "Running iteration " + str(i)
			    aOut = s.association( X,Y, strMethod = strMethod, bParam = bParam, bPval = iPval )
			    A_emp.append(aOut)

		##Set meta objects 
		A_emp_flatten = None 
		if iPval == -1:
			A_emp_flatten = [numpy.reshape( numpy.abs(a), iRow**2 ) for a in A_emp] 
		elif iPval == 1:
			## Remember, associations are _always_ notion of strength, not closeness; this is an invariant I will strictly enforce 
			A_emp_flatten = [1.0 - numpy.reshape( numpy.abs(a), iRow**2 ) for a in A_emp] 
		
		A_flatten = [numpy.reshape( A, iRow**2 ) for _ in range(iIter)]

		##Generate roc curves 
		aROC = s.roc(A_flatten, A_emp_flatten, astrLabel = ["run " + str(i) for i in range(iIter)],
			strTitle = strTitle, strFile = strFile )
		
		print "ROC values:"
		print aROC 

		return aROC 

	except Exception:
		return subprocess.call( ["touch",strFile] )
		

argp = argparse.ArgumentParser( prog = "test_associations.py",
        description = "Test different types of associations in strudel + halla." )

### BUGBUG: can we run on real files?  
#argp.add_argument( "istm",              metavar = "input.txt",
#        type = argparse.FileType( "r" ),        default = sys.stdin,    nargs = "+",
#        help = "Tab-delimited text input file, one row per feature, one column per measurement" )

argp.add_argument( "-o",                dest = "strFile",                  metavar = "output_plot",
       type = str,        default = None,
        help = "Optional output file name for script-generated plot" )

argp.add_argument( "-m",                dest = "strMethod",             metavar = "method_name",
        type = str,   default = "pearson",
        help = "Association method. [pearson, spearman, mi, norm_mi, kw, x2, halla]" )

argp.add_argument( "--row",                dest = "iRow",             metavar = "num_rows",
        type = int,   default = "20",
        help = "Number of rows" )

argp.add_argument( "--col",                dest = "iCol",             metavar = "num_cols",
        type = int,   default = "20",
        help = "Number of columns" )

argp.add_argument( "--parametric",                dest = "bParam",
        action = "store_true",
        help = "Parametric pvalue generation? else permutation based error bar generation. The only ones with parametric error bars are pearson, spearman, x2" )

argp.add_argument( "-i",                dest = "iIter",             metavar = "num_iteration",
        type = int,   default = "3",
        help = "Number of iterations for each association method" )

argp.add_argument( "-s",                dest = "fSparsity",             metavar = "sparsity",
        type = float,   default = "0.5",
        help = "Sparsity parameter, value in [0.0,1.0]" )

argp.add_argument( "-n",                dest = "fNoise",             metavar = "noise",
        type = float,   default = "0.1",
        help = "Noise parameter, value in [0.0,1.0]" )

argp.add_argument( "--spike_method",                dest = "strSpike",             metavar = "spike_method",
        type = str,   default = "parabola",
        help = "Spike method: [linear, vee, sine, parabola, cubic, log, half_circle]" )

argp.add_argument( "-b",                dest = "strBase",             metavar = "base_distribution",
        type = str,   default = "normal",
        help = "Base distribution: [normal, uniform]" )

argp.add_argument( "-p",                dest = "iPval",             metavar = "use_pval",
        type = int,   default = -1,
        help = "Parameter to request for association values or p-values. -1 -> only association value, 0 -> both association and p-value (do not use for now), 1 -> only p-value" )


args = argp.parse_args( ) 
_main( args.strFile, args.iRow, args.iCol, args.strMethod, args.iIter, args.fSparsity, args.fNoise, args.strSpike, args.strBase, args.bParam, args.iPval )
