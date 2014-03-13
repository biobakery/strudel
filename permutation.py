import strudel
import argparse 
import sys 
import numpy, scipy

import multiprocessing 
from multiprocessing import Process, Queue
from time import gmtime, strftime
import random

def _permutation( X, Y, strMultiProcessing ,strVerboseOutput, strAssociation, iIter    ):
	def __invariance( aOut ):
 
			"""
			Enforce invariance: when nonparametric pvalue generation is asked, 
			parametric pval should not be outputted.  
			"""
			try:
				aOut[1]
				return aOut[0] 
			except Exception:
				return aOut 

	def __permute( X, Y, pAssociation ):
		"""
		Give value of pAssociation on one instance of permutation 
		"""
		sys.stdout.write("Iteration number: " + str(ix+1) + "  - " + strftime("%a, %d %b %Y %H:%M:%S ", gmtime()) +"\n")
		aPermX = numpy.random.permutation( X )##without loss of generality, permute X and not Y
		return __invariance( pAssociation( aPermX, Y ) )
 
	X,Y = X.flatten(), Y[0].flatten() #* <--------------  The original code was using Y not Y[0], so when used rows cols parm gave error

	s = strudel.Strudel( )
	pAssociation = s.hash_association_method[strAssociation]
	fAssociation = __invariance( pAssociation( X,Y ) )

	#***********************************************************************************
	#*  Calculate permutation with multi processing or iteration depending on flag     *
	#***********************************************************************************
 	if  strMultiProcessing == "N":
		aDist = [__permute(X,Y, pAssociation=pAssociation) for ix in range(iIter)] ##array containing finite estimation of sampling distribution 
	else:
		aDist = Calc_Permutations_Using_Multiprocessing(X,Y,  iIter, pAssociation, strVerboseOutput)       #* Perform calculations of permutations using multiprocessing	


	fPercentile = scipy.stats.percentileofscore( aDist, fAssociation, kind="strict" ) ##source: Good 2000 
	### \frac{ \sharp\{\rho(\hat{X},Y) \geq \rho(X,Y) \} +1  }{ k + 1 }
	### k number of iterations, \hat{X} is randomized version of X 
	### PercentileofScore function ('strict') is essentially calculating the additive inverse (1-x) of the wanted quantity above 
	### consult scipy documentation at: http://docs.scipy.org/doc/scipy-0.7.x/reference/generated/scipy.stats.percentileofscore.html

	fP = ((1.0-fPercentile/100.0)*iIter + 1)/(iIter+1)
	return fAssociation, fP 
	


	
	
#********************************************************************************
#*  Calculate the permutations using Multiprocessing                            *
#********************************************************************************
def Calc_Permutations_Using_Multiprocessing(X,Y,   iIter, pAssociation,  strVerboseOutput):
	aDist = list()
	procs = []
 	out_q = multiprocessing.Queue()
  
		
	for i in range(iIter):
		if   strVerboseOutput == "YY": #  If requested verbose output	
			sys.stderr.write("Starting Process  " + str(i+1) + "  - " + strftime("%a, %d %b %Y %H:%M:%S ", gmtime())+ "\n")
		
	 
		p = multiprocessing.Process(
			target=calc_permutation_worker,
			args = (X,Y, pAssociation, strVerboseOutput,  out_q))
		procs.append(p)
		p.start()

	# Collect all results  

	
	for i in range(iIter):
		if   strVerboseOutput == "YY": #  If requested super verbose output
			sys.stderr.write("Collecting results from  Process #" + str(i+1) + "  - " + strftime("%a, %d %b %Y %H:%M:%S ", gmtime()) +"\n")
		aOut = out_q.get(['block', None])
		aDist.append(aOut)

		
	if   strVerboseOutput == "YY": #  If requested verbose output	
		print "adist=",aDist
	# Wait for all worker processes to finish
	i = 0
	for p in procs:
		i+=1
		if   strVerboseOutput == "YY": #  If requested verbose output
			sys.stderr.write("Joining  Process " + str(i)+ "  - " + strftime("%a, %d %b %Y %H:%M:%S ", gmtime()) + "\n")
		p.join()
 	return  aDist


#********************************************************************************
#*  Calculate the associations worker                                           *
#********************************************************************************
def  calc_permutation_worker(X,Y, pAssociation,strVerboseOutput,  out_q):
	numpy.random.seed()							#So that each worker provides different result
 	aPermX = numpy.random.permutation( X )             ##without loss of generality, permute X and not Y
 	aOut = pAssociation( aPermX, Y )
 	CalcResult = aOut
	if len(aOut) > 1:
		CalcResult = aOut[0]
  	out_q.put(CalcResult)
	return 0
	
	
	 
	
	
	
	
	
	

def _main( strBase, fNoise, iRow, iCol, fSparsity, strSpike, strMethod, iIter, fPermutationNoise,  strMultiProcessing, strVerboseOutput  ):
	s = strudel.Strudel()
	s.set_base(strBase)
	s.set_noise(fNoise)
	s.set_permutation_noise(fPermutationNoise)
	#s.set_sparsity_param( fSparsity )

	##Generate synthetic data 
 
	X = s.randmat( shape = (iRow, iCol) )
	Y = s.spike( X, sparsity = fSparsity, strMethod = strSpike )

	return _permutation( X, Y , strMultiProcessing,strVerboseOutput, strAssociation = strMethod, iIter = iIter  )

if __name__ == "__main__":

	argp = argparse.ArgumentParser( prog = "permutation.py",
	        description = "Test different types of permutation methods in strudel + halla." )

	argp.add_argument( "-m",                dest = "strMethod",             metavar = "method_name",
	        type = str,   default = "pearson",
	        help = "Association method. [pearson, spearman, mi, norm_mi, kw, x2, halla]" )

	argp.add_argument( "--row",                dest = "iRow",             metavar = "num_rows",
	        type = int,   default = "1",
	        help = "Number of rows" )

	argp.add_argument( "--col",                dest = "iCol",             metavar = "num_cols",
	        type = int,   default = "20",
	        help = "Number of columns" )

	argp.add_argument( "-i",                dest = "iIter",             metavar = "num_iteration",
	        type = int,   default = "3",
	        help = "Number of iterations for each association method" )

	argp.add_argument( "-s",                dest = "fSparsity",             metavar = "sparsity",
	        type = float,   default = "1.0",
	        help = "Sparsity parameter, value in [0.0,1.0]" )

	argp.add_argument( "-n",                dest = "fNoise",             metavar = "noise",
	        type = float,   default = "0.1",
	        help = "Noise parameter, value in [0.0,1.0]" )

	argp.add_argument( "--permutation_noise",                dest = "fPermutationNoise",             metavar = "permutation_noise",
	        type = float,   default = "0.1",
	        help = "Permutation noise parameter, value in [0.0,1.0]" )

	argp.add_argument( "--spike_method",                dest = "strSpike",             metavar = "spike_method",
	        type = str,   default = "parabola",
	        help = "Spike method: [linear, vee, sine, parabola, cubic, log, half_circle]" )

	argp.add_argument( "-b",                dest = "strBase",             metavar = "base_distribution",
	        type = str,   default = "normal",
	        help = "Base distribution: [normal, uniform]" )
			
	argp.add_argument( "--multiprocessing", "-mp",                dest = "strMultiProcessing",             metavar = "MultiprocessingRequest",
	        type = str,   default = "Y",
	        help = "Request to process iterations using multiprocessing - default: Y" )			

	argp.add_argument( "--verbose_output", "-v",                dest = "strVerboseOutput",             metavar = "VerboseOutput",
	        type = str,   default = "N",
	        help = "Request Verbose Output -  Could be Y, or YY(very verbose) or N; default: N  " )			
				
	args = argp.parse_args( ) 

	if  args.strVerboseOutput.startswith("Y"): #  	If requested verbose output
		sys.stderr.write("Program starting "   + strftime("%a, %d %b %Y %H:%M:%S ", gmtime()) + "\n")

	
	fAssociation, fP = _main( args.strBase,
		args.fNoise, 
		args.iRow,  
		args.iCol,  
		args.fSparsity, 
		args.strSpike, 
		args.strMethod ,  
		args.iIter, 
		args.fPermutationNoise,  
		args.strMultiProcessing,
		args.strVerboseOutput)
	sys.stdout.write( "\t".join( ["fAssociation",str(fAssociation)] ) + "\n"  )
	sys.stdout.write( "\t".join( ["fP",str(fP)] ) + "\n" )
	if  args.strVerboseOutput.startswith("Y"):    #  If requested verbose output
		sys.stderr.write("Program ended "  + strftime("%a, %d %b %Y %H:%M:%S ", gmtime()) + "\n")