import strudel
import argparse 
import sys 
import numpy, scipy

def _permutation( X, Y, strAssociation, iIter ):
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
		aPermX = numpy.random.permutation( X )##without loss of generality, permute X and not Y
		return __invariance( pAssociation( aPermX, Y ) )
	
	X,Y = X.flatten(), Y.flatten() 

	s = strudel.Strudel( )
	pAssociation = s.hash_association_method[strAssociation]
	fAssociation = __invariance( pAssociation( X,Y ) )

	### This part should be parallelized 
	aDist = [__permute(X,Y, pAssociation=pAssociation) for _ in range(iIter)] ##array containing finite estimation of sampling distribution 
	### This part should be parallelized

	fPercentile = scipy.stats.percentileofscore( aDist, fAssociation, kind="strict" ) ##source: Good 2000 
	### \frac{ \sharp\{\rho(\hat{X},Y) \geq \rho(X,Y) \} +1  }{ k + 1 }
	### k number of iterations, \hat{X} is randomized version of X 
	### PercentileofScore function ('strict') is essentially calculating the additive inverse (1-x) of the wanted quantity above 
	### consult scipy documentation at: http://docs.scipy.org/doc/scipy-0.7.x/reference/generated/scipy.stats.percentileofscore.html

	fP = ((1.0-fPercentile/100.0)*iIter + 1)/(iIter+1)

	return fAssociation, fP 

def _main( strBase, fNoise, iRow, iCol, fSparsity, strSpike, strMethod, iIter, fPermutationNoise ):
	s = strudel.Strudel()
	s.set_base(strBase)
	s.set_noise(fNoise)
	s.set_permutation_noise(fPermutationNoise)
	#s.set_sparsity_param( fSparsity )

	##Generate synthetic data 
	X = s.randmat( shape = (iRow, iCol) )
	Y = s.spike( X, sparsity = fSparsity, strMethod = strSpike )

	return _permutation( X, Y, strAssociation = strMethod, iIter = iIter )

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

	args = argp.parse_args( ) 
	fAssociation, fP = _main( args.strBase, args.fNoise, args.iRow, args.iCol, args.fSparsity, args.strSpike, args.strMethod , args.iIter, args.fPermutationNoise )
	sys.stdout.write( "\t".join( ["fAssociation",str(fAssociation)] ) + "\n"  )
	sys.stdout.write( "\t".join( ["fP",str(fP)] ) + "\n" )