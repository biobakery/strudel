"""
=====================================================
STRUDEL: Synthetic Rudimentary Data Exploration
=====================================================

Synthetic data generation for common exploration tasks in microbial data analysis 

Authors
 YS Joseph Moon, Curtis Huttenhower 

URL
 http://huttenhower.org/strudel 

Notes 
-----------

* The base distribution is _always_ the prior and the base parameters are _always_ the hyperparameters 
* Known useful generations 
	* zero- and one-inflated beta distributions
	* zero- and one-inflated gaussian distributions 
	* dirichlet-type data (true and false zeros)
	* non-parametric generation of time series data 
	* parametric generation fo time-series data (e.g. Markov chains, generalized linear models) 
	* test suite including SparseDOSSA	 
* Incorporate themes from `mlbench` 
* Add inference pipeline for certain known models 
	* Mixture models (EM)
	* Clusterd Data 
	* Non-parametric and parametric estimates for mutual information calculation 
* Build class in such a way that making distributions arbitrarily complicated is easy to write down 
* Make meta-strudel a reality 


"""

import scipy
import numpy 
from numpy import array 
import csv 
import sys 
import itertools 

####### namespace for distributions 
from scipy.stats import invgamma, norm, uniform, logistic
from numpy.random import normal, multinomial

####### namespace for plotting 
from pylab import hist, plot, figure

class Strudel:
	### Avoid lazy evaluation when possible, to avoid depency problems 
	### Make sure to build the class in such a way that making the distributions arbitrarily complicated is easy to write down 
	### Make meta-strudel a reality 

	def __init__( self ):

		class linear: ## wrapper for the linear "distribution" method 
			def __init__( self, param1 = None, param2 = None ):
				self.object = None 
				if param1 and param2:
					self.object = lambda shape: numpy.linspace( param1, param2, shape )

			def rvs( self, shape ):
				return self.object( shape )

		self.hash_distributions	= { "uniform"	: uniform, 
									"normal"	: norm, 
									"linear"	: linear,
									"invgamma"	: invgamma,
									}

		self.hash_conjugate		= { "normal" : ("normal", "invgamma"),
									 } 

		### Let the base distribution be just a single distribution, or an arbitrary tuple of distributions 
		self.base 				= "uniform"
		
		self.base_param 		= (0,1)

		self.base_distribution	= self._eval( self.hash_distributions, self.base )

		#self.base_distribution	= self.hash_distributions[self.base] #scipy stats object; uniform by default  

		self.test_distribution	= None 

		self.base_array 		= []

		self.test_array			= []

		self.noise_param		= 0 # a value between [0,1]; controls amount of noise added to the test distribution 

		self.noise_distribution = lambda variance: norm(0,variance)

		self.small 				= 0.001

	### Private helper functions 

	def __eval( self, base, param, pEval = None ):
		"""
		one-one evaluation
		"""
		if isinstance( base, dict ): ## evaluate dict
			return base[param] 
		elif isinstance( param, str ) and param[0] == ".": ##"dot notation"
			param = param[1:]
			return getattr( base, param )
		else: ## else assume that base is a plain old function
			return base(*param) if pEval == "*" else base( param )

	def _eval( self, aBase, aParam, pEval = "*" ):
		"""
		Allows:
			one, one
			many, many
			many, one 
		Disallows:
			one, many 
		"""
		#if isinstance( aParam, str ) and aParam[0] == ".": ## if aParam is a string, then take it to mean that you are performing `aBase.aParam`
		#	aParam = aParam[1:]
		#	if isinstance( aBase, tuple or list ): ## map to multiple aBase
		#		return [getattr( b, aParam )( pEval ) for b in aBase] 
		#	else: ## aBase is actually single object 
		#		return getattr( aBase, aParam )( pEval )
		#else:
		if isinstance( aBase, tuple or list ) and isinstance( aParam[0], tuple or list ): ## Both aBase and aParam are iterables in 1-1 fashion
			assert( len(aBase) == len(aParam) )
			return [self.__eval(f,x,pEval) for f,x in zip(aBase,aParam)]

		elif isinstance( aBase, tuple or list ) and isinstance( aParam, tuple or list ): ## aParam same for all in aBase
			aParam = [aParam] * len( aBase )
			return [self.__eval(f,x,pEval) for f,x in zip(aBase,aParam)]

		else: ## aBase and aParam are actually single objects 
			try: 
				return self.__eval( aBase, aParam, pEval )
			except TypeError: ## last resort is one to many, since this is ambiguous; aParam is usually always an iterable 
				return [self.__eval(aBase, x) for x in aParam]


	def _rvs( self, aBase ):
		return self._eval( aBase, aParam = ".rvs" )

	### Public functions 

	def set_base( self, aDist ):
		self.base = aDist 
		self.base_distribution = self._eval( self.hash_distributions, self.base )
		return self.base_distribution 

	def set_noise( self, noise ):
		self.noise_param = noise 

	def set_param( self, param ):
		self.base_param = param 

	### Base generation 

	def randmat( self, shape = (10,10) ):
		"""
		Returns a shape-dimensional matrix given by base distribution pDist 
		Order: Row, Col 
		"""	
		H = self.base_distribution #base measure 
		
		iRow, iCol = None, None 

		try:
			iCol = int( shape )
		except TypeError:
			iRow, iCol = shape[0], shape[1]
		
		return [ self._eval( p, ".rvs", )  for p in self._eval( H, self.base_param, pEval = "*" ) ]
		return H( *self.base_param ).rvs( shape if iRow else iCol ) 

	def randmix( self, shape = 10, param = [(0,1), (1,1)], pi = [0.5,0.5] ):
		"""
		Draw N copies from a mixture distribution with pdf $ pi^T * H( \cdot | param ) $
		
		Parameters
		------------

			shape <- number of components
			param <- length $k$ parameters to base distribution, $\theta$  
			pi <- length $k$ tuple (vector) to categorical rv Z_n 

		Returns
		----------
		
			N copies from mixture distribution $\sum_{k=1}^{K} \pi_k H(.| \theta )$ 
		
		""" 
		iRow, iCol = None, None 

		try:
			iCol = int( shape )
		except TypeError:
			iRow, iCol = shape[0], shape[1]

		H = self.base_distribution 
		
		assert( len( param ) == len( pi ) )
		
		aOut = [] 

		K = len( param ) 
		
		def _draw():
			return H( *[x for x in itertools.compress( param, multinomial( 1, pi ) )][0] ).rvs()
				
		return [[ _draw() for _ in range(iCol)] for _ in (range(iRow) if iRow else range(1)) ]


	def randclust( self, param ):
		"""
		Draw clustered data; linked through bayesian net 

		Parameters
		-----------

		Returns
		----------

		Notes
		----------

			Example
			{w} = G < -- x --> F = {y,z}

		"""
		pass 

	## Parametricized shapes under uniform base distribution 
	## Good reference is http://cran.r-project.org/web/packages/mlbench/index.html 
	## Incorporate useful elements from it in the future 

	def identity( self, shape = 100 ):
		H = self.base_distribution( *self.base_param )
		v = H.rvs( shape )
		x = v + self.noise_distribution( self.noise_param ).rvs( shape )
		return v,x 

	def half_circle( self, shape = 100 ): 
		H = self.base_distribution( *self.base_param )
		v = H.rvs( shape )
		x = numpy.sqrt( 1-v**2 ) + self.noise_distribution( self.noise_param ).rvs( shape )
		return v,x 

	def sine( self, shape = 100 ):
		H = self.base_distribution( *self.base_param )
		v = H.rvs( shape )
		x = numpy.sin( v*numpy.pi ) + self.noise_distribution( self.noise_param ).rvs( shape )
		return v,x 
	
	def parabola( self, shape = 100 ):
		H = self.base_distribution( *self.base_param )
		v = H.rvs( shape )
		x = v**2 + self.noise_distribution( self.noise_param ).rvs( shape )
		return v,x 

	def cubic( self, shape = 100):
		H = self.base_distribution( *self.base_param )
		v = H.rvs( shape )
		x = v**3 + self.noise_distribution( self.noise_param ).rvs( shape )
		return v,x 

	def log( self, shape = 100):
		H = self.base_distribution( *self.base_param )
		v = H.rvs( shape )
		x = numpy.log( 1 + v + self.small ) + self.noise_distribution( self.noise_param ).rvs( shape )
		return v,x 

	def vee( self, shape = 100 ): 
		H = self.base_distribution( *self.base_param )
		v = H.rvs( shape )
		x = numpy.sqrt( v**2 ) + self.noise_distribution( self.noise_param ).rvs( shape )
		return v,x 

	## Probability distributions under base prior distribution 

	def norm( self, shape = 100 ):
		H = self.base_distribution( *self.base_param )
		v = H.rvs( shape )
		x = v**2 + self.noise_distribution( self.noise_param ).rvs( shape )
		return v,x 




	def run( self ):
		self.set_noise( 0.01 )
		self.set_base("linear")
		self.set_param((-1,1))

		for item in ["identity", "half_circle", "sine", "parabola", "cubic", "log", "vee"]:
			figure() 
			v,x = getattr(self, item)()
			print v
			print x 
			plot(v,x)

	### Linkage helper functions 

	def partition_of_unity( self, iSize = 2 ):
		iSize+=1 
		aLin = numpy.linspace(0,1,iSize)
		aInd = zip(range(iSize-1),range(1,iSize+1))
		return [(aLin[i],aLin[j]) for i,j in aInd]
	
	def generate_clustered_data( self, num_clusters = 3, num_children = 3, num_examples = 20, param = [((0,0.1),(1,1)), ((0,0.25),(3,1)), ((0,0.5),(5,0.5))]):
		"""
		Input: Specify distributions 
		
			num_clusters  
			num_children <- children that share prior for each cluster 
			num_examples 

		Output: p x n matrix, with clustered variables 
		"""

		aOut = [] 

		zip_param = zip(*param)

		atHyperMean, atHyperVar = zip_param[0], zip_param[1]

		for k in range(num_clusters):
			alpha, beta = atHyperVar[k]
			prior_mu, prior_sigma = atHyperMean[k]

			for j in range(num_children):
				sigma = invgamma.rvs(alpha)
				mu = norm.rvs(loc=prior_mu,scale=prior_sigma)

				iid_norm = norm.rvs( loc=mu, scale=sigma, size=num_examples)

				aOut.append( iid_norm )

		return numpy.array(aOut)

	def indicator( self, pArray, pInterval ):

		aOut = [] 

		for i,value in enumerate( pArray ):
			aOut.append( ([ z for z in itertools.compress( range(len(pInterval)), map( lambda x: x[0] <= value < x[1], pInterval ) ) ] or [0] )[0] ) 

		return aOut

	def classify( self, pArray, method = "logistic", iClass = 2 ):
		
		aInterval = self.partition_of_unity( iSize = iClass )

		return self.indicator( logistic.cdf( pArray ), aInterval )

	def generate_linkage( self, num_clusters = 3, num_children = 3, num_examples = 20):
		"""
		Input: 
		Output: Tuple (predictor matrix, response matrix) 
		"""

		aBeta = uniform.rvs(size=num_clusters)

		iRows = num_clusters * num_children
		iCols = num_examples 
		
		predictor_matrix = self.generate_clustered_data( num_clusters, num_children, num_examples )
		raw = predictor_matrix[:]
		response_matrix = []

		while raw.any():
			
			pCluster, raw = raw[:num_clusters], raw[num_clusters:]
			response_matrix.append( self.classify( numpy.dot( aBeta, pCluster ), iClass = 5 ) ) ## currently the `classify` function is used to discretize the continuous data 
			## to generate synthetic metadata linked by an appropriate linkage function 

		return predictor_matrix, response_matrix

#if __name__ == "__main__":
	#pass 
	#predictor, response = generate_linkage( num_clusters =3, num_children=10, num_examples=20 )

	#csvw = csv.writer( sys.stdout, csv.excel_tab )

	#csvw.writerow( ["#Predictor"] )

	#for i, item in enumerate( predictor ): 
	#	csvw.writerow( ["x" +str(i)] + list( item ) )

	#csvw.writerow( ["#Response"])

	#for i, item in enumerate( response ):
	#	csvw.writerow( ["y" + str(i)]  + list ( item ) )


