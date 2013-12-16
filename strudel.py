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

* Invariance principle: generally, everything should be a list or array when passed into distribution generation functions, if they are not, 
match up so length is preserved 

	** Parameters passed to distributions or functions should be _tuples_ 
	** The object enclosing multiple parameters should be _lists_ 

* Let's not reinvent the wheel; for more complicated Bayesian networks, should look at PyMC and implementations such as bnlearn 

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
* Make meta-strudel a reality: I think at that point using PyMC, PyMix, and python modules for creating networks is necessary 


"""

import scipy
import numpy 
from numpy import array 
import csv 
import sys 
import itertools 

####### namespace for distributions 
from scipy.stats import invgamma, norm, uniform, logistic, gamma
from numpy.random import normal, multinomial, dirichlet 

####### namespace for plotting 
from pylab import hist, plot, figure

class Strudel:
	### Avoid lazy evaluation when possible, to avoid depency problems 
	### Make sure to build the class in such a way that making the distributions arbitrarily complicated is easy to write down 
	### Make meta-strudel a reality 

	def __init__( self ):

		### Wrappers to make frozen rv objects for those distributions that are from numpy 

		class linear: ## wrapper for the linear "distribution" method 
			def __init__( self, param1 = None, param2 = None ):
				self.object = None 
				if param1 and param2:
					self.object = lambda shape: numpy.linspace( param1, param2, shape )

		class dirichlet: ## wrapper for the dirichlet method 
			def __init__( self, alpha = None ):
				self.object = None 
				if alpha: 
					self.object = lambda shape: dirichlet( alpha, shape )

			def rvs( self, shape ):
				return self.object( shape )

		self.hash_distributions	= { "uniform"	: uniform, 
									"normal"	: norm, 
									"linear"	: linear,
									"gamma"		: gamma, 
									"invgamma"	: invgamma,
									"lognormal"	: lognorm,
									"dirichlet" : dirichlet 
									}

		self.hash_conjugate		= { "normal" : ("normal", "invgamma"),
									 } 

		### Let the base distribution be just a single distribution, or an arbitrary tuple of distributions 
		self.base 				= "uniform"
		
		self.base_param 		= (0,1)

		self.shape 				= 100 #number of samples to generate 

		self.base_distribution	= self._eval( self.hash_distributions, self.base )

		#self.base_distribution	= self.hash_distributions[self.base] #scipy stats object; uniform by default  

		self.linkage 			= [] 

		self.noise_param		= 0 # a value between [0,1]; controls amount of noise added to the test distribution 

		self.noise_distribution = lambda variance: norm(0,variance)

		self.small 				= 0.001

		## dynamically add distributions to class namespace

		for k,v in self.hash_distributions.items():
			setattr( self, k, v) 
	#========================================#
	# Private helper functions 
	#========================================#

	def __eval( self, base, param, pEval = None ):
		"""
		one-one evaluation helper function for _eval 
		"""
		if isinstance( base, dict ): ## evaluate dict
			return base[param] 
		elif isinstance( param, str ) and param[0] == ".": ##"dot notation"
			param = param[1:]
			return getattr( base, param )( pEval )
		else: ## else assume that base is a plain old function
			return base(*param) if pEval == "*" else base( param )

	def _eval( self, aBase, aParam, pEval = "*" ):
		"""

		Parameters 
		----------

			aBase : object or list of objects
			aParam : parameters or list of parameters 
			pEval : str 
				Specifies what evaluation rule should be used 
				E.g. "*" feeds in the "aBase(*aParam)" fashion 

		Returns
		---------

			Z : object or list of objects 
				evaluation returned by aBase given aParam; can be single value or iterable. 

		Notes
		---------
		Allows:
			one, one
			many, many
			many, one 
		Ambiguous:
			one, many 
		"""
		
		if (isinstance( aBase, tuple ) or isinstance( aBase, list )) and (isinstance( aParam[0], tuple ) or
			 isinstance( aParam[0], list )): ## Both aBase and aParam are iterables in 1-1 fashion
			assert( len(aBase) == len(aParam) )
			return [self.__eval(f,x,pEval) for f,x in zip(aBase,aParam)]

		elif ( isinstance( aBase, tuple ) or isinstance( aBase, list ) ) and ( isinstance( aParam, tuple ) or 
			isinstance( aParam, list ) or isinstance( aParam, str )): ## aParam same for all in aBase; many to one 
			aParam = [aParam] * len( aBase )
			return [self.__eval(f,x,pEval) for f,x in zip(aBase,aParam)]

		else: ## aBase and aParam are actually single objects 
			try: 
				return self.__eval( aBase, aParam, pEval )
			except Exception: ## last resort is one to many, since this is ambiguous; aParam is usually always an iterable 
				return [self.__eval(aBase, x) for x in aParam]

	def _rvs( self, aBase, pEval = () ):
		tmp_eval = pEval 
		return self._eval( aBase, aParam = ".rvs", pEval = tmp_eval )

	def _eval_rvs( self, aBase, aParam, pEval = () ):
		return self._rvs( self._eval( aBase, aParam, "*")  , pEval)

	def _convert( self, pObj, iLen = None ):
		"""
		Convert to iterable, if applicable 
		"""

		if not iLen:
			iLen = 1 
		return [pObj for _ in range(iLen)]

	def _check( self, pObject, pType ):
		"""
		Wrapper for type checking 
		"""

		if not (isinstance(pObject,list) or isinstance(pObject,tuple) or isinstance(pObject,array)):
			aType = [pType]
		else:
			aType = pType 

		return reduce( lambda x,y: x or y, [isinstance( pObject, t ) for t in aType], False )

	def _is_list( self, pObject ):
		return self._check( pObject, list )

	def _is_tuple( self, pObject ):
		return self._check( pObject, tuple )

	def _is_iter( self, pObject ):
		"""
		Is the object a list or tuple? 
		Disqualify string as a true "iterable" in this sense 
		"""

		return self._check( pObject, [list, tuple, array] )


	def _make_invariant( self, pObject, pMatch ):
		"""
		Invariance principle; match up pObject with an invariance imposed by pMatch 

		Most used case is as a wrapper for the homogeneous case 
		
		[0] -> [0,0,0,...] etc 
		"""

		aObject = None 

		iMatch = None 

		if self._check( pMatch, int ):
			iMatch = pMatch 
		elif self._is_iter( pMatch ):
			iMatch = len( pMatch )

		if iMatch:
			if self._is_list( pObject ):
				if len(  pObject ) == 1:
					aObject = [pObject for _ in range(iMatch)]
				else:
					raise Exception("Length of object does not match specified match function")

			elif self._is_tuple( pObject ):
				aObject = [pObject for _ in range(iMatch)]

			else:
				aObject = [pObject for _ in range(iMatch)]

		return aObject 


	def _categorical( self, aProb, aCategory = None ):
		if not aCategory:
			aCategory = range(len(aProb))
		return next( itertools.compress( aCategory, multinomial( 1, aProb )  ) )
		
	#========================================#
	# Public functions 
	#========================================#

	def set_base( self, aDist ):
		self.base = aDist 
		self.base_distribution = self._eval( self.hash_distributions, self.base )
		return self.base_distribution 

	def set_noise( self, noise ):
		self.noise_param = noise 

	def set_param( self, param ):
		self.base_param = param 

	#========================================#
	# Base generation 
	#========================================#

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
		
		return self._eval_rvs( H, self.base_param, ( shape if iRow else iCol ) )

	def randmix( self, shape = 10, param = [(0,1), (5,1)], pi = [0.5,0.5], adj = False ):
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
		
		K = len( pi )

		### Is the base distribution homogeneous or heterogeneous? 
		if isinstance( self.base, list ) or isinstance( self.base, tuple ):
			H = H
		elif isinstance( self.base, str ):
			H = [H for _ in K]

		assert( len( param ) == len( pi ) )
		
		aOut = []  
		
		def _draw():
			z = self._categorical( pi )
			return self._eval_rvs( H[z], param[z], () )

		return [[ _draw() for _ in range(iCol)] for _ in (range(iRow) if iRow else range(1)) ]



	def randclust( self, num_clusters = 3, num_children = 3, num_examples = 10, dist = ["normal", "normal", "normal"], 
			param = [((0,0.01),(1,1)), ((5,0.01),(2,1)), ((10,0.01),(3,1))], adj = False ):
		"""

		Generate clustered data by graphical model representation.

		Parameters
		---------------

		  	num_clusters : int 
		  		Number of iid clusters in the graphical model 
			num_children : int 
				children that share prior for each cluster 
			num_examples : int 
				number of samples per distribution 
			dist : object or list of objects 
				distributions to be used per cluster 
			param : float or tuple of float or list of tuple of floats 
				hyperparameters of the prior distribution 

		Returns 
		------------
			Z : numpy ndarray 
				p x n matrix, with clustered variables 
		

		Notes
		----------

			Example
			{w} = G < -- x --> F = {y,z}

		"""



		"""
		### Some exception handling 

		if isinstance( dist, tuple ) or isinstance( dist, list ): 

			aDist = [getattr(self, d) for d in dist]
		elif isinstance( dist, str ): ### passing in a string that the module recognizes 
			pass


		if number_clusters != len(dist)
		"""

		aDist = [getattr(self, d) for d in dist]

		aOut = [] 

		zip_param = zip(*param)

		atHyper = zip_param 

		#atHyperMean, atHyperVar = zip_param[0], zip_param[1]

		assert( num_clusters >= 1 )

		#aBase = [self.base_distribution] if num_clusters == 1 else self.base_distribution

		#sigma, mu = None, None 

		for k in range(num_clusters):

			#prior_mu, prior_sigma = atHyperMean[k]
			#alpha, beta = atHyperVar[k]

			#sigma = invgamma.rvs(alpha)
			#mu = norm.rvs(loc=prior_mu,scale=prior_sigma)
			
			## atHyper has columns of hyperparameters 

			#dist_param = self._eval_rvs( self.base_distribution, atHyper )

			self._eval_rvs( self.base_distribution, atHyper )

			#dist_param = [ self.base_distribution(*t).rvs() for t in atHyper ] 

			for j in range(num_children):

				#iid_norm = norm.rvs( loc=mu, scale=sigma, size=num_examples)
				aIID = aDist[k].rvs( *dist_param, size=num_examples )

				aOut.append( iid_norm )

		return numpy.array(aOut)


	#=============================================================#
	# Adjacency matrix helper functions 
	#=============================================================#

	def adjacency( self ):
		"""
		Produce probabilistic adjacency matrix describing the linkage pattern in 
		the synthetic data produced by strudel.

		Parameters
		-------------

		Returns 
		----------


		"""
		pass 

	#==============================================================================#
	# Parametricized shapes under uniform base distribution 
	## Good reference is http://cran.r-project.org/web/packages/mlbench/index.html 
	## Incorporate useful elements from it in the future 
	#==============================================================================#

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

	#=============================================================#
	# Pipelines  
	#=============================================================#
	

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

	#=============================================================#
	# Linkage helper functions   
	#=============================================================#

	def partition_of_unity( self, iSize = 2 ):
		iSize+=1 
		aLin = numpy.linspace(0,1,iSize)
		aInd = zip(range(iSize-1),range(1,iSize+1))
		return [(aLin[i],aLin[j]) for i,j in aInd]
	

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


