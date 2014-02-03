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


* The base distribution is the 0th level distribution; prior distribution is the -1th level distribution in clustered data 
	** The base distribution is used whenever possible 

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

To Do 

* I need prior distributions to be tuples, as per invariance principle 
* Make the _check function more arbitrary, not just for checking types, but checking the truth value for other useful features 

* meta _make_invariant should be written to check against bad prior types 
* Let's not reinvent the wheel; for more complicated Bayesian networks, should look at PyMC and implementations such as bnlearn 

* Make presets for distributions, a la HAllA 

"""

####### halla and strudel exists together 
import halla 

import scipy
import numpy 
from numpy import array 
import csv 
import sys 
import itertools 

####### namespace for distributions 
from scipy.stats import invgamma, norm, uniform, logistic, gamma, lognorm, beta, pareto, pearsonr  
from numpy.random import normal, multinomial, dirichlet 

####### namespace for plotting 
from pylab import hist, plot, figure, scatter

class Strudel:
	### Avoid lazy evaluation when possible, to avoid depency problems 
	### Make sure to build the class in such a way that making the distributions arbitrarily complicated is easy to write down 
	### Make meta-strudel a reality 

	def __init__( self ):

		### Wrappers to make frozen rv objects for those distributions that are from numpy 

		class linear: ## wrapper for the linear "distribution" method 
			def __init__( self, param1 = None, param2 = None ):
				self.rvs = lambda shape: numpy.linspace( param1, param2, shape ) 
				self.param1 = param1
				self.param2 = param2 

		class dirichlet: ## wrapper for the dirichlet method 
			def __init__( self, alpha = None ):
				self.object = None 
				if alpha: 
					self.object = lambda shape: dirichlet( alpha, shape )

			def rvs( self, shape ):
				return self.object( shape )

		self.__doc__ 			= __doc__

		self.__description__ 	= """
		   _____ _   _____       _____  ______ _ 
		  / ____| | |  __ \     |  __ \|  ____| |
		 | (___ | |_| |__) |   _| |  | | |__  | |
		  \___ \| __|  _  / | | | |  | |  __| | |
		  ____) | |_| | \ \ |_| | |__| | |____| |
		 |_____/ \__|_|  \_\__,_|_____/|______|_|

		                                         
		Strudel Object for simple synthetic data generation and validation for bioinformatics data.

		Main Functions:

		One-shot functions 
		* randvec : generate iid random samples from specified base distribution 
		* randmat : generate random matrix from specified base distribution 
		* randmix : generate random mixture 
		* randnet : generate random network 

		Pipelines 
		* generate_synthetic data : returns synthetic data with discretized counterpart 

		Presets -- the following presets are available for ease of user use 
		* default 
		* normal
		* uniform 
		"""
		
		self.__version__ 		= "0.0.1"
		self.__author__			= ["YS Joseph Moon", "Curtis Huttenhower"]
		self.__contact__		= "moon.yosup@gmail.com"

		self.keys_attribute 	= ["__description__", "__author__", "__contact__" , "__version__", "base", "base_param", "shape", "num_var", 
			"noise_param", "prior", "prior_param", "prior_shape"]

		self.hash_distributions	= { "uniform"	: uniform, #This distribution is constant between loc and loc + scale.
									"normal"	: norm, 
									"linear"	: linear,
									"gamma"		: gamma, 
									"invgamma"	: invgamma,
									"lognormal"	: lognorm,
									"beta"		: beta,
									"dirichlet" : dirichlet, 
									"pareto"	: pareto, 
									}
		
		self.hash_spike_method = {"linear"			: self.__spike_linear,
									"sine"			: self.__spike_sine,
									"half_circle" 	: self.__spike_half_circle,
									"parabola"		: self.__spike_parabola,
									"cubic"			: self.__spike_cubic,
									"log"			: self.__spike_log,
									"vee"			: self.__spike_vee,	}


		self.hash_conjugate		= { "normal" 	: ("normal", "invgamma"), 
									"uniform"	: "pareto", 
									 } 

		self.hash_num_param		= {"normal"		: 2,
									"invgamma"	: 2,
									"linear"	: 2,
									"gamma"		: 2,
									"pareto"	: 2,
									"dirichlet"	: "?",
									"beta"		: 2, }

		self.generation_methods = ["identity", "half_circle", "sine", "parabola", "cubic", "log", "vee"]

		### Distributions parameters 
		### Let the base distribution be just a single distribution, or an arbitrary tuple of distributions 
		
		### Base distribution 

		self.base 				= "uniform"
		
		self.base_param 		= (-1,2)

		self.shape 				= 100 #number of samples to generate 

		self.num_var			= 10

		self.num_sample 		= self.shape  		

		self.base_distribution	= self._eval( self.hash_distributions, self.base )

		### Prior distribution 

		self.prior 				= self._eval( self.hash_conjugate, self.base ) 

		self.prior_distribution = self._eval( self.hash_distributions, self.prior )

		self.prior_shape		= 100 

		self.prior_param		= ((0,1),(10,0))

		### Linkage 

		self.linkage 			= [] 

		### Noise 

		self.noise_param		= 0.01 # a value between [0,1]; controls amount of noise added to the test distribution 

		self.noise_distribution = lambda variance: norm(0,variance) 

		### Auxillary 

		self.small 				= 0.001

		self.num_iteration 		= 100

		## dynamically add distributions to class namespace

		for k,v in self.hash_distributions.items():
			setattr( self, k, v)  

		assert( 0.0 <= self.noise_param <= 1.0 ), \
			"Noise parameter must be a float between 0 and 1"

	#========================================#
	# Presets 
	#========================================#

	def __preset_default( ):
		pass 

	def __preset_uniform( ):
		pass 

	def __preset_normal( ): 
		pass 

	def __preset_mixture_gaussian( ):
		pass

	def __preset_mixture_uniform( ):
		pass 

	#========================================#
	# Private helper functions 
	#========================================#

	def __eval( self, base, param, pEval = None ):
		"""
		One-one evaluation helper function for _eval 
		""" 

		if isinstance( base, dict ): ## evaluate dict
			try:
				return base[param] 
			except KeyError: 
			## This sometimes happens when you meant for (k1,k2) 
			## to be mapped across the dictionary, and not read as literal key.
			## fail gracefully in this case. 
				try:	
					return tuple([base[p] for p in param])
				except Exception:
					return None 
		elif isinstance( param, str ) and param[0] == ".": ##"dot notation"
			param = param[1:]
			return getattr( base, param )( pEval )
		else: ## else assume that base is a plain old function
			return base(*param) if pEval == "*" else base( param )

	def __eval_many( self, base, aParam, pEval = None ):
		"""
		One-to many evaluation helper function for _eval 

		() -> 
			try as literal
			except Exception 
				try *()
			For dictionaries, try as literal 
			except KeyError
				try map () 
		[] -> map, no questions asked  
		"""

		if self._is_tuple( aParam ):
			return self.__eval( base, aParam, pEval )
		elif self._is_list( aParam ) or self._is_array( aParam ): 
			return [self._eval( base, p, pEval) for p in aParam]
		else:
			return self._eval( base, aParam, pEval )

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
		
		pOut = None 

		if self._is_iter( aBase ) and self._is_iter( aParam[0] ): ## Both aBase and aParam are iterables in 1-1 fashion
			assert( len(aBase) == len(aParam) )
			pOut = [self.__eval(f,x,pEval) for f,x in zip(aBase,aParam)]

		elif self._is_iter( aBase ) and (self._is_tuple( aParam ) or self._is_str( aParam )): ## aParam same for all in aBase; many to one 
			aParam = [aParam for _ in range(len(aBase))] 
			pOut = [self.__eval(f,x,pEval) for f,x in zip(aBase,aParam)]

		elif not self._is_iter( aBase ): ## aBase and aParam are actually single objects 

			try: 
				pOut = self.__eval( aBase, aParam, pEval )
			except Exception: ## last resort is one to many, since this is ambiguous; aParam is usually always an iterable 
				try:
			
					pOut = [self.__eval(aBase, x) for x in aParam] ## one to many, where the keys are taken as literal 
				except Exception:
			
					pOut = [self.__eval_many(aBase, x) for x in aParam] ## one to many, where the keys are themselves taken as iterable 

		return pOut 

	def _rvs( self, aBase, pEval = () ):
		tmp_eval = pEval 
		return self._eval( aBase, aParam = ".rvs", pEval = tmp_eval )

	def _eval_rvs( self, aBase, aParam, pEval = () ):
		return self._rvs( self._eval( aBase, aParam, "*")  , pEval)

	def _len( self, pObj ):
		"""
		Tuples are always considered single objects 
		"""
		if self._is_list( pObj ):
			return len(pObj)
		elif self._is_tuple( pObj ):
			return 1 
		else:
			return 1 

	def _convert( self, pObj, iLen = None ):
		"""
		Convert to iterable, if applicable 
		"""

		if not iLen:
			iLen = 1 
		return [pObj for _ in range(iLen)]

	def _check( self, pObject, pType, pFun = isinstance, pClause = "or" ):
		"""
		Wrapper for type checking 
		"""

		if (isinstance(pType,list) or isinstance(pType,tuple) or isinstance(pType,numpy.ndarray)):
			aType = pType 
		else:
			aType = [pType]

		return reduce( lambda x,y: x or y, [isinstance( pObject, t ) for t in aType], False )

	def _is_list( self, pObject ):
		return self._check( pObject, list )

	def _is_tuple( self, pObject ):
		return self._check( pObject, tuple )

	def _is_array( self, pObject ):
		return self._check( pObject, numpy.ndarray )

	def _is_iter( self, pObject ):
		"""
		Is the object a list or tuple? 
		Disqualify string as a true "iterable" in this sense 
		"""

		return self._check( pObject, [list, tuple, numpy.ndarray] )

	def _is_str( self, pObject ):
		return self._check( pObject, str )

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
				elif len( pObject ) == iMatch:
					aObject = pObject 
				else:
					raise Exception("Length of object does not match specified match function")

			elif self._is_tuple( pObject ): ## Size is 1 
				aObject = pObject if iMatch == 1 else [pObject for _ in range(iMatch)]

			else: ## Size is 1 
				aObject = pObject if iMatch == 1 else [pObject for _ in range(iMatch)]

		return aObject 

	def _make_invariant_many( self, pObject, apMatch ):
		
		assert( self._is_iter( apMatch ) )
		if len( apMatch ) == 1:
			return self._make_invariant( pObject, apMatch[0] )
		else:
			return [self.__make_invariant( pObject, p ) for p in apMatch]
		
	def _categorical( self, aProb, aCategory = None ):
		if not aCategory:
			aCategory = range(len(aProb))
		return next( itertools.compress( aCategory, multinomial( 1, aProb )  ) )
		
	#========================================#
	# Public functions 
	#========================================#

	def get_attribute( self ):
		"""
		Display function for the user to see current settings 
		"""

		for item in self.keys_attribute:
			sys.stderr.write( "\t".join( [item,str(getattr( self, item ))] ) + "\n" ) 


	def set_base( self, aDist ):
		self.base = aDist 
		self.base_distribution = self._eval( self.hash_distributions, self.base )
		self.prior = self._eval( self.hash_conjugate, self.base ) 
		self.prior_distribution = self._eval( self.hash_distributions, self.prior ) if self.prior else None 

	def set_prior( self, aDist ):

		num_param_dist = self.hash_num_param[self.base] #number of parameters required by the base distribution 

		if self._is_iter( aDist ): #aDist is iterable 
			assert( len(aDist) ==  num_param_dist ), "Number of prior distributions must match the number of parameters in the base distribution."
			aDist = aDist 
		
		else: #aDist is a string 
			aDist = self._make_invariant( aDist, num_param_dist )

		self.prior = aDist 
		self.prior_distribution = self._eval( self.hash_distributions, self.prior )

	def set_noise( self, noise ):
		self.noise_param = noise 

	def set_shape( self, shape_param ):
		self.shape = shape_param 

	def set_num_iteration( self, iIter ):
		assert( type(iIter) == int ), "Number of iterations must be an integer."
		self.num_iteration = iIter 
		return self.num_iteration 

	def set_base_param( self, param ):
		param = self._make_invariant( param, self._len( self.base ) ) 
		self.base_param = param 

	def set_prior_param( self, prior_param ):
		prior_param = self._make_invariant( prior_param, self._len( self.base ) )
		self.prior_param = prior_param 

	def set_prior_shape( self, prior_shape ):
		self.prior_shape = prior_shape 

	def set_preset( self, strMethod ):
		strPrefix = "__preset_"
		pMethod = getattr( self, strPrefix + strMethod )
		return pMethod( )

	#========================================#
	# Base generation 
	#========================================#

	def randvec( self, shape = None, dist = None ):
		"""
		Draw `shape` number of iid random samples from a base distribution 
		"""

		if not shape:
			shape = self.shape

		assert( isinstance( shape, int ) ), "Shape must be an int." 

		if not dist:
			dist = self.base_distribution
		else:
			try:
				dist()
			except TypeError:
				try:
					dist = self.hash_distributions[dist]
				except KeyError:
					raise Exception("Please provide a valid distribution")

		return array([dist.rvs( ) for _ in range(shape)])


	def randmat( self, shape = (10,10), dist = None ):
		"""
		Returns a shape-dimensional matrix drawn IID from base distribution 
		Order: Row, Col 

		"""	
		H = self.base_distribution if not dist else dist #base measure 
		
		iRow, iCol = None, None 

		try:
			iCol = int( shape )
		except TypeError:
			iRow, iCol = shape[0], shape[1]
		
		return self._eval_rvs( H, self.base_param, ( shape if iRow else iCol ) )

	def randmix( self, shape = None, pi = None, adj = False ):
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

		if not shape:
			shape = self.shape 

		if not pi:
			iBase = self._len( self.base )
			pi = [1.0/iBase]*iBase 

		iRow, iCol = None, None 

		try:
			iCol = int( shape )
		except TypeError:
			iRow, iCol = shape[0], shape[1]

		param = self._make_invariant( self.base_param, pi )

		H = self.base_distribution 
		
		K = len( pi )

		### Is the base distribution homogeneous or heterogeneous? 
		if isinstance( self.base, list ) or isinstance( self.base, tuple ):
			H = H
		elif isinstance( self.base, str ):
			H = [H for _ in range(K)]

		assert( len( param ) == len( pi ) )
		
		aOut = []  
		
		def _draw():
			z = self._categorical( pi )
			return self._eval_rvs( H[z], param[z], () )

		return [[ _draw() for _ in range(iCol)] for _ in (range(iRow) if iRow else range(1)) ]

	#[((0,0.01),(1,1)), ((5,0.01),(2,1)), ((10,0.01),(3,1))]

	def randnet( self, num_children = 2, shape = None, adj = False ):
		"""

		Generate random clustered network by graphical model representation.

		Parameters
		---------------

			num_children : int 
				children that share prior for each cluster 
			shape : int 
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

			Do I want the base distribution to be the target distribution or the prior distribution? 
			I think making it the prior distribution makes more sense in the framework, but 
			making it the target makes it so much easier to use. 

			Example
			{w} = G < -- x --> F = {y,z}

		"""

		num_clusters = self._len( self.base )

		param = self.prior_param 
		param = self._make_invariant( param, num_clusters )

		if not shape:
			shape = self.shape 
	
		prior_dist = self.prior_distribution 
		prior_dist = self._make_invariant( prior_dist ,num_clusters )		

		### Sanity assertion checks 
		assert( self._len( param ) == num_clusters ), "Number of clusters is not equal to the number of prior parameters"
		assert( num_clusters >= 1 )

		aOut = [] 

		for k in range(num_clusters):

			dist_param = self._eval_rvs( prior_dist[k] if num_clusters != 1 else prior_dist, param[k] if num_clusters != 1 else param )

			for j in range(num_children):

				aIID = self._eval_rvs( self.base_distribution[k] if num_clusters != 1 else self.base_distribution, dist_param, shape )

				aOut.append( aIID )

		return numpy.array(aOut)


	#=============================================================#
	# Adjacency matrix helper functions 
	#=============================================================#

	def generate( self, method = "randmat" ):
		pass 

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

	#========================================#
	# Summary methods
	#========================================#

	def association( self, X, Y, strMethod = "pearson", bPval = False, bParam = True, 
		iIter = None, strNPMethod = "permutation" ):
		"""
		Test the association between arrays X and Y. 
		X and Y can be 1-dimensional arrays or multi-dimensional arrays;
		in the multidimensional case, appropriate multivariate generalizations 
		are to be used in place of traditional methods.

		If a non-parametric test is used, `self.num_iteration` is used for the iteration parameter
		if not specified otherwise. 

		Parameters
		-------------

			X: numpy.ndarray

			Y: numpy.ndarray

			strMethod: str 

			bPval: bool 
				True if p-value is requested 

			bParam: bool
				True if parametric p-value generation requested; otherwise, 
				nonparametric permutation test used instead 


		Returns 
		--------------

			d: float
				association value 

			p: float
				optional p-value 

		"""

		hash_method = {"pearson": None,
						"spearman": None, 
						"kw": None, 
						"anova": None, 
						"x2": None,
						"fisher": None, 
						}

		##Does parametric test exist? 
		hash_parametric = {"pearson": True,
							"spearman": True,
							"anova": True,
							} 

		pMethod = hash_method[strMethod]


		if not iIter:
			iIter = self.num_iteration 

		def _np_error_bars( X, Y, pAssociation, iIter, strMethod = "permutation" ):
			"""
			Helper function to generate error bars in a non-parametric way 

			strMethod: str
				{"bootstrap", "permutation"}
			"""

			def _bootstrap( X, Y, pAssociation, iIter ):
				def __sample( X, Y, pAssociation, iIter ):
					"""
					Perhaps the number of iterations should vary with number of samples 
					"""
					iRow, iCol = X.shape 
					aDraw = array([np.random.randint( iRow ) for _ in range(iIter)])
					aBootX, aBootY = X[aDraw], Y[aDraw]
					return pAssociation( aBootX, aBootY )
				
				aDist = [__sample( X, Y, pAssociation, iIter ) for _ in range(iIter)]
				
				fAssociation = pAssociation( X, Y ) 
				
				fP = sp.stats.percentileofscore( fAssociation, aDist )
				
				return fAssociation, fP  

			def _permutation( X, Y, pAssociation, iIter ):
				def __permute( X, Y, pAssociation ):
					"""
					Give value of pAssociation on one instance of permutation 
					"""
					aPermX = np.random.permutation( X )##without loss of generality, permute X and not Y
					return pAssociation( aPermX, Y )
					
				fAssociation = pAssociation( X,Y ) 
				aDist = [__permute(X,Y) for _ in range(iIter)] ##array containing finite estimation of sampling distribution 
				
				fP = sp.stats.percentileofscore( aPermX, aDist ) ##not sure about syntax; check later
				return fAssociation, fP 

			hashMethod = {"bootstrap": _bootstrap,
							"permutation": _permutation }

			pMethod = None 

			try: 
				pMethod = hashMethod[strMethod]
			except KeyError:
				pMethod = _permutation 

			return pMethod( X, Y, pAssociation, iIter = iIter )

		if bParam:
			assert( hash_paramtric[strMethod] ), "Parametric error bar generation does not exist for the %s method" %strMethod
			aOut = pMethod(X,Y)
			return aOut[0] if not bPval else aOut 

		else:
			if bPval:
				return _np_error_bars( X, Y, pAssociation = pMethod, iIter = iIter, strMethod = strNPMethod )
			else:
				return pMethod(X,Y)				


	#=============================================================#
	# Spike functions 
	#=============================================================#

	def spike( self, X, strMethod = "linear", aArgs = [] ):
		"""
		Introduce spikes between variables. 

		Parameters
		------------

			X: numpy.ndarray 

			strMethod: str 
			{"linear", "sine", "half_circle", "parabola", "cubic", "log", "vee"}

		Returns 
		----------

			Y: numpy.ndarray 
			Transformed Matrix 

		"""
		pMethod = self.hash_spike_method[strMethod]
		if aArgs:
			return pMethod(X, *aArgs)
		else:
			return pMethod(X) 

	def __spike_linear( self, X ):
		shape = X.shape  
		return X + self.noise_distribution( self.noise_param ).rvs( shape )

	def __spike_sine( self, X ):
		shape = X.shape 
		return numpy.sin( X*numpy.pi ) + self.noise_distribution( self.noise_param ).rvs( shape )

	def __spike_half_circle( self, X ):
		shape = X.shape  
		return numpy.sqrt( 1-X**2 ) + self.noise_distribution( self.noise_param ).rvs( shape )

	def __spike_parabola( self, X ):
		shape = X.shape
		return X**2 + self.noise_distribution( self.noise_param ).rvs( shape )

	def __spike_cubic( self, X ):
		shape = X.shape
		return X**3 + self.noise_distribution( self.noise_param ).rvs( shape )

	def __spike_log( self, X ):
		shape = X.shape
		return numpy.log( 1 + X + self.small ) + self.noise_distribution( self.noise_param ).rvs( shape )

	def __spike_vee( self, X ):
		shape = X.shape
		return numpy.sqrt( X**2 ) + self.noise_distribution( self.noise_param ).rvs( shape )

	#==============================================================================#
	# Parametricized shapes under uniform base distribution 
	## Good reference is http://cran.r-project.org/web/packages/mlbench/index.html 
	## Incorporate useful elements from it in the future 
	#==============================================================================#

	### all of these rvs's should be numpy arrays 
	### self-contained devices 

	def identity( self, shape = 100, rvs = array([]) ):
		H = self.base_distribution( *self.base_param )
		v = H.rvs( shape ) if not rvs.any() else rvs 
		x = v + self.noise_distribution( self.noise_param ).rvs( shape )
		return v,x 

	def half_circle( self, shape = 100, rvs = array([]) ): 
		H = self.base_distribution( *self.base_param )
		v = H.rvs( shape ) if not rvs.any() else rvs 
		x = numpy.sqrt( 1-v**2 ) + self.noise_distribution( self.noise_param ).rvs( shape )
		return v,x 

	def sine( self, shape = 100, rvs = array([]) ):
		H = self.base_distribution( *self.base_param )
		v = H.rvs( shape ) if not rvs.any() else rvs 
		x = numpy.sin( v*numpy.pi ) + self.noise_distribution( self.noise_param ).rvs( shape )
		return v,x 
	
	def parabola( self, shape = 100, rvs = array([]) ):
		H = self.base_distribution( *self.base_param )
		v = H.rvs( shape ) if not rvs.any() else rvs 
		x = v**2 + self.noise_distribution( self.noise_param ).rvs( shape )
		return v,x 

	def cubic( self, shape = 100, rvs = array([]) ):
		H = self.base_distribution( *self.base_param )
		v = H.rvs( shape ) if not rvs.any() else rvs 
		x = v**3 + self.noise_distribution( self.noise_param ).rvs( shape )
		return v,x 

	def log( self, shape = 100, rvs = array([]) ):
		H = self.base_distribution( *self.base_param )
		v = H.rvs( shape ) if not rvs.any() else rvs 
		x = numpy.log( 1 + v + self.small ) + self.noise_distribution( self.noise_param ).rvs( shape )
		return v,x 

	def vee( self, shape = 100, rvs = array([]) ): 
		H = self.base_distribution( *self.base_param )
		v = H.rvs( shape ) if not rvs.any() else rvs 
		x = numpy.sqrt( v**2 ) + self.noise_distribution( self.noise_param ).rvs( shape )
		return v,x 

	#=============================================================#
	# Pipelines  
	#=============================================================#
	
	def generate_synthetic_data( self, num_var, sparsity ):
		"""
		Pipeline for synthetic data generation 

		Parameters
		-------------

			num_var : int 
				number of variables in the dataset 

			sparsity : float 
				portion of the dataset that is random noise 

		Returns 
		------------

			X : numpy.ndarray 
				Dataset containing `num_var` rows and `self.shape` columns 

		Notes
		-----------

			* Fix so that random data generation propagates in a greedy fashion. 

		"""

		### Set up correct parameters 
		#self.set_base( "linear" ); self.set_base_param((-1,1)) ## This if I want linear data generation 

		aMethods = [getattr(self, m) for m in self.generation_methods]
		iMethod = len( aMethods )

		assert( 0<= sparsity <=1 ), "sparsity parameter must be a float between 0 and 1"
		prob_random = sparsity ## probability of generating random adjacency 
		prob_linked = 1-prob_random ## probability of generating true adjacency 

		## Currently, assume that each method in aMethods is equally likely to be chosen 
		prob_method = [1.0/iMethod] * iMethod 

		num_samples = self.shape 

		## Initialize adjacency matrix 
		A = numpy.zeros( (num_var, num_var) )

		## Initialize dataset 
		X = numpy.zeros( (num_var, num_samples) )

		bool_list = [self._categorical( [prob_random, prob_linked] ) for i in range(num_var) ]
		print bool_list  

		base_rv = array([]) 

		## Populate the X matrix and identity elements in A matrix 
		for i in range(num_var):
			A[i][i] = 1 
			if bool_list[i]: 
				if not base_rv.any():
					base_rv = self.randmat( num_samples ) 
					X[i] = base_rv 
				else:
					cI = self._categorical( prob_method )
					_, transformed_data = aMethods[cI]( shape = num_samples, rvs = base_rv )
					X[i] = transformed_data 
			else:
				X[i] = self.randmat( num_samples )
				
		## Populate the A matrix for groups 

		linkage_list = itertools.compress( range(num_var), bool_list )

		for i,j in itertools.combinations( linkage_list, 2 ): 
			A[i][j] = 1 ; A[j][i] = 1 

		def _complicated_method( ):

			for i,j in itertools.product( range(num_var), range(num_var) ):
				
				### BUGBUG: need to implement associativity 

				if i > j: ## already visited nodes
					fVal = A[j][i]
					A[i][j] = fVal 
					continue 
				elif i == j:
					A[i][j] = 1
				else:
					if X[i].any() and X[j].any(): ##make sure transitive property of association holds 
						continue 
					elif X[i].any() and not(X[j].any()):
						bI = self._categorical( [prob_random,prob_linked] ) ##binary indicator 
						if not bI:
							X[j] = self.randmat( num_samples )
							## A[i][j] = 0 
						else: 
							cI = self._categorical( prob_method )
							X[i],X[j] = aMethods[cI]( shape = num_samples, rvs = X[i] ) #v,x 
							A[i][j] = 1 
					elif not(X[i].any()) and X[j].any():
						bI = self._categorical( [prob_random,prob_linked] ) ##binary indicator 
						if not bI:
							X[i] = self.randmat( num_samples )
							## A[i][j] = 0 
						else: 
							cI = self._categorical( prob_method )
							X[j], X[i] = aMethods[cI]( shape = num_samples, rvs = X[j] ) #v,x 
							A[i][j] = 1; A[j][i] = 1
					else: ## both need to be initialized 
						## Random or linkage?  
						bI = self._categorical( [prob_random,prob_linked] ) ##binary indicator 
						if not bI:
							X[i] = self.randmat( num_samples )
							X[j] = self.randmat( num_samples ) 
							## A[i][j] = 0 
						else: 
							cI = self._categorical( prob_method )
							X[i], X[j] = aMethods[cI]( shape = num_samples )
							A[i][j] = 1; A[j][i] = 1

			def _enforce_transitivity( A ):
				"""
				make sure transitivity holds
				"""
				first_row = A[0]
				iCol = len(first_row)
				for i,j in itertools.combinations( range(iCol), 2 ):
					if i == 0 or j == 0:
						continue 
					elif first_row[i] and first_row[j]:
						A[i][j] = 1 ; A[j][i] = 1

				return A
				
		return X,A

	def run( self, method = "shapes" ):
		if method == "shapes":

			self.set_noise( 0.01 )
			self.set_base("linear")
			self.set_base_param((-1,1))

			for item in ["identity", "half_circle", "sine", "parabola", "cubic", "log", "vee"]:
				figure() 
				v,x = getattr(self, item)()
				print v
				print x 
				scatter(v,x)
		else:
			pass 

	#=============================================================#
	# Data visualization helpers 
	#=============================================================#

	def view( self, X, A, method = "pearson" ):

		pMethod = method 
		
		hashDiscretize = {"pearson": False, "spearman": False, 
						"mi": True, "mid": True, "adj_mi":True, 
						"adj_mid": True, "norm_mi": True, "norm_mid": True }

		hashMethods = {"pearson": halla.distance.cor, "norm_mi": halla.distance.norm_mi,
						"mi": halla.distance.mi, "norm_mid" : halla.distance.norm_mid}

		pFunDiscretize = halla.stats.discretize 

		pFunMethod = hashMethods[pMethod]

		if hashDiscretize[pMethod]: 
			X = pFunDiscretize(X)

		iRow, iCol = A.shape 
		assert( iRow == iCol )
		
		aOut = [] 

		for i,j in itertools.product(range(iRow),range(iCol)):
			aOut.append( [(i,j), pFunMethod(X[i],X[j]), A[i][j]] ) 

		return array(aOut)

	def roc( self, true_labels, prob_vec ):
		fpr, tpr, thresholds = halla.stats.roc_curve( true_labels, prob_vec )
		roc_auc = halla.stats.auc( fpr, tpr )
		halla.stats.plot_roc( fpr, tpr )
		return roc_auc 

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

	def generate_linkage( self, num_children = 2, shape = None, iClass = 5, method = "randmat" ):
		"""
		Input: 
		Output: Tuple (predictor matrix, response matrix) 
		"""

		if not shape:
			shape = self.shape 

		num_clusters = self._len( self.base )
		num_examples = self.shape 

		aBeta = uniform.rvs(size=num_clusters)

		iRows = num_clusters * num_children
		iCols = num_examples 
		
		#predictor_matrix = self.randnet( num_clusters, num_children, num_examples )
		predictor_matrix = self.randmat( )
		raw = predictor_matrix[:]
		response_matrix = []

		while raw.any():
			
			pCluster, raw = raw[:num_clusters], raw[num_clusters:]
			response_matrix.append( self.classify( numpy.dot( aBeta, pCluster ), iClass ) ) ## currently the `classify` function is used to discretize the continuous data 
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


