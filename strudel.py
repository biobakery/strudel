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

* General programming style: do NOT pluralize anything at all; only if absolutely necessary or is already used idiomatically in the scientific vernacular 

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
* Let's absorb HAllA inside the strudel framework -- strudel ver 0.2 

* Add ability to measure time spent in the `association` function 
* Parallelizability 
	
strudel-py/StrudelPy vs. scikit-explore?
"""

####### Exception handling for optional packages 

try:
	import halla 
	bHAllA = True 
except ImportError:
	bHAllA = False 

####### Required packages; eventually, will be wrapped around Strudel/HAllA objects 

import scipy
import numpy 
from numpy import array 
import sklearn
import csv 
import sys 
import itertools 

####### namespace for distributions 
from scipy.stats import invgamma, norm, uniform, logistic, gamma, lognorm, beta, pareto, pearsonr  
from numpy.random import normal, multinomial, dirichlet 
from sklearn.metrics import roc_curve, auc 

####### namespace for plotting 
#from pylab import hist, plot, figure, scatter

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
		
		self.__version__ 		= "0.0.2"
		self.__author__			= ["YS Joseph Moon", "Curtis Huttenhower"]
		self.__contact__		= "moon.yosup@gmail.com"

		self.keys_attribute 	= ["__description__", "__author__", "__contact__" , "__version__", "base", "base_param", "shape", "num_var", 
			"noise_param", "prior", "prior_param", "prior_shape"]


		### Modes 

		self.mode_verbose = False
		self.mode_easy = True 

		### Prefixes 
		self.prefix_preset_synthetic_data = "__preset_synthetic_data"

		self.prefix_prest 		= "__preset"

		## "line" is used as a noun; "linear" is used as an adjective with respect to the distribution 

		self.hash_distribution	= { "uniform"	: uniform, #This distribution is constant between loc and loc + scale.
									"normal"	: norm, 
									"linear"	: linear,
									"gamma"		: gamma, 
									"invgamma"	: invgamma,
									"lognormal"	: lognorm,
									"beta"		: beta,
									"dirichlet" : dirichlet, 
									"pareto"	: pareto, 
									}
		
		self.hash_distribution_param_default = { "uniform"	: (-1,2), #This distribution is constant between loc and loc + scale.
									"normal"	: (0,1), 
									"linear"	: (-1,1),
									"gamma"		: None, 
									"invgamma"	: None,
									"lognormal"	: None,
									"beta"		: None,
									"dirichlet" : None, 
									"pareto"	: None, 
									}

		###### BUGBUG: mixtures should be scaleable to arbitrary number of dimensions 
		self.hash_distribution_param_mixture = {"normal":[(0,1),(4,1)], "uniform": [(-1,2),(-0.5,2)]}

		self.hash_distribution_param = {"default":self.hash_distribution_param_default,"mixture": self.hash_distribution_param_mixture}

		self.hash_spike_method = {	"line"			: self.__spike_line,
									"sine"			: self.__spike_sine,
									"half_circle" 	: self.__spike_half_circle,
									"circle"		: self.__spike_circle, 
									"parabola"		: self.__spike_parabola,
									"cubic"			: self.__spike_cubic,
									"log"			: self.__spike_log,
									"vee"			: self.__spike_vee,
									"ellipse"		: self.__spike_ellipse, 
									"line_line"		: self.__spike_line_line,
									"x"				: self.__spike_x,
									"line_parabola"	: self.__spike_line_parabola,
									"non-coexistence": self.__spike_non_coexistence,
									}


		## Reshef Types: Two Lines, Line and parabola, X, ellipse, sinusoid, non-coexistence 
		## Reshef Types: Random, Linear, Cubic, Exponential, Sinusoidal, Categorical, Periodic/Linear, Parabolic
		## Tibshirani Types 

		## Common statistical functions: 
		## http://docs.scipy.org/doc/scipy/reference/stats.html


		self.hash_association_method = {"pearson": scipy.stats.pearsonr,
										"spearman": scipy.stats.spearmanr, 
										"kw": scipy.stats.mstats.kruskalwallis, 
										"anova": scipy.stats.f_oneway, 
										"x2": scipy.stats.chisquare,
										"fisher": scipy.stats.fisher_exact, 
										"norm_mi": halla.distance.norm_mi, 
										"mi": halla.distance.mi,
										"norm_mid": halla.distance.norm_mid,
										"halla": lambda X,Y: halla.HAllA(X,Y).run( ),
										}

		self.hash_association_method_discretize = {"pearson": False,
										"spearman": False,
										"kw": False,
										"anova": False,
										"x2": False,
										"fisher": False,
										"norm_mi": True,
										"mi": True,
										"norm_mid": True,
										"halla": False
										}

		self.hash_association_parametric = {"pearson": True,
										"spearman": True,
										"anova": True,
										} 

	
		self.hash_meta_association_method = {"halla": {"halla_mi": None,
											"halla_mid": None, 
											"halla_norm_mid": None,
											"halla_copula": None }}


		self.hash_conjugate		= { "normal" 	: ("normal", "invgamma"), 
									"uniform"	: "pareto", 
									 } 
		self.hash_num_param		= {"normal"		: 2,
									"invgamma"	: 2,
									"linear"	: 2,
									"gamma"		: 2,
									"pareto"	: 2,
									"dirichlet"	: -1, ##this to indicate that you don't know 
									"beta"		: 2, }


		self.list_distribution = self.hash_distribution.keys() 

		self.list_spike_method = self.hash_spike_method.keys() 

		self.list_association_method = self.hash_association_method.keys() 

		self.generation_method = ["identity", "half_circle", "sine", "parabola", "cubic", "log", "vee"]

		### Distributions parameters 
		### Let the base distribution be just a single distribution, or an arbitrary tuple of distributions 
		
		### Base distribution 

		self.base 				= "uniform"
		
		self.base_param 		= (-1,2)

		self.shape 				= 100 #number of samples to generate 

		self.num_var			= 10

		self.num_sample 		= self.shape  		

		self.base_distribution	= self._eval( self.hash_distribution, self.base )

		### Prior distribution 

		self.prior 				= self._eval( self.hash_conjugate, self.base ) 

		self.prior_distribution = self._eval( self.hash_distribution, self.prior )

		self.prior_shape		= 100 

		self.prior_param		= ((0,1),(10,0))

		### Linkage 

		self.linkage 			= [] 

		### Noise 

		self.noise_param		= 0.1 # a value between [0,1]; controls amount of noise added to the test distribution 

		self.noise_distribution = lambda variance: norm(0,variance) 

		### Permutation Noise 

		self.permutation_noise_param = 0.0 # a value between [0,1]; controls amount of noise induced in linkages by permutation 

		self.permutation_noise_deg = 10 # degree of accuracy of the permutation noise parameter 

		### Sparsity 

		self.sparsity_param		= 0.5 # a value between [0,1]; 0 means completely sparse; 1 means completely dense 
		#this should really be called `density_param` but "density" can mean so many different things and is ambiguous 

		### Auxillary 

		self.small 				= 0.000

		self.big 				= 10.0

		self.num_iteration 		= 100

		### P values and Thresholds 
		self.p 					= 0.05
		self.q 					= 0.05

		### Set Data 

		self.meta_array 		= None 

		## dynamically add distributions to class namespace

		for k,v in self.hash_distribution.items( ):
			setattr( self, k, v)  

		## dynamically add preset definitions 
		self.hash_preset = {"spiked_data": {}, "distribution": {}}
		#self.__set_definition_preset_spiked_data( ) 

		assert( 0.0 <= self.noise_param <= 1.0 ), \
			"Noise parameter must be a float between 0 and 1"

		#self.hash_preset = {"vec": {}, "mat": {"normal_linear_spike": getattr(self, "__preset_synthetic_data_normal_linear_spike"),
		#									"normal_sine_spike": getattr(self,"__preset_synthetic_data_normal_sine_spike"),
		#									"uniform_linear_spike": getattr( self, "__preset_synthetic_data_uniform_linear_spike" ),
		#									"mixture_normal_linear_spike": getattr( self, "__preset_synthetic_data_mixture_normal_linear_spike" ),
		#									"mixture_uniform_linear_spike": getattr( self, "__preset_synthetic_data_mixture_uniform_linear_spike" )}, "pipe": {}}

		#self.list_vector_preset = self.hash_preset["vec"].keys()
		self.list_spiked_data_preset = self.hash_preset["spiked_data"].keys()
		#self.list_pipeline_preset = self.hash_preset["pipe"].keys()

		for strSpikeMethod in self.list_spike_method:
			for strType in self.hash_distribution_param.keys():
				for strDist in self.hash_distribution_param[strType].keys():
					strKeyFinal = strType + "_" + strDist + "_" + strSpikeMethod + "_spike" 
					self.list_spiked_data_preset.append( strKeyFinal )

	#==========================================================#
	# Static Methods 
	#==========================================================# 

	#### This is modified code for Strudel; there is a need to consolidate 
	#@staticmethod 
	def m( self, pArray, pFunc, axis = 0 ):
		""" 
		Maps pFunc over the array pArray 
		"""

		if bool(axis): 
			pArray = pArray.T
			# Set the axis as per numpy convention 
		if isinstance( pFunc , numpy.ndarray ):
			return pArray[pFunc]
		else: #generic function type
			return array( [pFunc(item) for item in pArray] ) 

	#@staticmethod 
	def mp( self, pArray, pFunc, axis = 0 ):
		"""
		Map _by pairs_ ; i.e. apply pFunc over all possible pairs in pArray 
		"""

		if bool(axis): 
			pArray = pArray.T

		pIndices = itertools.combinations( range(pArray.shape[0]), 2 )

		return array([pFunc(pArray[i],pArray[j]) for i,j in pIndices])

	def md( self, pArray1, pArray2, pFunc, axis = 0 ):
		"""
		Map _by dot product_
		"""

		if bool(axis): 
			pArray1, pArray2 = pArray1.T, pArray2.T

		iRow1 = len(pArray1)
		iRow2 = len(pArray2)


		assert( iRow1 == iRow2 )
		aOut = [] 
		for i,item in enumerate(pArray1):
			aOut.append( pFunc(item, pArray2[i]) )
		return aOut 

	#@staticmethod 
	def mc( self, pArray1, pArray2, pFunc, axis = 0, bExpand = False ):
		"""
		Map _by cross product_ for ; i.e. apply pFunc over all possible pairs in pArray1 X pArray2 
		
		If not bExpand, gives a flattened array; else give full expanded array 
		"""

		if bool(axis): 
			pArray1, pArray2 = pArray1.T, pArray2.T

		#iRow1, iCol1 = pArray1.shape
		#iRow2, iCol2 = pArray2.shape 

		iRow1 = len(pArray1)
		iRow2 = len(pArray2)

		pIndices = itertools.product( range(iRow1), range(iRow2) )

		aOut = array([pFunc(pArray1[i],pArray2[j]) for i,j in pIndices])
		return ( aOut if not bExpand else numpy.reshape( aOut, (iRow1, iRow2) ) )

	#@staticmethod 
	def r( self, pArray, pFunc, axis = 0 ):
		"""
		Reduce over array 

		pFunc is X x Y -> R 

		"""
		if bool(axis):
			pArray = pArray.T

		return reduce( pFunc, pArray )

	#@staticmethod 
	def rd( self, pArray, strMethod = "pca" ):
		"""
		General reduce-dimension method 

		Parameters 
		--------------

			pArray: numpy.ndarray
			strMethod : str 
				{"pca", "mca"}

		Returns 
		-------------

		"""
		pass 

	def rc( self, pArray1, pArray2, strMethod = "cca" ):
		"""
		Reduce dimensions couple-wise 
		"""
		pass 

	def p( self, pArray, axis = 0 ):
		"""
		Permute along given axis
		"""
		if bool(axis):
			pArray = pArray.T

		return numpy.random.permutation( pArray )

	def pp( self, pArray, iIter = 1, axis = 0 ):
		"""
		Permute a random pair within the array along given axis 
		"""

		if bool(axis):
			pArray = pArray.T

		iLen = len(pArray)

		def _draw_two( ):
			aOut = [] 
			for _ in range(2):
				aOut.append( self._categorical( [1.0/iLen]*iLen ) ) 
				## Sometimes you can pick the same two; but this probability becomes O(n^-2) small
			return aOut 

		def _shift( _pArray, iOne, iTwo ):
			_pArray[iTwo], _pArray[iOne] =  _pArray[iOne], _pArray[iTwo] 
			return _pArray 

		for _ in range(iIter): 
			iOne, iTwo = _draw_two( )
			pArray = _shift( pArray, iOne, iTwo )

		return pArray


	def pn( self, pArray, fNoise = 0.1, axis = 0 ):
		"""
		Induce permutation noise 
		"""

		if bool(axis):
			pArray = pArray.T

		if not fNoise:
			fNoise = self.permutation_noise_param 

		assert( 0.0 <= fNoise <= 1.0 )

		if fNoise == 0.0:
			return pArray 

		else:
			iLen = len(pArray)
			iSizeEff = iLen/2  ## effective size 
			aInterval = self.partition_of_unity( iSize = iSizeEff )
			iIter = self.indicator(array([fNoise]), aInterval)[0] ##how many times?

			pArrayNew = self.pp( pArray = pArray, iIter = iIter )
			return pArrayNew

	def interleave( self, pArray1, pArray2, axis = 0 ):
		a,b = pArray1, pArray2
		c = numpy.empty((a.size + b.size,), dtype=a.dtype)
		c[0::2] = a
		c[1::2] = b
		return c 

	#========================================#
	# Presets 
	#========================================#

	#----------------------------------------#
	# General Pipeline 
	#----------------------------------------#

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

	## self.hash_preset

	#----------------------------------------#
	# Synthetic Data Generation 
	#----------------------------------------#

	"""
	Key: __preset_synthetic_data_ 

	*** Consider different types of inputs (both continuous + categorical):
	**** Uniform random, no cluster structure, linear spikes
	**** Normal random, no cluster structure, linear spikes
	**** Uniform mixture model (clusters), linear spikes
	**** Normal mixture model (clusters), linear spikes
	**** Normal random, no cluster structure, sine spike
	"""

	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	# Random Vector 
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

	## On-the-fly preset generation 

	#----------------------------------------#
	# Plots/Visualization 
	#----------------------------------------#

	#========================================#
	# Private helper functions 
	#========================================#

	def __load_datum( self ):
		pass 
	
	def __load_data( self ):
		pass 
 
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
		elif isinstance( param, int ):
			return base[param]
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


		Example
		-------------
			#import strudel, halla
			#s = strudel.Strudel( )
			#hash = {"a":1,"b":2,"c":3}
			#hash_new = {"a":10,"b":20,"c":30} 
			#s._eval(halla, [".Tree"]*3) ##rather than [getattr( halla, x ) for x in ["Tree"]*3]; significantly shorter
			#s._eval(hash,["a","b"]) ##rather than [hash[strKey] for strKey in ["a","b"]]
			#s._eval([hash,hash_new], "a") ##c.f. [h["a"] for h in [hash, hash_new]]

		"""
		
		pOut = None 

		#if self._is_iter( aBase ) and self._is_iter( aParam[0] ): ## Both aBase and aParam are iterables in 1-1 fashion
		### WARNING: I hope this doesn't cause a mess -- why did I do the [0] in the first place? 

		if self._is_iter( aBase ) and self._is_iter( aParam ): ## Both aBase and aParam are iterables in 1-1 fashion
			#print "many to many!"
			assert( len(aBase) == len(aParam) )
			pOut = [self.__eval(f,x,pEval) for f,x in zip(aBase,aParam)]

		elif self._is_iter( aBase ) and (self._is_tuple( aParam ) or self._is_str( aParam ) or self._is_int( aParam ) ): ## aParam same for all in aBase; many to one 
			#print "many to one"
			aParam = [aParam for _ in range(len(aBase))] 
			pOut = [self.__eval(f,x,pEval) for f,x in zip(aBase,aParam)]

		elif not self._is_iter( aBase ): ## aBase and aParam are actually single objects 
			#print "single objects"
			try: 
				pOut = self.__eval( aBase, aParam, pEval )
			except Exception: ## last resort is one to many, since this is ambiguous; aParam is usually always an iterable 
				try:
			
					pOut = [self.__eval(aBase, x) for x in aParam] ## one to many, where the keys are taken as literal 
				except Exception:
		
					pOut = [self.__eval_many(aBase, x) for x in aParam] ## one to many, where the keys are themselves taken as iterable 
					## Note: one to many is ambiguous because base is either a single object or an iterable, whereas the parameters can be 
					## nasty things like ((1,1),(1,2)) where the instances (1,1) and (1,2) should be taken as single objects, not a true 
					## "iterable"

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

	def _convert_to_iterable( self, pObj, iLen = None ):
		"""
		Convert to iterable, if applicable 
		"""

		if not iLen:
			iLen = 1 
		return [pObj for _ in range(iLen)]

	#----------------------------------------#
	# Type-Checking Methods 
	#----------------------------------------#

	def __status( self ):
		"""
		Checks that the current status of HAlLA is okay, w.r.t. types, etc 
		"""

		assert( 0.0 <= self.noise_param <= 1.0 ), \
			"Noise parameter must be a float between 0.0 and 1.0"

		assert( 0.0 <= self.sparsity_param <= 1.0 ), \
			"Sparsity parameter must be a float between 0.0 and 1.0"


	def _check( self, pObject, pType, pFun = isinstance, pClause = "or" ):
		"""
		Wrapper for type checking 
		"""

		if (isinstance(pType,list) or isinstance(pType,tuple) or isinstance(pType,numpy.ndarray)):
			aType = pType 
		else:
			aType = [pType]

		return reduce( lambda x,y: x or y, [isinstance( pObject, t ) for t in aType], False )

	def _cross_check( self, pX, pY, pFun = len ):
		"""
		Checks that pX and pY are consistent with each other, in terms of specified function pFun. 
		"""
	
	def _is_meta( self, pObject ):
		"""	
		Is pObject an iterable of iterable? 
		"""

		try: 
			pObject[0]
			return self._is_iter( pObject[0] )	
		except IndexError:
			return False 

	def _is_empty( self, pObject ):
		"""
		Wrapper for both numpy arrays and regular lists 
		"""
		
		aObject = array(pObject)

		return not aObject.any()

	### These functions are absolutely unncessary; get rid of them! 
	def _is_list( self, pObject ):
		return self._check( pObject, list )

	def _is_tuple( self, pObject ):
		return self._check( pObject, tuple )

	def _is_str( self, pObject ):
		return self._check( pObject, str )

	def _is_int( self, pObject ):
		return self._check( pObject, int )    

	def _is_array( self, pObject ):
		return self._check( pObject, numpy.ndarray )

	def _is_1d( self, pObject ):
		"""
		>>> import strudel 
		>>> s = strudel.Strudel( )
		>>> s._is_1d( [] )
		"""

		strErrorMessage = "Object empty; cannot determine type"
		bEmpty = self._is_empty( pObject )

		## Enforce non-empty invariance 
		if bEmpty:
			raise Exception(strErrorMessage)

		## Assume that pObject is non-empty 
		try:
			iRow, iCol = pObject.shape 
			return( iRow == 1 ) 
		except ValueError: ## actual arrays but are 1-dimensional
			return True
		except AttributeError: ## not actual arrays but python lists 
			return not self._is_iter( pObject[0] )

	def _is_iter( self, pObject ):
		"""
		Is the object a list or tuple? 
		Disqualify string as a true "iterable" in this sense 
		"""

		return self._check( pObject, [list, tuple, numpy.ndarray] )

	#----------------------------------------#
	# Invariance principle 
	#----------------------------------------#

	def _make_invariant( self, pObject, pMatch ):
		"""
		Invariance principle; match up pObject with an invariance imposed by pMatch 

		Most used case is as a wrapper for the homogeneous case 

		Usually pObject is an int; use case is as follows:

			[0] -> [0,0,0,...] 

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

		
	#========================================#
	# Public functions 
	#========================================#

	#----------------------------------------#
	# "Get" Methods 
	#----------------------------------------#

	def get_matrix_preset( self ):
		return self.list_matrix_preset 

	def get_preset_hash( self ):
		return self.hash_preset 

	def get_spike_method( self ):
		return self.list_spike_method 

	def get_association_method( self ):
		return self.list_association_method 

	def get_distribution_method( self ):
		return self.list_distribution 

	def get_generation_method( self ):
		return self.generation_methods 

	def get_attribute( self ):
		"""
		Display function for the user to see current settings 
		"""

		for item in self.keys_attribute:
			sys.stderr.write( "\t".join( [item,str(getattr( self, item ))] ) + "\n" ) 

	def get_data( self ):
		return self.meta_array

	def get_feature( self ):
		pass 

	#----------------------------------------#
	# "Set" Methods 
	#----------------------------------------#

	def set_base( self, aDist ):
		self.base = aDist 
		self.base_distribution = self._eval( self.hash_distribution, self.base )
		
		###BUGBUG: must fix for general usage pattern; currently, the following conditionals assume that aDist is singular
		if self.mode_easy and self._is_str( aDist ):
			aSplit = aDist.split("-")
			if "mixture" in aSplit:
				if self.hash_distribution_param["mixture"].get(aDist): ##mixture has higher priority than default
					self.set_base_param( self.hash_distribution_param["mixture"][aDist])
			else:
				if self.hash_distribution_param["default"].get(aDist): ##automatically set base parameters 
					self.set_base_param( self.hash_distribution_param["default"][aDist])
				
		self.prior = self._eval( self.hash_conjugate, self.base ) 
		self.prior_distribution = self._eval( self.hash_distribution, self.prior ) if self.prior else None 

	def set_prior( self, aDist ):

		num_param_dist = self.hash_num_param[self.base] #number of parameters required by the base distribution 

		if self._is_iter( aDist ): #aDist is iterable 
			assert( len(aDist) ==  num_param_dist ), "Number of prior distributions must match the number of parameters in the base distribution."
			aDist = aDist 
		
		else: #aDist is a string 
			aDist = self._make_invariant( aDist, num_param_dist )

		self.prior = aDist 
		self.prior_distribution = self._eval( self.hash_distribution, self.prior )

	def set_noise( self, noise ):
		self.noise_param = noise 

	def set_shape( self, shape_param ):
		self.shape = shape_param 

	def set_num_iteration( self, iIter ):
		assert( type(iIter) == int ), "Number of iterations must be an integer."
		self.num_iteration = iIter 
		return self.num_iteration 

	def set_permutation_noise( self, fPermutationNoise ):
		self.permutation_noise_param = fPermutationNoise
		return self.permutation_noise_param

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
	# Inference Methods 
	#========================================#

	def fit_to_data( self, strMethod ):
		"""
		Infer parameters based on model, now use those to generate synthetic data or run some baseline pipeline 
		"""
		pass 

	#========================================#
	# Summary Methods -- exploration 
	#========================================#

	## As adherence to design philosophy, append what the function _is_ (e.g. "summary_") even if this lengthens the function call 
	## First write summary methods for 1-d arrays, then for general arrays, and then for pairs, and then for arbitrary number of pairs 

	def summary_explore( self ):
		"""
		Explore the given dataset, giving summary statistics and association parameters. 
		Can utilize optional modules, such as HAllA. 
		"""
		hashMethods = {}

	def summary_association( self, X, Y ):
		"""
		A summary slew of association values in dataset 
		"""
		hashMethods = {} 
		pass 

	def _association( self, X, Y, strMethod = "pearson", bPval = False, bParam = False, 
		bNormalize = False, iIter = None, strNPMethod = "permutation", preset = None ):
		"""
		1-1 association testing
		"""		

		if not "halla" in strMethod:
			assert( not(self._is_iter(X[0])) and not(self._is_iter(Y[0])) ), "X and Y must be 1-dimensional arrays or python iterable object"
		
		#hash_method = {"pearson": scipy.stats.pearsonr,
		#				"spearman": scipy.stats.spearmanr, 
		#				"kw": scipy.stats.mstats.kruskalwallis, 
		#				"anova": scipy.stats.f_oneway, 
		#				"x2": scipy.stats.chisquare,
		#				"fisher": scipy.stats.fisher_exact, 
		#				"norm_mi": halla.distance.norm_mi, 
		#				"mi": halla.distance.mi,
		#				"norm_mid": halla.distance.norm_mid 
		#				}


		##Does parametric test exist? 
		
		# Automatically determine if have to be discretized? 
		#hashDiscretize = {"pearson": False, "spearman": False, 
		#				"mi": True, "mid": True, "adj_mi":True, 
		#				"adj_mid": True, "norm_mi": True, "norm_mid": True }

		bDiscretize = bNormalize ##BUGBUG fix this later when there are more normalization methods other than discretization 

		pMethod = self.hash_association_method[strMethod]

		bDiscretize = self.hash_association_method_discretize[strMethod]

		if bDiscretize:
			X,Y = halla.discretize( X ), halla.discretize( Y )

		if not iIter:
			iIter = self.num_iteration 

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
					aDraw = array([numpy.random.randint( iRow ) for _ in range(iIter)])
					aBootX, aBootY = X[aDraw], Y[aDraw]
					return __invariance( pAssociation( aBootX, aBootY ) )
				
				#sys.stderr.write("Generating " + str(iIter) + " bootstraps ...\n")
				aDist = [__sample( X, Y, pAssociation, iIter ) for _ in range(iIter)]
				
				fAssociation = __invariance( pAssociation( X, Y ) )
				
				fP = scipy.stats.percentileofscore( aDist, fAssociation )/100.0
				
				return fAssociation, fP  

			def _permutation( X, Y, pAssociation, iIter ):
				def __permute( X, Y, pAssociation ):
					"""
					Give value of pAssociation on one instance of permutation 
					"""
					aPermX = numpy.random.permutation( X )##without loss of generality, permute X and not Y
					return __invariance( pAssociation( aPermX, Y ) )
					
				fAssociation = __invariance( pAssociation( X,Y ) )
				#sys.stderr.write( "Generating " + str(iIter) + " permuations ...\n" )
				aDist = [__permute(X,Y, pAssociation=pAssociation) for _ in range(iIter)] ##array containing finite estimation of sampling distribution 
				

				fP = 1.0 - scipy.stats.percentileofscore( aDist, fAssociation )/100.0 
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
			#sys.stderr.write("Parametric pval generation\n")
			assert( self.hash_association_parametric[strMethod] ), "Parametric error bar generation does not exist for the %s method" %strMethod
			aOut = pMethod(X,Y)
			
			if bPval == -1:
				return aOut[0]
			elif bPval == 0:
				return aOut 
			elif bPval == 1:
				return aOut[1]

		else:
			#sys.stderr.write("Nonparametric pval generation\n")
			if bPval == -1:
				#sys.stderr.write("Nonparametric association\n")
				aOut = pMethod(X,Y)				
				## BUGBUG: define general association/distance objects so that this can be avoided
				## Currently, there is a need to wrap around different association definition 
				return __invariance( aOut )
			
			elif bPval == 0:
				return _np_error_bars( X, Y, pAssociation = pMethod, iIter = iIter, strMethod = strNPMethod )
			elif bPval == 1:
				#sys.stderr.write("Nonparametric pval\n")
				return _np_error_bars( X, Y, pAssociation = pMethod, iIter = iIter, strMethod = strNPMethod )[1]
				

	def _association_many( self, X, Y, strMethod = "pearson", bPval = False, bParam = False, 
		bNormalize = False, iIter = None, strNPMethod = "permutation", strReduceMethod = None, 
		strCorrectionMethod = None, preset = None ):

		assert( not self._is_1d( X ) and not self._is_1d( Y ) ) 

		if not strReduceMethod:
			aAssoc = self.mc( X, Y, lambda x,y: self._association( x,y, strMethod = strMethod, bPval = bPval, bNormalize = bNormalize, 
				iIter = iIter, strNPMethod = strNPMethod, preset = preset ) )

			if strCorrectionMethod:
				aAssoc_adjusted = ( halla.stats.p_adjust( [x[1] for x in aAssoc], method = "fdr" ) if bPval else aAssoc ) 
			else:
				aAssoc_adjusted = aAssoc 

			return aAssoc_adjusted 

			
	def association( self, X, Y = None, strMethod = "pearson", bPval = -1, bParam = False, 
		bNormalize = False, iIter = None, strNPMethod = "permutation", strReduceMethod = None, 
		strCorrectionMethod = None, preset = None ):
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

			bPval: int  
		
				-1 -> give only association value 
				0 -> give both association value and pvalue 
				1 -> give only pvalue 

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
		
		## just like in the "_eval" functions, there are potentially 4 cases
		## 1. one-to-one <- 
		## 2. one-to-many
		## 3. many-to-one 
		## 4. many-to-many <-

		## 1 and 4 are the msot important; focus on those 

		X,Y = array(X), array(Y) 

		if self._is_empty( Y ):
			assert( not self._is_1d( X ) )
			#"print Y is empty"
			return self.mp( X, lambda x,y: self._association( x,y, strMethod = strMethod, bPval = bPval, bNormalize = bNormalize, 
				iIter = iIter, strNPMethod = strNPMethod, preset = preset ) )

		elif not self._is_empty( X ) and not self._is_empty( Y ):
			if not self._is_1d( X ) and not self._is_1d( Y ): 
				if not("halla" in strMethod):
					return self._association_many( X, Y, strMethod = strMethod, bPval = bPval, bParam = bParam, bNormalize = bNormalize, iIter = iIter, strNPMethod = strNPMethod )
				else:
					#return self._association( X,Y, strMethod = strMethod, bPval = bPval, bParam = bParam, bNormalize = bNormalize, iIter = iIter, strNPMethod = strNPMethod  )
					aOut = halla.HAllA(X,Y).run()
					return aOut[0]
			elif self._is_1d( X ) and self._is_1d( Y ): 
				return self._association( X, Y, strMethod = strMethod, bPval = bPval, bParam = bParam, bNormalize = bNormalize, iIter = iIter, strNPMethod = strNPMethod )
	
		#assert( self._is_array( X ) and self._is_array( X[0] ) ), "X is not a non 1-dimensional array" ##make sure X is a proper, non-degenerate array
	
	#========================================#
	# Distribution helpers 
	#========================================#

	def _categorical( self, aProb, aCategory = None ):
		"""
		Draw from categorical distribution as defined by the array of probabilities aProb. 
		"""
		if not aCategory:
			aCategory = range(len(aProb))
		return next( itertools.compress( aCategory, multinomial( 1, aProb )  ) )


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
					dist = self.hash_distribution[dist]
				except KeyError:
					raise Exception("Please provide a valid distribution")

		return array([dist.rvs( ) for _ in range(shape)])


	def randmat( self, shape = None, dist = None ):
		"""
		Returns a shape-dimensional matrix drawn IID from base distribution 
		Order: Row, Col 

		"""	

		if not shape:
			shape = self.shape

		H = self.base_distribution if not dist else self.hash_distribution[dist] #base measure 
		
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
		
		def _draw():
			z = self._categorical( pi )
			return self._eval_rvs( H[z], param[z], () )

		aOut = array([[ _draw() for _ in range(iCol)] for _ in (range(iRow) if iRow else range(1)) ])
		
		return aOut if iRow else aOut[0] 
		##returns array and not a list of arrays when there is only one element 
		##let's do what is sane, not what is technically type-proof; this is python after all.


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


	#=============================================================#
	# Spike functions 
	#=============================================================#

	def spike( self, X, strMethod = "line", sparsity = 1.0, bAdjacency = False, aArgs = [] ):
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
		fSparsity = sparsity

		if self._is_1d( X ):
			if aArgs:
				return pMethod(X, *aArgs)
			else:
				return pMethod(X) 
		else:

			aOut = [] 
			iRow, iCol = X.shape 
			abSpike = [] 

			for i in range(iRow):
				bSpike = self._categorical( [1-fSparsity,fSparsity] ) 
				abSpike.append(bSpike)
				aOut.append( ((pMethod(X[i], *aArgs) if aArgs else pMethod(X[i])) if bSpike else self.randmat( shape = iCol )) )
			aOut = array(aOut)
			A = numpy.diag( abSpike )
			return aOut if not bAdjacency else aOut, A 

	def __spike_line( self, X ):
		shape = X.shape  
		aOut = X + self.noise_distribution( self.noise_param ).rvs( shape )
		return self.pn( aOut, self.permutation_noise_param )

	def __spike_sine( self, X ):
		shape = X.shape 
		aOut = numpy.sin( X*numpy.pi ) + self.noise_distribution( self.noise_param ).rvs( shape )
		return self.pn( aOut, self.permutation_noise_param )

	def __spike_half_circle( self, X ):
		shape = X.shape  
		aOut = numpy.sqrt( 1-X**2 + self.small ) + self.noise_distribution( self.noise_param ).rvs( shape )
		return self.pn( aOut, self.permutation_noise_param )

	def __spike_circle( self, X ):
		shape = len(X)
		aIndices1 = array(range(0,shape,2))
		aIndices2 = 1+aIndices1
		aOut1 = (numpy.sqrt( 1-X[aIndices1]**2 + self.small ) + self.noise_distribution( self.noise_param ).rvs( len(aIndices1) ))
		aOut2 = -1*(numpy.sqrt( 1-X[aIndices2]**2 + self.small ) + self.noise_distribution( self.noise_param ).rvs( len(aIndices2) ))
		aOut = self.interleave( aOut1, aOut2 )
		return self.pn( aOut, self.permutation_noise_param )

	def __spike_parabola( self, X ):
		shape = X.shape
		aOut = X**2 + self.noise_distribution( self.noise_param ).rvs( shape )
		return self.pn( aOut, self.permutation_noise_param )

	def __spike_cubic( self, X ):
		shape = X.shape
		aOut = X**3 + self.noise_distribution( self.noise_param ).rvs( shape )
		return self.pn( aOut, self.permutation_noise_param )

	def __spike_log( self, X ):
		shape = X.shape
		aOut = numpy.log( 1 + X + self.big ) + self.noise_distribution( self.noise_param ).rvs( shape )
		return self.pn( aOut, self.permutation_noise_param )

	def __spike_vee( self, X ):
		shape = X.shape
		aOut = numpy.sqrt( X**2 ) + self.noise_distribution( self.noise_param ).rvs( shape )
		return self.pn( aOut, self.permutation_noise_param )

	def __spike_ellipse( self, X ):
		shape = len(X)
		aIndices1 = array(range(0,shape,2))
		aIndices2 = 1+aIndices1
		aOut1 = (numpy.sqrt( 1-0.5*X[aIndices1]**2 + self.small ) + self.noise_distribution( self.noise_param ).rvs( len(aIndices1) ))
		aOut2 = -1*(numpy.sqrt( 1-0.5*X[aIndices2]**2 + self.small ) + self.noise_distribution( self.noise_param ).rvs( len(aIndices2) ))
		aOut = self.interleave( aOut1, aOut2 )
		return self.pn( aOut, self.permutation_noise_param )
					
	def __spike_line_line( self, X ):
		shape = len(X)
		aIndices1 = array(range(0,shape,2))
		aIndices2 = 1+aIndices1
		return self.interleave( self.hash_spike_method["line"](X[aIndices1]), -1*self.hash_spike_method["line"](X[aIndices2]) )

	def __spike_x( self, X ):
		return self.hash_spike_method["line_line"]( X )

	def __spike_line_parabola( self, X):
		shape = len(X)
		aIndices1 = array(range(0,shape,2))
		aIndices2 = 1+aIndices1
		return self.interleave( self.hash_spike_method["line"](X[aIndices1]), self.hash_spike_method["parabola"](X[aIndices2]) )

	def __spike_non_coexistence( self, X ):
		pass 


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

	"""
	*** Consider different types of inputs (both continuous + categorical):
	**** Uniform random, no cluster structure, linear spikes
	**** Normal random, no cluster structure, linear spikes
	**** Uniform mixture model (clusters), linear spikes
	**** Normal mixture model (clusters), linear spikes
	**** Normal random, no cluster structure, sine spike
	"""

	def spike_synthetic_data( self, X, A = None, sparsity = None, spike_method = "parabola" ):
		"""
		Spikes X and returns a spiked array Y with association matrix Y 

		Parameters
		---------------

			X
			spike_method 

		Returns 
		--------------

			Y,B 

		Notes
		--------------

			First write this without "grouping" structure in X; then expand to take this case into account. 

		"""

		X, A = array(X),array(A)

		assert( not self._is_1d( X ) )

		if not sparsity:
			sparsity = self.sparsity_param

		pMethod = self.hash_spike_method[spike_method]

		if self._is_empty( A ):
			return self.spike( X, strMethod = spike_method, sparsity = sparsity )

		else:
			return None 


	def generate_synthetic_data( self, shape = None, sparsity = None, method = "default_normal_linear_spike" ):
		"""
		Pipeline for synthetic data generation 

		Parameters
		-------------

			X : numpy.ndarray
				If this is non-empty, uses this to produce linkage on Z; otherwise generates one from scratch 

			num_var : int 
				number of variables in the dataset 

			sparsity : float 
				portion of the dataset that is random noise 

		Returns 
		------------

			Z : numpy.ndarray 
				Dataset containing `num_var` rows and `self.shape` columns 

			A : numpy.ndarray 
				Dataset of actual associations 

		Notes
		-----------

			* Fix so that random data generation propagates in a greedy fashion. 

		"""

		pShape = shape
		fSparsity = sparsity 

		strMethod = method 
		hDP = self.hash_distribution_param
		strKeyDefault = "default"
		strKeyMixture = "mixture"

		strDist = None 
		strSpikeMethod = None

		for _strSpikeMethod in self.hash_spike_method.keys():
			if _strSpikeMethod in strMethod:
				strSpikeMethod = _strSpikeMethod 

		if not strSpikeMethod:
			raise Exception("Unknown spike method.")

		for _strDist in self.hash_distribution.keys():
			if _strDist in strMethod:
				strDist = _strDist 

		if not strDist:
			raise Exception("Unknown distribution.")

		if strKeyDefault in strMethod: ##default generation 
			pParam = hDP[strKeyDefault][strDist]
			self.set_base(strDist)
			self.set_base_param(pParam)
			return self.generate_spiked_data( shape = pShape, sparsity = fSparsity, generation_method = "randmat", spike_method = strSpikeMethod )
			

		elif strKeyMixture in strMethod: ##mixture generation 
			pParam = hDP[strKeyMixture][strDist]
			self.set_base([strDist]*len(pParam))
			self.set_base_param(pParam)
			return self.generate_spiked_data( shape = pShape, sparsity = fSparsity, generation_method = "randmix", spike_method = strSpikeMethod )


	def generate_spiked_data( self, shape = None, spike_method = None, generation_method = "randmat", sparsity = None ):
		"""
		Generates linked data and true labels
		
		Sparsity is how populated the dataset is; 1 means fully linked, 0 means everything is iid 
		"""
		
		if not shape:
			shape = self.shape 

		try:
			iRow, iCol = shape 
			num_var = iRow  
			num_samples = iCol 
		except (ValueError,TypeError):
			num_var, num_samples = shape, shape 

		assert( isinstance( num_var, int ) ), "num_var should be an integer" 

		if not sparsity:
			sparsity = self.sparsity_param

		assert( 0.0 <= sparsity <= 1 ), "sparsity parameter must be between 0 and 1" 

		if self._is_str( spike_method ):
			spike_method = [spike_method]

		elif self._is_iter( spike_method ):
			spike_method = list(spike_method)

		else:
			spike_method = self.list_spike_method 


		strGenerationMethod = generation_method 
		pGenerationMethod = None

		try:
			pGenerationMethod = getattr( self, strGenerationMethod )
		except AttributeError:
			Exception("Invalid generation method. Available options are: randmat, randvec, randmix, randnet")

		#aMethods = [getattr(self, m) for m in self.generation_methods] 
		aMethods = spike_method 
		iMethod = len( aMethods )

		#print iMethod 

		assert( 0.0 <= sparsity <= 1.0 ), "sparsity parameter must be a float between 0 and 1"
		prob_random = 1.0-sparsity ## probability of generating random adjacency 
		prob_linked = 1.0-prob_random ## probability of generating true adjacency 

		## Currently, assume that each method in aMethods is equally likely to be chosen 
		prob_method = [1.0/iMethod] * iMethod 

		## Initialize adjacency matrix 
		A = numpy.zeros( (num_var, num_var) )

		## Initialize dataset 
		X = numpy.zeros( (num_var, num_samples) )

		bool_list = [self._categorical( [prob_random, prob_linked] ) for i in range(num_var) ]
		#print bool_list  

		base_rv = array([]) 

		## Populate the X matrix and identity elements in A matrix 
		for i in range(num_var):
			A[i][i] = 1 
			if bool_list[i]: 

				if not base_rv.any():
					base_rv = pGenerationMethod( shape = num_samples )  #self.randmat( num_samples ) 
					X[i] = base_rv 
				else:
					cI = self._categorical( prob_method )

					transformed_data = self.spike( base_rv, strMethod = aMethods[cI] )
					#_, transformed_data = aMethods[cI]( shape = num_samples, rvs = base_rv )
					X[i] = transformed_data 
			else:
				X[i] = pGenerationMethod( shape = num_samples ) #self.randmat( num_samples ) 
				
		## Populate the A matrix for groups 

		linkage_list = itertools.compress( range(num_var), bool_list )

		for i,j in itertools.combinations( linkage_list, 2 ): 
			A[i][j] = 1 ; A[j][i] = 1 				
		
		return X,A


	def run( self, method = "shapes" ):
		"""
		Generic run method; currently goes through hash_methods-- needs to be made more general 
		"""
		if method == "shapes":

			self.set_noise( 0.01 )
			self.set_base("linear")
			self.set_base_param((-1,1))

			for item in self.hash_method:
				figure() 
				v,x = getattr(self, item)()
				print v
				print x 
				scatter(v,x)
		else:
			pass 


	#=============================================================#
	# Linkage helper functions   
	#=============================================================#

	def partition( self, fStart, fEnd, iSize ):
		iSize+=1 
		aLin = numpy.linspace(fStart,fEnd,iSize)
		aInd = zip(range(iSize-1),range(1,iSize+1))
		return [(aLin[i],aLin[j]) for i,j in aInd]

	def partition_of_unity( self, iSize = 2 ):
		return self.partition( 0, 1, iSize = iSize )

	def indicator( self, pArray, pInterval ):

		aOut = [] 

		for i,value in enumerate( pArray ):
			aOut.append( ([ z for z in itertools.compress( range(len(pInterval)), map( lambda x: x[0] <= value <= x[1], pInterval ) ) ] or [0] )[0] ) 
			## <= on both directions, since I am taking out the first element anyways 

		return aOut

	def threshold( self, pArray, fValue ):
		return self.m( pArray, lambda x: int(x <= fValue) )

	def classify( self, pArray, method = "logistic", iClass = 2 ):
		"""
		Classify to discrete bins using cdf of standard distributions 

		method: str 
			logistic, beta 
		"""

		hashMethod = {"logistic": logistic.cdf,
						"beta": lambda x: beta.cdf(x,2,2)} 
		
		pMethod = hashMethod["logistic"]

		if method:
			pMethod = hashMethod[method] 


		aInterval = self.partition_of_unity( iSize = iClass )

		return self.indicator( pMethod( pArray ), aInterval )

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


	#=============================================================#
	# Test Evaluation Methods 
	#=============================================================#

	def _compare( self, X, A, strMethod = "pearson"):
		pass 

	def compare_one( self, X, A, strMethod = "pearson" ):
		"""
		Compare random matrix X with known association A
		"""
		
		pMethod = strMethod  
		
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

		#### IMPORTANT
		#### I need to return the PROBABILITY VALUE OF BEING THE TRUE LABEL; 
		#### hence, if associated with pval, then return 1-pval. 

		return array(aOut)

	#=============================================================#
	# Data visualization helpers + Plotter
	#=============================================================#

	def view( self, X, A, method = "pearson" ):
		"""
		Generic `view` method 
		"""
		pass 

	def plot_roc( self, fpr, tpr, strTitle = None, astrLabel = [], strFile = None ):
		"""
		Plots the roc curve for a given set of false/true positve rates  

		Parameters
		------------

		fpr: array of float
			False Positive Rate or 1-Specificity 
		tpr: array of float
			True Positive Rate or Sensitivity 

		Returns 
		----------

		pPlot: numpy plot object 
			Pointer to the plot object 
		"""

		import pylab 
		pl = pylab 

		if self._is_meta( fpr ):
			assert( self._is_meta( tpr ) ) 
			assert( len(fpr) == len(tpr) )

		elif self._is_1d( fpr ) and self._is_1d( tpr ):
			assert( len(fpr) == len(tpr) )
			fpr, tpr = [fpr], [tpr] ##invariance 

		gLabels = itertools.cycle( astrLabel ) if astrLabel else itertools.repeat("")

		## available colors: blue, green, red, cyan, magenta, yellow, black, white 
		aColors = ["blue", "red", "green", "black", "magenta"]
		gColors = itertools.cycle( aColors ) ##generator object 

		roc_auc = [auc(f,t) for f,t in zip(fpr,tpr)]
		
		pl.figure() ##initialize figure 

		pl.plot([0, 1], [0, 1], 'k--')
		pl.xlim([0.0, 1.0])
		pl.ylim([0.0, 1.0])
		pl.xlabel('False Positive Rate')
		pl.ylabel('True Positive Rate')
		pl.title( strTitle or 'Receiver operating characteristic' )

		for i, fFPR in enumerate( fpr ):
			strColor = next(gColors)
			strLabel = next(gLabels)
			pl.plot(fFPR, tpr[i], c = strColor, label= strLabel + '(ROC = %0.2f)' % roc_auc[i])
			pl.legend(loc="lower right")
		
		if strFile:
			pl.savefig( strFile )
			return roc_auc 

		else:
			pl.show()
			return roc_auc 

	def roc( self, true_labels, prob_vec, strTitle = None, astrLabel = None, strFile = None ):
		"""
		Takes true labels and the probability vectors and calculates the corresponding fpr/tpr; return roc object

		Parameters
		------------

		true_labels: array of float
			Binary labels
		prob_vec: array of float
			Probability of being a true label 

		Returns 
		----------

		roc_auc: float
			AUC value 
		"""

		from sklearn.metrics import roc_curve, auc

		if self._is_meta( true_labels ):
			assert( self._is_meta( prob_vec ) ) 
			assert( len(true_labels) == len(prob_vec) )

		elif self._is_1d( true_labels ) and self._is_1d( prob_vec ):
			assert( len(true_labels) == len(prob_vec) )
			true_labels, prob_vec = [true_labels], [prob_vec] ##invariance 

		aRoc = array([roc_curve( t, p ) for t,p in zip(true_labels, prob_vec)])
		fpr, tpr, thresholds = aRoc[:,0], aRoc[:,1], aRoc[:,2]
		roc_auc = self.plot_roc( fpr, tpr, strTitle = strTitle, astrLabel = astrLabel, strFile = strFile )
		return roc_auc 

	def accuracy( self, true_labels, emp_labels ):
		assert( len(true_labels) == len(emp_labels) )
		iLen = len(true_labels)
		return sum( self.md( true_labels, emp_labels, lambda x,y: int(x==y) ) )*(1/float(iLen))

	def accuracy_with_threshold( self, true_labels, prob_vec, fThreshold = 0.05 ):
		if not fThreshold:
			fThreshold = self.q 
		return self.accuracy( true_labels, self.threshold( prob_vec, fThreshold ) )