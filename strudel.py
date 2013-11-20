"""
STRUDEL: Synthetic Rudimentary Data Exploration 

Synthetic data generation for common exploration tasks in microbial data analysis 
"""

import scipy
import numpy 
import csv 
import sys 

## namespace for distributions 
from scipy.stats import invgamma, norm, uniform, logistic 

def generate_clustered_data( num_clusters = 3, num_children = 3, num_examples = 20):
	"""
	Input: Specify distributions 
	
		num_clusters  
		num_children <- children that share prior for each cluster 
		num_examples 

	Output: p x n matrix, with clustered variables 
	"""

	aOut = [] 

	atHyperVar = [(1,1), (3,1), (5,0.5)] #hyperparameters for variance 

	atHyperMean = [(0,0.1), (0,0.25), (0,0.5)] #hyperparameters for mean 

	for k in range(num_clusters):
		alpha, beta = atHyperVar[k]
		prior_mu, prior_sigma = atHyperMean[k]

		for j in range(num_children):
			sigma = invgamma.rvs(alpha)
			mu = norm.rvs(loc=prior_mu,scale=prior_sigma)

			iid_norm = norm.rvs( loc=mu, scale=sigma, size=num_examples)

			aOut.append( iid_norm )

	return numpy.array(aOut)

def partition_of_unity( iSize = 2 ):
	iSize+=1 
	aLin = numpy.linspace(0,1,iSize)
	aInd = zip(range(iSize-1),range(1,iSize+1))
	return [(aLin[i],aLin[j]) for i,j in aInd]

def indicator( pArray, pInterval ):
	from itertools import compress 

	aOut = [] 

	for i,value in enumerate( pArray ):
		aOut.append( [ z for z in compress( range(len(pInterval)), map( lambda x: x[0] <= value < x[1], pInterval ) ) ][0] ) 

	return aOut

def classify_by_logistic( pArray, iClass = 2 ):
	
	aInterval = partition_of_unity( iSize = iClass )

	return indicator( logistic.cdf( pArray ), aInterval )

def generate_linkage( num_clusters = 3, num_children = 3, num_examples = 20):
	"""
	Input: 
	Output: Tuple (predictor matrix, response matrix) 
	"""

	aBeta = uniform.rvs(size=num_clusters)

	iRows = num_clusters * num_children
	iCols = num_examples 
	
	predictor_matrix = generate_clustered_data( num_clusters, num_children, num_examples )
	raw = predictor_matrix[:]
	response_matrix = []

	while raw.any():
		
		pCluster, raw = raw[:num_clusters], raw[num_clusters:]
		response_matrix.append( classify_by_logistic( numpy.dot( aBeta, pCluster ) ) )

	return predictor_matrix, response_matrix

if __name__ == "__main__":

	predictor, response = generate_linkage()

	csvw = csv.writer( sys.stdout, csv.excel_tab )

	csvw.writerow( ["#Predictor"] )

	for item in predictor:
		csvw.writerow( item )

	csvw.writerow( ["#Response"])

	for item in response:
		csvw.writerow( item )


