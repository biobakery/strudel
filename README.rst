=============================================================
Strudel: STratified RUdimentary Data ExpLoration 
Tasty recipes for exploratory data analysis 
=============================================================

..  This document follows reStructuredText syntax and conventions.
	You can compile this file to a PDF or HTML document.
	For instructions on how to do so, visit the reStructeredText webpage
	(http://docutils.sourceforge.net/rst.html).

Authors 
 Yo Sup Moon, George Weingart, Curtis Huttenhower  

.. To Do 
	* unified name space for common statistical distributions 
	* a way to programmatically wrap dependencies together 
	* a nice interface for generating graphical models in picture format 
	* a principled way of writing down the config file in the most intuitive way possible 

..	Important notions 
	* conditional (in)dependence 
	* IID 
	* Literate programming (Knuth)-- make functions and documentation as readable to humans as possible 

Getting Started
============================================ 

Summary
--------------------------------------------

Strudel (Stratified Rudimentary Data Exploration) is a python package designed to provide a modular, expandable, and easy-to-use interface for exploratory and structural data analysis. At the core philosophy of Strudel is the belief that the data should speak for itself; that is, the data and the data alone should motivate the use of particular computational, statistical, and graphical models. Complicated data structures -- data that is high-dimensional, heterogeneous (consisting of discrete, lexical, continuous components), and noisy hamper the scientist's effort to make sense of its underlying latent structure. Strudel aims to be a general-purpose tool to aid in reproducible, high-throughput exploratory data analysis that is both simple to use and highly expandable to create arbitrarily complex data exploration pipelines. 

Strudel is designed with scalability in mind: with it users can create a stratified, efficient search over the data space in limited (just a couple of cores) to expansive (hundreds or thousands of cores) computational settings. Furthermore, it is designed to combat the loss of statistical power in multiple hypothesis testing schemes through dimensionality reduction: hierarchical and consensus-based clustering techniques are used when possible to curb for false discoveries in sparse data (p >> n) settings and to aid in quick feature selection for more advanced analysis. Strudel’s modular design allows it to incorporate various statistical testing pipelines, including HAllA (http://huttenhower.sph.harvard.edu/halla), CAKE (http://huttenhower.sph.harvard.edu/cake), and SparseDOSSA (http://huttenhower.sph.harvard.edu/sparsedossa). 

As it currently stands, Strudel offers these functionalities: (1) random variable generation, (2) random matrix generation, (3) random mixture generation, (4) bayesian network generation, (5) linear and nonlinear spikes for association testing, (6) categorical data generation via coupling through linkage functions, and (7) helper functions for pipelining various association tests and visualizations. Future plans include automated parameter estimation from test data and improved methods for incorporating and visualizing summary statistics. 


Operating System  
--------------------------------------------

* Supported 
	* Ubuntu Linux (>= 12.04) 
	* Mac OS X (>= 10.7)

* Unsupported 
	* Windows (>= XP) 

Dependencies 
--------------------------------------------

* Required
	* Python (>= 2.7)
	* Numpy (>= 1.7.1)
	* Scipy (>= 0.12) 
	* Scikit-learn (>=0.13)  
	* rpy (>=2.0)

* Recommended Tools for documentation 
	* Docutils
	* itex2MML


Getting Strudel
--------------------------------------------

Strudel can be downloaded from its bitbucket repository: https://github.com/biobakery/strudel

Example Usage 
============================================ 

