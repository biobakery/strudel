=============================================================
STRUDEL: SynThetic RUdimentary Data ExpLoration 
=============================================================

..  This document follows reStructuredText syntax and conventions.
	You can compile this file to a PDF or HTML document.
	For instructions on how to do so, visit the reStructeredText webpage
	(http://docutils.sourceforge.net/rst.html).

Authors 
 Yo Sup Moon 

Dependencies 
==================

Requires:

* Python 
* Numpy/Scipy 

To Do 
=====================

* unified name space for common statistical distributions 
* a way to programmatically wrap dependencies together 
* a nice interface for generating graphical models in picture format 
* a principled way of writing down the config file in the most intuitive way possible 

Important notions 

* conditional (in)dependence 
* IID 

Notes 

* functions should be vectorized so vectors can be passed in as arguments for simplicity ::

	Normal( [(0,1),(0,2), ... ] ) -> Draws from normal of those parameter pairs 

Example :: 

	[Dist(params) for _ in K]                                                                                                                                                                            
	
