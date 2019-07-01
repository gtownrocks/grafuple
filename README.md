# Grafuple


Description
=========== 

Random forest two-class classifier for determining malciousness of .NET binaries.
             The model is trained on graph-structured data obtained through decompilation.  .NET
             binaries are scored by first decompiling, vectorizing the result, and feeding the resulting
             vector into the predict method of the random forest.

Usage
===========

`grafuple` has the following dependencies:

 + Mono 4.0.2
 + CLRParser 1.1.6.0000 (to be open-sourced)
 + RandomForest model file (same version as module)


### Uniform Measure
#### Training and Scoring
```
virtualenv -p python3 env
source env/bin/activate
python3 uniform.py
```

### PageRank Measure
```
virtualenv -p python3 env
source env/bin/activate
```
#### Training
```
python3 train FileName vectors_dir model_dir
```
#### Scoring
```
>>> pagerankrandomforest = PageRankRandomForest()
>>> score = pagerankrandomforest.predict(path_to_file)
```

Model Details
============

## Parameters
	rfc: RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
            max_depth=None, max_features='sqrt', max_leaf_nodes=None,
            min_impurity_split=6.55894113067e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0,
            n_estimators=592, n_jobs=1, oob_score=False, random_state=0,
            verbose=0, warm_start=False)

## Performance
	Accuracy on Training Set: 0.99952222
	Accuracy on Validation Set: 0.98172570
	Accuracy on Test Set: 0.98272994

             precision    recall  f1-score   support

     Benign     0.9788    0.9937    0.9862    696827
    Malware     0.9894    0.9647    0.9769    424420

	avg / total     0.9828    0.9827    0.9827   1121247

	False Positive Rate: 0.0105
	False Negative Rate: 0.0172

Data
=====
## Overview 
For a given file, decompilation produces several types of graph-structured data stored as json.  As of 
      6/9/19 this information includes a single function-to-function graph describing how the set of functions
      contained in the file are connected, a data flow graph or each function, an abstract syntax tree for 
      each function, and a graph, called the sdfg, obtained by connecting all paths through the abstract
      syntax tree.


Vectorization
=============
## Overview
As of 6/9/19 the only data being used for vectorization is the sdfg.  The sdfg, referred to
                here as G, is smoothed into the corresponding google matrix so that pagerank can be computed. 
                Pagerank acts as a measure (probability distribution) over the set of nodes of G.  We then
                consider functions of the form f:Vert(G)-->R (real numbers) and compute E[f|_S] for various
                sets S\in Vert(G).  This results in a vector (E[f_i|S_j]) for a set of functions {f_i} and 
                set of subsets S_j\subset Vert(G).  We obtain a single vector for the file by computing 
                E[E[f_i|S_j]] and Std[E[f_i|S_j]] for each (i,j) pair, where the expectation and standard
                deviation is taken over the set of sdfg's in the file.


## Example

	G.nodes() = {"1":{'type':'Call','arguments':[v1,v2]},2:{'type':'FieldReference','value':4},
		      "3":{'type':'Call','arguments':[v9]},4:{'type':'FnPtrObj','name':'doggy'}}
	G.edges() = [("1","2"),("1","3"),("2","4"),("3","4")]

	Consider the following functions:
	f(node)=(node['type']=='Call')*len(node['arguments'])
	g(node)=(node['type']=='FieldReference')*node['value']

	This yields: 
	f = ("1"=2,"2"=0,"3"=1,"4"=0)
	g = ("1"=0,"2"=4,"3"=0,"4"=0)	

	P (PageRank) = {'1':0.1257,'2':0.1880,'3':0.1880,'4':0.4981}

	Let G_p = {v\in G.nodes() | PageRank(v)<p}

	E[f|G_20] = <f|G_20,P|G_20> = <(1),(0.1257)> = 0.1257
	E[f|G_40] = <f|G_40,P|G_40> = <(2,0,1),(0.1257,0.1880,0.1880)> = 0.4396
	E[f|G_100] = <f|G_100,P|G_100> = <(2,0,1,0),(0.1257,0.1880,0.1880,0.4981)> = 0.4396

	E[g|G_20] = <g|G_20,P|G_20> = <(0),(0.1257)> = 0
	E[g|G_40] = <g|G_40,P|G_40> = <(0,4,0),(0.1257,0.1880,0.1880)> = 0.7520
	E[g|G_100] = <g|G_100,P|G_100> = <(0,4,0,0),(0.1257,0.1880,0.1880,0.4981)> = 0.7520

	The result is a vectorization of G: v_G:=(0.1257,0.4396,0.4396,0,0.7520,0.7520).  Each decompiled .NET file F contains many graphs G, 
	and we obtain a vectorization of the file as a whole by computing mean and standard deviation of each coordinate in v_G.  
	Concretely, F_features = {mean{(v_G)_i}} union {std{(v_G)_i}}.     





          
		
	

