# References --------------------------------------------------------------------------------

# Pipeline Examples
# https://civisanalytics.com/blog/data-science/2016/01/06/workflows-python-using-pipeline-gridsearchcv-for-compact-code/
# http://scikit-learn.org/stable/auto_examples/model_selection/grid_search_text_feature_extraction.html

# Model Persistence
# http://scikit-learn.org/stable/modules/model_persistence.html


# Randomized Search vs GridSearch
# http://scikit-learn.org/stable/auto_examples/model_selection/randomized_search.html#example-model-selection-randomized-search-py

# Feature Union
# http://scikit-learn.org/stable/auto_examples/feature_stacker.html#example-feature-stacker-py


# # API Reference
# # http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
 
# # Math Reference
# # http://scikit-learn.org/stable/modules/ensemble.html#forest

# http://stats.stackexchange.com/questions/59630/test-accuracy-higher-than-training-how-to-interpret


# http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV
# http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline.get_params
# Required Modules ------------------------------------------------------------------------

from __future__ import division
from sklearn.decomposition import PCA, RandomizedPCA as RPCA 
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_selection import SelectKBest,chi2,f_classif
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.metrics import classification_report, recall_score, precision_score
from sklearn.datasets.samples_generator import make_blobs
from sklearn.model_selection import train_test_split
import numpy as np
from operator import itemgetter
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from sklearn.neural_network import BernoulliRBM
from sklearn.externals import joblib
import pickle
import time
from multiprocessing import cpu_count
from multiprocessing.pool import Pool
import pandas as pd
from pandas import DataFrame
import sys
from itertools import chain
from itertools import combinations
from tabulate import tabulate
from sklearn.model_selection import RandomizedSearchCV
from sys import path
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import learning_curve
from sys import path
import pickle
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

# from os import path
# path.append(path.dirname(path.dirname(path.abspath(__file__))))

# Source and Destination Folders ------------------------------------------------------------------

# Call script as:
#       python3 train FileName /sourceDir/ /outdir/

name = sys.argv[1]
sourceDir = sys.argv[2]
outdir = sys.argv[3]

# Helper Functions ----------------------------------------------------------------------

# Utility function to report best scores
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

# Compute powerset of a given set
def powerset(iterable,l):
    s = list(iterable)
    c = list(chain.from_iterable(combinations(s,r) for r in range(len(s)+1)))
    return [list(x) for x in c if len(x)>l-1]

# Compute estimation time
def sec_to_dhms(s):
    m,s=divmod(s,60)
    h,m=divmod(m,60)
    d,h=divmod(h,24)
    return '%d Day, %d Hours, %d Minutes, %d Seconds' % (d,h,m,s)

# Compute false negative rate
def false_negative_rate(y_true,y_pred):

    recall = recall_score(y_true,y_pred)

    return 1-recall

# Compute false positive rate
def false_positive_rate(y_true,y_pred):

    num_true_pos = y_true.sum()
    num_true_neg = len(y_true)-num_true_pos
    precision = precision_score(y_true,y_pred)

    return float(num_true_pos*((1-precision)/precision))/(num_true_pos*((1-precision)/precision)+num_true_neg)


# Prepare Data -----------------------------------------------------------------------------

# load main data
df = joblib.load(sourceDir+name)

df = df.iloc[np.random.permutation(len(df))]

# grab names of raw features
feature_names = df.columns.values
features = [item for item in feature_names if item not in set(['aggressive_label','TIME'])]

# grab labels and type to int
y = df['aggressive_label'].values
y.astype(int)

# grab raw features                            
X = df[features]                                                                             

# write names of raw features to terminal
#for feature in features: print feature   

# split raw data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.005, random_state=0)

# test if train_test_split succeeded
print 'Train Set Equals Test Set: %s' % str(X_train.equals(X_test))


# Model Estimation ----------------------------------------------------------------------------------

# instantiate core model object 
rfc = RFC(random_state=0)

# instantiate feature engineering objects
pca = PCA(svd_solver='randomized')

# build pipeline
selection_chi2 = SelectKBest(chi2)
selection_f_classif = SelectKBest(f_classif)
select = SelectKBest()
combined_features = FeatureUnion([('pca',pca),('univ_select_chi2',selection_chi2),
                                  ('univ_select_f_classif',selection_f_classif),('select',select)])

# instantiate pipeline object
pipeline = Pipeline([('rfc',rfc)])
n_features = len(features)
param_dist = dict(rfc__n_estimators = sp_randint(300,800),
                  rfc__criterion = ['gini','entropy'],
                  rfc__max_features = ['log2',12,'sqrt'],
                  rfc__max_depth = [None,100],
                  rfc__min_samples_split = [2,10],
                  rfc__min_samples_leaf = [1,5,10],
                  rfc__min_weight_fraction_leaf = [0],
                  rfc__max_leaf_nodes = [None],
                  rfc__min_impurity_split = sp_uniform(1e-7,1e-4),
                  rfc__bootstrap = [True],
                  rfc__oob_score = [False,True],
                  rfc__warm_start = [False])
             
t0 = time.time()

# instantiate model object
grid_search = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=25, cv=7, random_state=0, verbose=2, n_jobs=12, refit=True)

# train model 
grid_search.fit(X_train,y_train)

# save model to disk
joblib.dump(grid_search, outdir + 'RandomForest' + '.pkl')

# score model on test set
y_true, y_pred = y_test, grid_search.predict(X_test)

# score model on training data to get fair comparison between train and test accuracy
y_train_pred = grid_search.predict(X_train)

#training time
train_time = sec_to_dhms(time.time()-t0)

# Write Model Results to Disk and to Terminal ---------------------------------------------------------------

# write accuracy to terminal
print 'Random Forest Accuracy on Training Set: %1.8f' % (sum(y_train==y_train_pred)/y_train.shape[0])
print 'Random Forest Accuracy on Validation Set: %1.8f' % (grid_search.best_score_)
print 'Random Forest Accuracy on Test Set: %1.8f' % (sum(y_true==y_pred)/y_true.shape[0])
print 'Training Time: %s' % (train_time)
print 'Number of Total Observations: %1.0f\n' % X.shape[0]
print 'False Positive Rate: %1.8f' % (false_positive_rate(y_true,y_pred))
print 'False Negative Rate: %1.8f' % (false_negative_rate(y_true,y_pred))


# grab parameters of best model out of all that were trained
best_model = grid_search.best_estimator_
best_params = best_model.get_params(deep=True)
#best_forest = pipeline.named_steps['rfc']

#values = sorted(zip(features,best_forest.feature_importances_),key=lambda x: x[1] * -1)
#headers = ['Feature','Score']
#t = tabulate(values,headers,tablefmt='plain')


# save model results to text file
target_names = ['Benign', 'Malware']
s = classification_report(y_true, y_pred, target_names=target_names, digits=4)
with open('/home/mslawinski/Model/modeling_report.txt','a') as myfile:
    myfile.write('\n')
    myfile.write('Total Number of Observations: %1.0f\n' % X.shape[0])
    myfile.write('Random Forest: '+' '.join(features)+'\n')
    myfile.write('Accuracy on Training Set: %1.8f\n' % (sum(y_train==y_train_pred)/(y_train.shape[0])))
    myfile.write('Accuracy on Validation Set: %1.8f\n' % (grid_search.best_score_))
    myfile.write('Accuracy on Test Set: %1.8f\n' % (sum(y_true==y_pred)/(y_true.shape[0])))
    myfile.write('False Positive Rate: %1.8f\n' % (false_positive_rate(y_true,y_pred)))
    myfile.write('False Negative Rate: %1.8f\n' % (false_negative_rate(y_true,y_pred)))
    myfile.write(s)
    myfile.write('\n')
    myfile.write('Training Time: %s\n' % train_time)
    myfile.write('RandomizedGridSearch Best Model:\n')

    
    for key in best_params.keys():
        myfile.write('%s: %s\n' % (key,best_params[key]))

# write best model parameters to terminal
for key in best_params.keys():
    print '%s: %s\n' % (key,best_params[key])



















