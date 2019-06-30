import ujson as json
import os
import numpy as np
import hashlib
import time
import glob
from multiprocessing import Pool,cpu_count,Manager
import joblib
from functions import functions 
from pandas import DataFrame,Series
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score


np.random.seed(1)


def get_sdfgs(d):

    graphs = []
    keys = sorted(list(d['functions'].keys()))
    for key in keys:
        if 'sdfg:json' in d['functions'][key]:
            x = d['functions'][key]['sdfg:json']
            graphs.append(x)
    return graphs


def vectorize_g(g,functions):

    function_values = {k:[] for k in list(functions.keys())}
    vec = {k:0 for k in list(functions.keys())}

    nodes = g['nodes']
    if nodes:

        for node_key in nodes:
            node = nodes[node_key]

            for f_name,f in functions.items():
                function_values[f_name].append(f(node))

    for f_name in list(vec.keys()):
        values = function_values[f_name]
        vec[f_name] = np.mean(values) if values else 0

    return vec


def vectorize_file(d,functions):

    total_vec = {k:[] for k in list(functions.keys())}
    graphs = get_sdfgs(d)
    for g in graphs:
        vec = vectorize_g(g,functions)

        for fn_name in list(functions.keys()):
            total_vec[fn_name].append(vec[fn_name])

    vec = {}
    vec.update({'%s_%s' % (fn_name,'mean'):np.mean(total_vec[fn_name]) if total_vec[fn_name] else 0 for fn_name in list(functions.keys())})
    vec.update({'%s_%s' % (fn_name,'std'):np.std(total_vec[fn_name])  if total_vec[fn_name] else 0  for fn_name in list(functions.keys())})

    return vec



def load_single(path):

    with open(path) as f:
        text = f.read()
        d = json.loads(text)

    return d


def dump_single(vec,name,save_dir):

    joblib.dump(vec,os.path.join(save_dir,name))


def vectorize_from_path(args):

    path,functions,save_dir = args
    vec = vectorize_file(load_single(path),functions)
    name = (path.split('/')[-1]).split('.')[0] + '.jlb'
    dump_single(vec,name,save_dir)


def vectorize_parallel(samples_dir,vectors_dir,functions):

    paths = glob.glob('%s*.json' % samples_dir)
    with Pool(cpu_count()) as p:
        p.map(vectorize_from_path,zip(paths,[functions for p in paths],[vectors_dir for p in paths]))



def load_vecs(args):

    mpd,file = args
    name,ext = (file.split('/')[-1]).split('.')

    if ext=='jlb':
        try:
            d = joblib.load(file)

            key = name.upper()
            d['label'] = labels[key]
            mpd[key] = d
        except Exception:
            pass


# ==================================================== Vectorize ==================================================

# vectorize
#vectorize_parallel('../data/raw/','../vectors/',functions)
#vectorize_parallel('../../../../../../media2/ephemeral2/data/','../vectors/',functions)
# ==================================================== Feature Table ==================================================

# create table 
labels = joblib.load('../data/labels/labels_dict.jlb')
files = glob.glob('../vectors/*jlb')#[:100000]
mpd = Manager().dict()
pool = Pool(cpu_count()-1)
pool.map(load_vecs,[[mpd,file] for file in files])
pool.close() 
pool.join()
mpd = dict(mpd)
vectors = DataFrame.from_dict(mpd,orient='index')
vectors = vectors.sample(frac=1)
#print(vectors.head(20))
vectors.replace([np.inf, -np.inf],0,inplace=True)
vectors.fillna(value=0.,inplace=True)
joblib.dump(vectors,'./vecs.jlb')
import time
#for c in vectors.columns:
#  print(vectors[c].min(),vectors[c].max())
#  time.sleep(0.4)

#vectors = joblib.load('./vecs.jlb')
# ==================================================== Train Model ==================================================

# grab names of raw features
feature_names = vectors.columns.values
features = [item for item in feature_names if (item not in set(['label']))]# and ('H' not in item)]

# grab labels and type to int
y = vectors['label'].values
y.astype(int)

# grab raw features
X = vectors[features]

# split raw data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

# from sklearn.ensemble import RandomForestClassifier
params = {'n_estimators':480,
          'n_jobs':cpu_count(),
          'random_state':1,
          'max_leaf_nodes':None,
          'min_samples_leaf':1,
          'warm_start':False,
          'min_weight_fraction_leaf':0,
          'oob_score':False,
          'min_samples_split':2,
          'criterion':'gini',
          'min_impurity_split':2.09876756095e-5,
          'max_depth':None,
          'bootstrap':True,
          'max_features':'sqrt'}

model = RandomForestClassifier(**params)
model.fit(X_train,y_train)



# ==================================================== Score Model ==================================================


y_true,y_pred = y_test,model.predict(X_test) 
y_train,y_train_pred = y_train,model.predict(X_train)


# FPR FNR
recall = recall_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
n_P = sum(y)
n_N = len(y) - sum(y)
FPR = (recall*n_P*(1/precision - 1))/n_N
FNR = 1 - recall


print('\n')
s = classification_report(y_true,y_pred,digits=4,target_names=['Malicious','Benign'])
with open('./modeling_report.txt','w') as fif:
    fif.write('\n')
    fif.write(s)
    fif.write('\nTrain Accuracy %0.6f\n' % (sum(y_train==y_train_pred)/y_train.shape[0]))
    fif.write('Test Accuracy %0.6f\n' % (sum(y_test==y_pred)/y_test.shape[0]))

    fif.write("FPR: %f, FNR: %f\n" % (FPR, FNR))
    fif.close()

print('\n')


print(s)
print('Train Accuracy %0.2f' % (sum(y_train==y_train_pred)/float(y_train.shape[0])))
print('Test Accuracy %0.2f' % (sum(y_test==y_pred)/y_test.shape[0]))
print("FPR: %f, FNR: %f" % (FPR, FNR))












