# /usr/bin/env python
# coding=utf-8
from mock import patch
import time
from nose import tools as nt
import networkx as nx
import numpy as np
from grafuple.pagerank.graphdualvs import GraphDualVS


def test_vectorize():
    """ Runs the unit test for the vectorization routine.  Phoney graph, phony data, all
        in the same format as output of CLRParserApp.exe.

    Args:
        None

    Returns:
        None

    """

    G = dict()

    # diamond shaped graph
    G['edges'] = {'0':['1','2'],'1':['3'],'2':['3']}

    # graph with dictionary node data
    G['nodes'] = {'0':{
                    'type':'dog',
                    'value':1
                        },
                    '1':{
                    'type':'dog',
                    'value':10
                        },
                    '2':{
                    'type':'cat',
                    'value':100
                        },
                    '3':{
                    'type':'mouse',
                    'value':1000
                        }
                 }

    # compute pagerank-based measure
    measure = nx.pagerank(nx.DiGraph(G['edges']))

    # define partition of vertices based on pagerank values
    partition = np.array([20,40,60,80,100])

    # create dictionary of node types - each (key,value) pair corresponds to (node type, functions which are potentially non trivial on nodes of that type)
    Functions = {'dog':[],'cat':[],'mouse':[]}

    # define set of functions which are defined on the node set based on type, and add these functions to Function dict
    def f_dog(*args):
        return 0.5*args[0]['value']
    Functions['dog'].append(f_dog)

    def g_dog(*args):
        return args[0]['value']
    Functions['dog'].append(g_dog)

    def f_cat(*args):
        return 0.5*args[0]['value']
    Functions['cat'].append(f_cat)

    def g_cat(*args):
        return args[0]['value']
    Functions['cat'].append(g_cat)

    def h_mouse(*args):
        return args[0]['value']
    Functions['mouse'].append(h_mouse)

    # compute functions and smoothings
    G_star = GraphDualVS(Functions,partition,'type')
    f = G_star.compute_dual_vecs(G)
    F = G_star.lebesgue_smooth_dual_vecs(G,measure)

    # output values checked for correctness 
    compute_dual_vecs = f == {'cat_g_cat': {'1': 0.0, '0': 0.0, '3': 0.0, '2': 100},
                              'dog_g_dog': {'1': 10, '0': 1, '3': 0.0, '2': 0.0}, 
                              'dog_f_dog': {'1': 5.0, '0': 0.5, '3': 0.0, '2': 0.0}, 
                              'mouse_h_mouse': {'1': 0.0, '0': 0.0, '3': 1000, '2': 0.0}, 
                              'cat_f_cat': {'1': 0.0, '0': 0.0, '3': 0.0, '2': 50.0}}
    lebesgue_smooth_dual_vecs_test = F == {'mouse_h_mouse_40': 0.0,
                                           'mouse_h_mouse_60': 0.0, 
                                           'dog_g_dog_80': 2.0969391988782391, 
                                           'mouse_h_mouse_20': 0.0, 
                                           'dog_g_dog_40': 0.13750444644841497,
                                           'dog_g_dog_60': 0.13750444644841497, 
                                           'dog_g_dog_20': 0.13750444644841497, 
                                           'cat_g_cat_80': 19.594347524298243, 
                                           'dog_f_dog_100': 1.0484695994391195, 
                                           'cat_g_cat_40': 0.0,
                                           'mouse_h_mouse_80': 0.0,
                                           'cat_g_cat_60': 0.0,
                                           'cat_f_cat_80': 9.7971737621491215,
                                           'cat_f_cat_100': 9.7971737621491215,
                                           'dog_g_dog_100': 2.0969391988782391, 
                                           'cat_g_cat_20': 0.0,
                                           'dog_f_dog_40': 0.068752223224207487,
                                           'dog_f_dog_60': 0.068752223224207487, 
                                           'cat_f_cat_20': 0.0,
                                           'cat_f_cat_40': 0.0,
                                           'cat_g_cat_100': 19.594347524298243,
                                           'dog_f_dog_20': 0.068752223224207487,
                                           'mouse_h_mouse_100': 0.0,
                                           'cat_f_cat_60': 0.0,
                                           'dog_f_dog_80': 1.0484695994391195}

    nt.assert_true(compute_dual_vecs)
    nt.assert_true(lebesgue_smooth_dual_vecs_test)



def test_data():
    """ Unit tests

        Args:
            None

        Returns:
            None

    """

    G = dict()
    G['edges'] = {'0':['1','2'],'1':['3'],'2':['3']}


    G['nodes'] = {'0':{
        'type':'dog',
        'value':1
    },
        '1':{
            'type':'dog',
            'value':10
        },
        '2':{
            'type':'cat',
            'value':100
        },
        '3':{
            'type':'mouse',
            'value':1000
        }
    }

    measure = nx.pagerank(nx.DiGraph(G['edges']))
    # partition = np.array([0.01,0.15,0.2,1.0])
    partition = np.array([20,40,60,80,100])
    Functions = {'dog':[],'cat':[],'mouse':[]}

    def f_dog(*args):
        return 0.5*args[0]['value']
    Functions['dog'].append(f_dog)

    def g_dog(*args):
        return args[0]['value']
    Functions['dog'].append(g_dog)

    def f_cat(*args):
        return 0.5*args[0]['value']
    Functions['cat'].append(f_cat)

    def g_cat(*args):
        return args[0]['value']
    Functions['cat'].append(g_cat)

    def h_mouse(*args):
        return args[0]['value']
    Functions['mouse'].append(h_mouse)

    # compute functions and smoothings
    G_star = GraphDualVS(Functions,partition,'type')
    f = G_star.compute_dual_vecs(G)
    F = G_star.lebesgue_smooth_dual_vecs(G,measure)

    # tests
    compute_dual_vecs = f == {'cat_g_cat': {'1': 0.0, '0': 0.0, '3': 0.0, '2': 100}, 'dog_g_dog': {'1': 10, '0': 1, '3': 0.0, '2': 0.0}, 'dog_f_dog': {'1': 5.0, '0': 0.5, '3': 0.0, '2': 0.0}, 'mouse_h_mouse': {'1': 0.0, '0': 0.0, '3': 1000, '2': 0.0}, 'cat_f_cat': {'1': 0.0, '0': 0.0, '3': 0.0, '2': 50.0}}
    lebesgue_smooth_dual_vecs_test = F == {'mouse_h_mouse_40': 0.0, 'mouse_h_mouse_60': 0.0, 'dog_g_dog_80': 2.0969391988782391, 'mouse_h_mouse_20': 0.0, 'dog_g_dog_40': 0.13750444644841497, 'dog_g_dog_60': 0.13750444644841497, 'dog_g_dog_20': 0.13750444644841497, 'cat_g_cat_80': 19.594347524298243, 'dog_f_dog_100': 1.0484695994391195, 'cat_g_cat_40': 0.0, 'mouse_h_mouse_80': 0.0, 'cat_g_cat_60': 0.0, 'cat_f_cat_80': 9.7971737621491215, 'cat_f_cat_100': 9.7971737621491215, 'dog_g_dog_100': 2.0969391988782391, 'cat_g_cat_20': 0.0, 'dog_f_dog_40': 0.068752223224207487, 'dog_f_dog_60': 0.068752223224207487, 'cat_f_cat_20': 0.0, 'cat_f_cat_40': 0.0, 'cat_g_cat_100': 19.594347524298243, 'dog_f_dog_20': 0.068752223224207487, 'mouse_h_mouse_100': 0.0, 'cat_f_cat_60': 0.0, 'dog_f_dog_80': 1.0484695994391195}

    nt.assert_true(compute_dual_vecs)

    nt.assert_true(lebesgue_smooth_dual_vecs_test)
