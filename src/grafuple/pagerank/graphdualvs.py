import numpy as np
import time
import ujson as json
from pandas import DataFrame


class GraphDualVS(object):
    """ Encapsulates methods for computing and smoothing of vectors in Hom(Vert(G),R), 
        where G is a graph and R is the real numbers.

    """

    def __init__(self,Functions,partition,label_name):
        """ GraphDual class constructor.

            Args:
                Functions (dict): dictionary containing functions to by applied to the vertex set of a graph G
                partition (numpy array): percentiles with which to partition the probability measure of the input graph G
                label_name (str): string giving the node attribute which acts as the node label 

            Returns:
                None

        """

        # dictionary containing dual vectors along with node tags (which functions to apply on which nodes)
        self.Functions = Functions

        # precompute smoothed function keys
        self.Results_keys = [key+'_'+str(f).split(' ')[1] for key in Functions.keys() for f in Functions[key]]

        # partition of [0,1] for lebesgue integral
        self.partition = partition

        # precompute smoothed function_partition_value keys
        self.frame_keys = [key+'_'+str(j) for key in self.Results_keys for j in partition]

        # node attribute, the values of which correspond to the set of node tags in Functions
        self.label_name = label_name

    
    
    def compute_dual_vecs(self,G):
        """ applies the set of dual vectors f in F to Vert(G) 

            Args:
                G (dict): keys: 'nodes', 'edges'. G['nodes'] gives dictionary of nodes and their attributes. G['edges'] is the edge dictionary

            Returns:
                Results (dict): contains values of functions on Vert(G) - keys are named as functionClass_functionName

        """

        # each key corresponds to a node type - all functions in Functions[fn_key] will be applied to each node with label_name==fn_key
        fn_keys = self.Functions.keys()

        # given graph G, create dictionary to hold each f(Vert(G)) - has the form {node_1:0,node_2:0,...}
        nodes = G['nodes']
        node_keys = nodes.keys()
        num_node_keys = len(node_keys)
        zero_dict = dict(zip(node_keys,np.zeros(num_node_keys)))

        # instantiate dictionary which holds the full set of vectors f(Vert(G))
        Results_keys = self.Results_keys
        Results = dict(zip(Results_keys,[zero_dict.copy() for i in range(len(Results_keys))]))

        # iterate through set of nodes and for each node type apply corresponding functions
        for node_key in node_keys:

            # grab node dictionary - contains label_name and other attributes
            node = nodes[node_key]

            # grab label_name value - corresponds to function category in Functions
            fn_key = node[self.label_name]
  
            # essentially a try block - might encounter new node type for which there is no function
            if fn_key in fn_keys:

                # apply each function appropriate for this node
                for f in self.Functions[fn_key]:

                    # Results key is combination of function category (fn_key) and specific function f
                    Results[fn_key+'_'+str(f).split(' ')[1]][node_key] = f(node)

        # just return value instead of calling self.Results = Results so that class can be static (picklable)
        return Results

    
    def lebesgue_smooth_dual_vecs(self,G,measure):
        """ Smooth the vectors f:Vert(G)-->R by taking lebesgue antiderivatives over [0,1] with measure measure

            Args:
                G (dict): keys: 'nodes', 'edges'. G['nodes'] gives dictionary of nodes and their attributes. G['edges'] is the edge dictionary
                measure (dict): dictionary with pairs (node_key,p(node_key)) where p is the probability measure on Vert(G)

            Returns:
                F (dict): contains smoothed versions of the functions in self.Functions

        """

        # compute values f:Vert(G)-->R for all f in Vert(G)^* := Hom(Vert(G),R)
        Results = self.compute_dual_vecs(G)
        Results_keys = self.Results_keys
    
        # probability measure of Vert(G) - usually given by PageRank with teleportation      
        p = measure
        Results['PR'] = p 
      
        # instantiate dataframe to make sure Results.values and p.values are key-aligned
        df = DataFrame(Results)
        data = df[Results_keys].values

        # create 3-tensor to hold len(partition) copies of the columns in df
        partition = np.percentile(np.array(measure.values()),self.partition)
        L = len(self.partition)
        data_3d = np.repeat(data[:, :, np.newaxis], L, axis=2)

        # recover measure in correct node order (node order is key-aligned with the columns f(Vert(G)-->R) in df)
        p = df['PR'].values

        # create 3-tensor by replicating p across column dimension of df and z dimension corresponding to partition
        q = np.repeat((np.repeat(p[:,np.newaxis],data.shape[1],axis=1))[:,:,np.newaxis],L,axis=2)
      
        # compute lebesgue smoothings all at once by comparing cubes
        results = data_3d*(q<partition)*q

        # sum over set of nodes with measure less than p_j for each p_j \in partition
        sums = results.sum(axis=0)

        # store results in dictionary where each (key,value) pair is (F,F(p_j)) where F the antiderivative of some f
        F = dict(zip(self.frame_keys,sums.flatten()))

        return F

