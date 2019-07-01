from __future__ import absolute_import
from __future__ import print_function
from builtins import object
from builtins import str
import ujson as json
import numpy as np
import time
import networkx as nx
import subprocess
import signal
import json

from pandas import DataFrame
from .graphdualvs import GraphDualVS
from .DualVecStable import Functions
from sklearn.externals import joblib
from scipy import stats

from grafuple.config import DOTNET_MODEL_PATH, DECOMPILER_PATH
from grafuple.util import  flatten_dict

import logging

global G_star
global columns


class PageRankRandomForest(object):
    """ Class which implements the pagerank random forest model.  Decompiles a .NET binary, vectorizes the resulting
        JSON, and scores the resulting vector.

    """
    def __init__(self):
        """ Loads random forest from disk.

            Args:
                self (PageRankRandomForest class handle: implements cognition base class methods

            Returns:
                None
        """

        # load model
        self.model = joblib.load(DOTNET_MODEL_PATH)


    @staticmethod
    def dotnetdecompile(file_path):
        """ Decompiles a .NET binary and returns the result as JSON.  Uses standalone app CLRParserApp.exe.

            Args:
                file_path (str): Local file path to file object

            Returns:
                output (dict): dictionary containing decompilation data as determined by command line inputs to CLRParserApp.exe

        """

        # class and function for handling timeout exception
        class Alarm(Exception):
            pass
        def alarm_handler(signum, frame):
            raise Alarm

        # instantiate Popen object for calling standalone app and piping output to stdout
        proc = subprocess.Popen(
            [
                "mono",
                DECOMPILER_PATH,
                "--output:json",
                "--dfg",
                "--sdfg",
                "--interFnGraph",
                "-d",
                "--rawFunctions",
                file_path
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        # timeout exception and alarm handling
        signal.signal(signal.SIGALRM, alarm_handler)
        TIMEOUT=10
        signal.alarm(TIMEOUT)

         # call decompiler and pipe output to stdout
        try:
            stdout_buffer, stderr_buffer = proc.communicate()
            signal.alarm(0)
        except Alarm:
            proc.kill()
            raise Exception("Timeout on CLRParser")
        if len(stdout_buffer.strip()) == 0:
            raise Exception("No output from CLRParser")
        try:
            json.loads(stdout_buffer)
        except:
            raise Exception("Invalid JSON from CLRParser")

        # put contents of stdout into python dictionary and return
        output = json.loads(stdout_buffer)
        return output

    @staticmethod
    def extra_data():
        """ Initialization method for loading variables necessary for vectorization

            Args:
                None

            Returns:
                extra_data (dict): dictionary containing certain feature names and GraphDualVS instance

        """

        # dictionary to hold extra data required for vectorization
        extra_data = dict()

        # partition of probability space for use in Lesbegue antiderivative
        partition = np.array([20,40,60,80,95])

        # instantiate GraphDual class
        G_star = GraphDualVS(Functions,partition,'type')

        # extra data columns
        extra_columns = ['Nkeys','PageRankEntropy']

        # load extra data into extra data dictionary
        extra_data['columns'] = extra_columns
        extra_data['G_star'] = G_star

        return extra_data


    @classmethod
    def graph_vectorizer(cls, decompiled):
        """ Method for converting a decompiled .NET file in the form of a json to a feature vector based on
            computing expected values of functions defined on the vertex sets of certain graphs obtained
            through decompilation

            Args:
                cls (PageRankRandomForest handle): reference to class instance
                decompiled (dict): json in the form of python dictionary

            Returns:
                df (pandas DataFrame): feature vector in the form of a pandas dataframe

        """

        try:

            # time vectorization for given json
            t = time.time()

            extra_data = cls.extra_data()

            # class which implements most of the methods involved in vectorization
            G_star = extra_data['G_star']

            # feature names
            extra_columns = extra_data['columns']

            # instantiate dictionary to hold nonempty graphs
            graphs = dict()
            hash_val = cls.hash_val
            # hash_val = file_path[file_path.rfind('/')+1:file_path.index('.')]

            # grab dictionary containing all sdfg graphs for given file - some are empty
            raw_graphs = decompiled['functions']

            for key in raw_graphs.keys():

                # filter out empty graphs
                if raw_graphs[key]['status']['result']=='success':
                    if len(raw_graphs[key]['sdfg:json']['nodes'])>0:
                        graphs[hash_val+'_'+key] = raw_graphs[key]['sdfg:json']

            # instantiate dataframe to contain vectorized graphs for current json
            indices = list(graphs.keys())
            columns = G_star.frame_keys + extra_columns
            df = DataFrame(np.zeros((len(indices),len(columns))),index=indices,columns=columns)

            # iterate through set of graphs and vectorize each
            for key in indices:

                # grab this graph
                G = graphs[key]

                # Grab edges dictionary and convert to list of integers so incidence matrix can be indexed into
                edges = G['edges']
                nodes = G['nodes']

                # combinatorial values
                GraphData = dict()

                # Compute number of nodes to instantiate the incidence matrix
                Nkeys = len(list(nodes.keys()))
                GraphData['Nkeys'] = Nkeys

                # choose measure - pagerank in this case and compute dual smoothings
                measure = nx.pagerank(nx.DiGraph(G['edges']))
                F = G_star.lebesgue_smooth_dual_vecs(G, measure)

                # pagerank values
                GraphData['PageRankEntropy'] = stats.entropy(np.array(measure.values()))

                # merge page rank and topology dictionaries
                GraphData.update(F)

                # place vectorized graph into json-level dataframe
                df.loc[key] = flatten_dict(GraphData)


            # replace nans with zeros so that mean/std/etc can be computed
            df.fillna(value=0)

            # add hash column so that we can group graphs by mean and standard deviation
            df['id'] = [u[:str(u).index('_')] for u in df.index]

            # compute mean and standard deviations of computed values for each hash (many graphs per hash)
            df_grouped_mean = df.groupby('id')[columns].mean().fillna(value=0)
            df_grouped_std = df.groupby('id')[columns].std().fillna(value=0)

            # add column giving indicating the size of the largest function in the json
            Nkeys_grouped_max = df.groupby('id')[['Nkeys']].max()
            Nkeys_grouped_max.rename(columns={'Nkeys':'Nkeys_max'},inplace=True)

            # append _std to column names so that mean and std dev columns can be merged into the same table
            newnames = dict()
            for n in df_grouped_std.columns:
                newnames[n] = n + '_std'
            df_grouped_std.rename(columns=newnames,inplace=True)

            # join mean, standard deviation, and max files
            df = df_grouped_mean.join(df_grouped_std)
            df = df.join(Nkeys_grouped_max)

            # if vector is empty, replace with zeros
            if df.shape[0]==0:

                df = df.append(DataFrame(np.zeros((1,df.shape[1])),columns=df.columns,index=[hash_val]))

            # output name of file and time it took to vectorize
            print((hash_val, time.time()-t))

        # throw if something failed in the vectorization process
        except Exception as e:
            print(e)
            return None

        # return location of resulting vector
        return df

    @classmethod
    def graph_vectorizer_parallel(self, decompiled):
        """ Method for converting a decompiled .NET file in the form of a json to a feature vector based on
            computing expected values of functions defined on the vertex sets of certain graphs obtained
            through decompilation

            Args:
                self (PageRankRandomForest handle): reference to class instance
                decompiled (dict): json in the form of python dictionary

            Returns:
                df (pandas DataFrame): feature vector in the form of a pandas dataframe

        """

        pass # not sure if this will be indexed by files or by graphs

    @classmethod
    def predict(self, file_path):
        """ Method for scoring binary dotnet file object

            Args:
                self (PageRankRandomForest handle): reference to PageRankRandomForest object
                file_path (binary): local file path to binary

            Returns:
                self.rfc.predict(v) (int): v\in{0,1} where 0 is benign and 1 is malicioius

        """
        # decompile binary
        decompiled = self.dotnetdecompile(file_path)

        # vectorize resulting json
        v = self.graph_vectorizer(decompiled)

        # score vector
        return self.model.predict(v)









