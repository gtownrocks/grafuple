from __future__ import absolute_import
# /usr/bin/env python
# coding=utf-8
from mock import patch
from nose import tools as nt

from . import test_decompile
from grafuple.PageRankRandomForest import PageRankRandomForest


def test_predict():
    """ Unit test for the predict function.  Compile and decompile HelloWorld.cs
        program.  Vectorize decompilation, and score the god damned thing.

        Args:
            None

        Returns:
            None

    """

    # instantiate object
    PR_RF = PageRankRandomForest()

    # output decompilation of HelloWorld.cs
    decompiled = test_decompile.test_decompile()

    # vectorize decompilation json of HelloWorld.exe
    vector = PR_RF.graph_vectorizer(decompiled)

    # score vector - should yield 0 since HelloWorld.exe should be benign
    score = PR_RF.model.predict(vector)

    nt.assert_equal(score, 0)
