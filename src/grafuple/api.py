""" Functions for external consumption
"""
from grafuple.PageRankRandomForest import PageRankRandomForest

_MODELS = {}


def model():
    if 'model' not in _MODELS:
        model = PageRankRandomForest()
        _MODELS['model'] = model
    return _MODELS['model']


def predict(file_path):
    """ Score dotnet file object located at file_path
    :param file_path: File path to local dotnet file for scoring
    :return:
    """
    return model().predict(file_path)
