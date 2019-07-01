# /usr/bin/env python
# coding=utf-8
import scipy.sparse as sparse
from mock import patch
from nose import tools as nt

import grafuple.config as cfg


def test_config():
    nt.assert_is_not_none(cfg.DOTNET_MODEL_PATH)
    nt.assert_is_not_none(cfg.DOTNET_MODEL_PATH)
