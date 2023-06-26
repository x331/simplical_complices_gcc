import argparse
import os
import time

import dgl
import numpy as np
import psutil
import torch

from gcc.datasets import data_util


from gcc.utils import rwr
from gcc.datasets.data_util import batcher
from gcc.models import GraphEncoder
from gcc.utils.misc import AverageMeter, adjust_learning_rate, warmup_linear

from gcc.utils.misc import get_gpu_memory

from gcc.datasets import data_util
from gcc.datasets.data_util import create_node_classification_dataset
from gcc.tasks import build_model

from gcc.datasets import (
    GRAPH_CLASSIFICATION_DSETS,
    GraphClassificationDataset,
    GraphClassificationDatasetLabeled,
    LoadBalanceGraphDataset,
    NodeClassificationDataset,
    NodeClassificationDatasetLabeled,
    worker_init_fn,
)