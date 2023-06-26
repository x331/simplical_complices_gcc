#!/usr/bin/env python
# encoding: utf-8
# File Name: train_graph_moco.py
# Author: Jiezhong Qiu
# Create Time: 2019/12/13 16:44
# TODO:

import argparse
import os
import time

import dgl
import numpy as np
import psutil
# import ignite.contrib.handlers.tensorboard_logger as tb_logger
import torch

from gcc.contrastive.criterions import NCESoftmaxLoss, NCESoftmaxLossNS
from gcc.contrastive.memory_moco import MemoryMoCo
from gcc.datasets import (
    GRAPH_CLASSIFICATION_DSETS,
    GraphClassificationDataset,
    GraphClassificationDatasetLabeled,
    LoadBalanceGraphDataset,
    NodeClassificationDataset,
    NodeClassificationDatasetLabeled,
    worker_init_fn,
)
from gcc.datasets.data_util import batcher
from gcc.models import GraphEncoder
from gcc.utils.misc import AverageMeter, adjust_learning_rate, warmup_linear

from gcc.utils.misc import get_gpu_memory


def test_moco(train_loader, model, opt):
    """
    one epoch training for moco
    """
    print('gm1')
    model.eval()
    print('gm2')
    
    # dataset = data_util.create_graph_classification_dataset('imdb-binary')
    # graphs = dataset.graph_lists
    # print(graphs[0])
    # g4 = graphs[0].to('cuda:1')
    # print(g4.device)
    

    emb_list = []
    for idx, batch in enumerate(train_loader):
        print('enumerate loop')
        
        # dataset = data_util.create_graph_classification_dataset('imdb-binary')
        # graphs = dataset.graph_lists
        # print(graphs[0])
        # g4 = graphs[4].to('cuda:1')
        # print(g4.device)    
        
        # graphs = dataset.graph_lists
        # print(graphs[0])
        # g4 = graphs[4].to('cuda:1')
        # print(g4.device)       
         
        graph_q, graph_k = batch
        bsz = graph_q.batch_size
        # print(f'opt.device:{opt.device}')
        print(f'graph_q:{graph_q}')
        # print(f'graph_q.device:{graph_q.device}')
        print(f'gpu mem:{get_gpu_memory()}')
        print(graph_q.ndata)
        print(graph_q.ndata['_ID'].shape)
        # graph_q = graph_q.to('cuda:1')
        # # graph_k.to(opt.device)
        # print(f'cuda:{str(opt.gpu)}')
        # graph_k = graph_k.to(f'cuda:{str(opt.gpu)}')
        # print(f'graph_q:{graph_q.device}')
        # print(f'graph_k:{graph_k.device}')
        
        # graph_q.ndata['_ID'].to(opt.device)# why does this line make it so graph q and k can be send to gpu?
        # graph_q.ndata['pos_undirected'].to(opt.device)
        # graph_q.ndata['seed'].to(opt.device)
        # graph_q.edata['_ID'].to(opt.device)
        # graph_k.ndata['_ID'] = graph_k.ndata['_ID'].to(opt.device)# why does this line make it so graph q and k can be send to gpu?
        # graph_k.ndata['pos_undirected'] = graph_k.ndata['pos_undirected'].to(opt.device)
        # graph_k.ndata['seed'] = graph_k.ndata['seed'].to(opt.device)
        # graph_k.edata['_ID'] = graph_k.edata['_ID'].to(opt.device)
        # print(f'graph_q:{graph_q.device}')
        # print(f'graph_k:{graph_k.device}')
        print('confused')
        graph_q = graph_q.to(opt.device)
        graph_k = graph_k.to(opt.device)
        print(f'graph_q:{graph_q.device}')
        print(f'graph_k:{graph_k.device}')

        with torch.no_grad():
            feat_q = model(graph_q)
            feat_k = model(graph_k)

        assert feat_q.shape == (bsz, opt.hidden_size)
        emb_list.append(((feat_q + feat_k) / 2).detach().cpu())
    return torch.cat(emb_list)


def main(args_test):
    print('call to main')
    # dataset = data_util.create_graph_classification_dataset('imdb-binary')
    # graphs = dataset.graph_lists
    # print(graphs[0])
    # g4 = graphs[4].to('cuda:1')
    # print(g4.device)
    
    if os.path.isfile(args_test.load_path):
        print("=> loading checkpoint '{}'".format(args_test.load_path))
        checkpoint = torch.load(args_test.load_path, map_location="cpu")
        print(
            "=> loaded successfully '{}' (epoch {})".format(
                args_test.load_path, checkpoint["epoch"]
            )
        )
    else:
        print("=> no checkpoint found at '{}'".format(args_test.load_path))
    args = checkpoint["opt"]

    assert args_test.gpu is None or torch.cuda.is_available()
    print("Use GPU: {} for generation".format(args_test.gpu))
    args.gpu = args_test.gpu
    args.device = torch.device("cpu") if args.gpu is None else torch.device(args.gpu)
    print('g1')

    if args_test.dataset in GRAPH_CLASSIFICATION_DSETS:
        # print('graph classification')
        train_dataset = GraphClassificationDataset(
            dataset=args_test.dataset,
            rw_hops=args.rw_hops,
            subgraph_size=args.subgraph_size,
            restart_prob=args.restart_prob,
            positional_embedding_size=args.positional_embedding_size,
        )
    else:
        train_dataset = NodeClassificationDataset(
            dataset=args_test.dataset,
            rw_hops=args.rw_hops,
            subgraph_size=args.subgraph_size,
            restart_prob=args.restart_prob,
            positional_embedding_size=args.positional_embedding_size,
        )
    args.batch_size = len(train_dataset)
    print(args.batch_size)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        collate_fn=batcher(),
        shuffle=False,
        num_workers=args.num_workers,
    )
    print('g2')

    # create model and optimizer
    model = GraphEncoder(
        positional_embedding_size=args.positional_embedding_size,
        max_node_freq=args.max_node_freq,
        max_edge_freq=args.max_edge_freq,
        max_degree=args.max_degree,
        freq_embedding_size=args.freq_embedding_size,
        degree_embedding_size=args.degree_embedding_size,
        output_dim=args.hidden_size,
        node_hidden_dim=args.hidden_size,
        edge_hidden_dim=args.hidden_size,
        num_layers=args.num_layer,
        num_step_set2set=args.set2set_iter,
        num_layer_set2set=args.set2set_lstm_layer,
        gnn_model=args.model,
        norm=args.norm,
        degree_input=True,
    )
    print('g3')


    model = model.to(args.device)
    print('g4')
    print(f'model cuda :{next(model.parameters()).is_cuda}')
    model.load_state_dict(checkpoint["model"])
    print('g5')
    del checkpoint
    
    print('g6')
    
    # dataset = data_util.create_graph_classification_dataset('imdb-binary')
    # graphs = dataset.graph_lists
    # print(graphs[0])
    # g4 = graphs[4].to(args.device)
    # print(g4.device)

    emb = test_moco(train_loader, model, args)
    print('g7')
    print(os.path.join(args.model_folder, args_test.dataset), emb.numpy())
    print(np.shape(emb.numpy()))
    np.save(os.path.join(args.model_folder, args_test.dataset), emb.numpy())


if __name__ == "__main__":
    print('begining of gnerate')
    # from gcc.datasets import data_util #why does this code block allow for generate.py to not have gpu cuda problems when moving graphs to gpu?
    # dataset = data_util.create_graph_classification_dataset('imdb-binary') 
    # graphs = dataset.graph_lists
    # print(graphs[0])
    # g4 = graphs[4].to('cuda:1')
    # print(g4.device)
    src_ids = torch.tensor([2, 3, 4])
    dst_ids = torch.tensor([1, 2, 3])
    g5 = dgl.graph((src_ids, dst_ids))
    g5 = g5.to('cuda:1')
    print(g5.device)

    parser = argparse.ArgumentParser("argument for training")
    # fmt: off
    parser.add_argument("--load-path", type=str, help="path to load model")
    parser.add_argument("--dataset", type=str, default="dgl", choices=["dgl", "wikipedia", "blogcatalog", "usa_airport", "brazil_airport", "europe_airport", "cora", "citeseer", "pubmed", "kdd", "icdm", "sigir", "cikm", "sigmod", "icde", "h-index-rand-1", "h-index-top-1", "h-index"] + GRAPH_CLASSIFICATION_DSETS)
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
    # fmt: on
    main(parser.parse_args())
