#!/usr/bin/env python
# encoding: utf-8
# File Name: data_util.py
# Author: Jiezhong Qiu
# Create Time: 2019/12/30 14:20
# TODO:

import io
import itertools
import os
import os.path as osp
from collections import defaultdict, namedtuple

import dgl
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.sparse as sparse
import sklearn.preprocessing as preprocessing
import torch
import torch.nn.functional as F
from dgl.data.tu import TUDataset
from scipy.sparse import linalg

import simplex_lists.helpers as slhelp
import simplex_lists.SIMBLOCKGNN as slSIM
from sklearn.decomposition import PCA
import fbpca

def batcher():
    def batcher_dev(batch):
        # print(batch)
        graph_q, graph_k = zip(*batch)
        graph_q, graph_k = dgl.batch(graph_q), dgl.batch(graph_k)
        # print(f'graph_q{graph_q}, graph_k:{graph_k}')
        return graph_q, graph_k

    return batcher_dev


def labeled_batcher():
    def batcher_dev(batch):
        graph_q, label = zip(*batch)
        graph_q = dgl.batch(graph_q)
        return graph_q, torch.LongTensor(label)

    return batcher_dev


Data = namedtuple("Data", ["x", "edge_index", "y"])


def create_graph_classification_dataset(dataset_name):
    name = {
        "imdb-binary": "IMDB-BINARY",
        "imdb-multi": "IMDB-MULTI",
        "rdt-b": "REDDIT-BINARY",
        "rdt-5k": "REDDIT-MULTI-5K",
        "collab": "COLLAB",
    }[dataset_name]
    dataset = TUDataset(name)
    # dataset.num_labels = dataset.num_labels[0]
    # dataset.num_labels = dataset.num_classes
    dataset.graph_labels = dataset.graph_labels.squeeze()
    return dataset


class Edgelist(object):
    def __init__(self, root, name):
        self.name = name
        edge_list_path = os.path.join(root, name + ".edgelist")
        node_label_path = os.path.join(root, name + ".nodelabel")
        edge_index, y, self.node2id = self._preprocess(edge_list_path, node_label_path)
        self.data = Data(x=None, edge_index=edge_index, y=y)
        self.transform = None

    def get(self, idx): #why is this taking an idx if it doesn't use it
        assert idx == 0
        return self.data

    def _preprocess(self, edge_list_path, node_label_path):
        with open(edge_list_path) as f:
            edge_list = []
            node2id = defaultdict(int)
            for line in f:
                x, y = list(map(int, line.split()))
                # Reindex
                if x not in node2id:
                    node2id[x] = len(node2id)
                if y not in node2id:
                    node2id[y] = len(node2id)
                edge_list.append([node2id[x], node2id[y]])
                edge_list.append([node2id[y], node2id[x]])

        num_nodes = len(node2id)
        with open(node_label_path) as f:
            nodes = []
            labels = []
            label2id = defaultdict(int)
            for line in f:
                x, label = list(map(int, line.split()))
                if label not in label2id:
                    label2id[label] = len(label2id)
                nodes.append(node2id[x])
                if "hindex" in self.name:
                    labels.append(label) #why no relabeling on hindex datasets?
                else:
                    labels.append(label2id[label])
            if "hindex" in self.name:
                median = np.median(labels)
                labels = [int(label > median) for label in labels] #this is why not relabeling
        assert num_nodes == len(set(nodes))
        y = torch.zeros(num_nodes, len(label2id))
        y[nodes, labels] = 1
        return torch.LongTensor(edge_list).t(), y, node2id


class SSSingleDataset(object):
    def __init__(self, root, name):
        edge_index = self._preprocess(root, name)
        self.data = Data(x=None, edge_index=edge_index, y=None)
        self.transform = None

    def get(self, idx):
        assert idx == 0
        return self.data

    def _preprocess(self, root, name):
        graph_path = os.path.join(root, name + ".graph")

        with open(graph_path) as f:
            edge_list = []
            node2id = defaultdict(int)
            f.readline()
            for line in f:
                x, y, t = list(map(int, line.split()))
                # Reindex
                if x not in node2id:
                    node2id[x] = len(node2id)
                if y not in node2id:
                    node2id[y] = len(node2id)
                # repeat t times
                for _ in range(t):
                    # to undirected
                    edge_list.append([node2id[x], node2id[y]])
                    edge_list.append([node2id[y], node2id[x]])

        num_nodes = len(node2id)

        return torch.LongTensor(edge_list).t() # why not returning a y and node2id like above class?

class SSDataset(object):
    def __init__(self, root, name1, name2):
        edge_index_1, dict_1, self.node2id_1 = self._preprocess(root, name1)
        edge_index_2, dict_2, self.node2id_2 = self._preprocess(root, name2)
        self.data = [
            Data(x=None, edge_index=edge_index_1, y=dict_1),
            Data(x=None, edge_index=edge_index_2, y=dict_2),
        ]
        self.transform = None

    def get(self, idx):
        assert idx == 0
        return self.data

    def _preprocess(self, root, name):
        dict_path = os.path.join(root, name + ".dict")
        graph_path = os.path.join(root, name + ".graph")

        with open(graph_path) as f:
            edge_list = []
            node2id = defaultdict(int)
            f.readline()
            for line in f:
                x, y, t = list(map(int, line.split()))
                # Reindex
                if x not in node2id:
                    node2id[x] = len(node2id)
                if y not in node2id:
                    node2id[y] = len(node2id)
                # repeat t times
                for _ in range(t):
                    # to undirected
                    edge_list.append([node2id[x], node2id[y]])
                    edge_list.append([node2id[y], node2id[x]])

        name_dict = dict()
        with open(dict_path) as f:
            for line in f:
                name, str_x = line.split("\t")
                x = int(str_x)
                if x not in node2id:
                    node2id[x] = len(node2id)
                name_dict[name] = node2id[x]

        num_nodes = len(node2id)

        return torch.LongTensor(edge_list).t(), name_dict, node2id

def create_node_classification_dataset(dataset_name):
    if "airport" in dataset_name:
        return Edgelist(
            "data/struc2vec/",
            {
                "usa_airport": "usa-airports",
                "brazil_airport": "brazil-airports",
                "europe_airport": "europe-airports",
            }[dataset_name],
        )
    elif "h-index" in dataset_name:
        return Edgelist(
            "data/hindex/",
            {
                "h-index-rand-1": "aminer_hindex_rand1_5000",
                "h-index-top-1": "aminer_hindex_top1_5000",
                "h-index": "aminer_hindex_rand20intop200_5000",
            }[dataset_name],
        )
    elif dataset_name in ["kdd", "icdm", "sigir", "cikm", "sigmod", "icde"]:
        return SSSingleDataset("data/panther/", dataset_name)
    else:
        raise NotImplementedError


def _rwr_trace_to_dgl_graph(
    g, seed, trace, positional_embedding_size, entire_graph=False
):
    # subv = torch.unique(torch.cat([trace[0],trace[1]])).tolist()
    subv = torch.unique(trace).tolist()
    # print(f'subv1:{subv}')
    #Vertex order doesn’t matter because most of graph neural networks are invariant to permutations of their inputs, probably can get rid of below line
    # np.random.shuffle(subv)
    # print(f'subv1.1:{subv}')
    try:
        subv.remove(seed)
    except ValueError:
        pass
    subv = [seed] + subv
    # print(f'subv2:{subv}')
    if entire_graph:
        subg = dgl.DGLGraph.subgraph(g,g.nodes())
    else:
        # print(subv)
        subg = dgl.DGLGraph.subgraph(g,subv)
        # print(subg.ndata)
        # print(g.ndata)

    
    
    subg = _add_undirected_graph_positional_embedding(subg, positional_embedding_size)

    subg.ndata["seed"] = torch.zeros(subg.number_of_nodes(), dtype=torch.long)
    if entire_graph:
        subg.ndata["seed"][seed] = 1
    else:
        subg.ndata["seed"][0] = 1
    return subg


def eigen_decomposision(n, k, laplacian, hidden_size, retry):
    if k <= 0:
        return torch.zeros(n, hidden_size)
    laplacian = laplacian.astype("float64")
    ncv = min(n, max(2 * k + 1, 20))
    # follows https://stackoverflow.com/questions/52386942/scipy-sparse-linalg-eigsh-with-fixed-seed
    v0 = np.random.rand(n).astype("float64")
    for i in range(retry):
        try:
            s, u = linalg.eigsh(laplacian, k=k, which="LA", ncv=ncv, v0=v0)
        except sparse.linalg.eigen.arpack.ArpackError:
            print("arpack error, retry=", i)
            ncv = min(ncv * 2, n)
            if i + 1 == retry:
                sparse.save_npz("arpack_error_sparse_matrix.npz", laplacian)
                u = torch.zeros(n, k)
        else:
            break
    x = preprocessing.normalize(u, norm="l2")
    x = torch.from_numpy(x.astype("float32"))
    x = F.pad(x, (0, hidden_size - k), "constant", 0)
    return x


def eigen_decomposision_torch(n, k, laplacian, hidden_size, retry):
    if k <= 0:
        return torch.zeros(n, hidden_size)
    
    
    val, vec = torch.linalg.eigh(laplacian)
    
    u = vec[:,:k]

    x = torch.nn.functional.normalize(u)
    x = x.to(dtype = torch.float32)
    x = F.pad(x, (0, hidden_size - k), "constant", 0)
    return x


def single_val_decomp(kl, kr, mat, hidden_size, retry):
    if kl <= 0 or kr <= 0: # this might not be correct and require more involved control flow
        return torch.zeros(mat.size[0], hidden_size), torch.zeros(mat.size[1], hidden_size)
    for i in range(retry):
        try:
            u, s , v= scipy.linalg.svd(mat.astype("float64"))
        except scipy.linalg.LinAlgError:
            print("LinAlgError error, retry=", i)
            if i + 1 == retry:
                sparse.save_npz("LinAlgError_error_sparse_matrix.npz", mat)
                u = torch.zeros(mat.size[0], hidden_size)
                v = torch.zeros(mat.size[1], hidden_size)
        else:
            break
    if u.shape[1] >= kl:
        u = u[:, :kl]
    fdim = u.shape[1]     
    u = preprocessing.normalize(u, norm="l2")
    u = torch.from_numpy(u.astype("float32"))
    u = F.pad(u, (0, hidden_size - fdim ), "constant", 0)
    
    if v.shape[1] >= kr:
        v = v[:, :kr]
    fdim = v.shape[1]     
    v = preprocessing.normalize(v, norm="l2")
    v = torch.from_numpy(v.astype("float32"))
    v = F.pad(v, (0, hidden_size - fdim ), "constant", 0)
    return u , v



def pca_torch(kl, kr, mat, hidden_size, retry):   #if more speed needed look into TORCH.LINALG.QR
    if kl <= 0 or kr <= 0: # this might not be correct and require more involved control flow
        return torch.zeros(mat.size(dim=0), hidden_size)
    
    u,s,v = torch.pca_lowrank(mat, q = min(kl, kr))
    mat = torch.matmul(mat, v[:, :min(kl, kr)]) 
    
    fdim = mat.shape[1]     
    mat = torch.nn.functional.normalize(mat)
    mat = mat.to(dtype = torch.float32)
    mat = F.pad(mat, (0, hidden_size - fdim ), "constant", 0)
    return mat

def pca_fbpca(kl, kr, mat, hidden_size, retry):  
    if kl <= 0 or kr <= 0: # this might not be correct and require more involved control flow
        return torch.zeros(mat.size(dim=0), hidden_size)
    
    
    u,s,v = fbpca.pca(mat.numpy(), k = min(kl, kr))
    mat = torch.matmul( mat,  torch.tensor(np.transpose(v)[:, :min(kl, kr)])) 

    
    fdim = mat.shape[1]     
    mat = torch.nn.functional.normalize(mat)
    mat = mat.to(dtype = torch.float32)
    mat = F.pad(mat, (0, hidden_size - fdim ), "constant", 0)
    return mat
    




# def _add_undirected_graph_positional_embedding(g, hidden_size, retry=10):
#     # We use eigenvectors of normalized graph laplacian as vertex features.
#     # It could be viewed as a generalization of positional embedding in the
#     # attention is all you need paper.
#     # Recall that the eignvectors of normalized laplacian of a line graph are cos/sin functions.
#     # See section 2.4 of http://www.cs.yale.edu/homes/spielman/561/2009/lect02-09.pdf
#     n = g.number_of_nodes()
#     # adj = g.adjacency_matrix_scipy(transpose=False, return_edge_ids=False).astype(float)
#     adj = g.adj()   
#     # print(f'dgladj{adj}')
#     #greatly question if I did this right
#     # data, row_ind, col_ind = adj.val, adj.row , adj.col
#     indptr, indices, data  = adj.csr()
#     # print(f'data:{data}, row_ind:{row_ind}, col_ind:{col_ind}')
#     # print(data.size(),row_ind.size(),col_ind.size())
#     adj = sparse.csr_array((data, indices, indptr), shape= adj.shape)
#     # norm = sparse.diags(
#     #     dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float
#     # )
#     norm = sparse.diags(
#         dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float
#     )
#     laplacian = norm @ adj @ norm #these * may need to be to @ 
#     # print(f'adj{adj}')
#     # print(f'laplacian{laplacian}')
#     k = min(n - 2, hidden_size)
#     x = eigen_decomposision(n, k, laplacian, hidden_size, retry)
#     g.ndata["pos_undirected"] = x.float()
#     return g







def _add_undirected_graph_positional_embedding(g, hidden_size, retry=10):
    # print("what")
    # We use eigenvectors of normalized graph laplacian as vertex features.
    # It could be viewed as a generalization of positional embedding in the
    # attention is all you need paper.
    # Recall that the eignvectors of normalized laplacian of a line graph are cos/sin functions.
    # See section 2.4 of http://www.cs.yale.edu/homes/spielman/561/2009/lect02-09.pdf
    g = dgl.to_bidirected(g) #hope getting rid of parrallel/multi edges doesnt mess this up
    n = g.number_of_nodes()
    adj = g.adj()   
    indices = adj.coo() 
    data =  adj.val
    adj = sparse.coo_array((data, indices) , adj.shape)
    norm = sparse.diags(
        dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float
    )
    laplacian = norm @ adj @ norm #these * may need to be to @ 
    # print(f'adj{adj}')
    # print(f'laplacian{laplacian}')
    k = min(n - 2, hidden_size)
    x = eigen_decomposision(n, k, laplacian, hidden_size, retry)
    g.ndata["pos_undirected"] = x.float()
    g.ndata["pos_undirected_0"] = x.float()
    
    # with open('/home/shakir/simplical_complices_gcc/test.txt', 'a') as f:
    #     f.write(f"{g.adj().to_dense()}")
    #     f.write(f"{adj.toarray()}")
    #     f.write(f'{g.adj().to_dense()-adj.toarray()}')
    
    # #should figure adding this F.softmax(F.relu(sim_block), dim = 1)
    # print('adj',g.adj().shape, np.count_nonzero(adj.toarray()) )
    # # print(g.edges())
    # print(0,x.shape)
    # print('edges', g.num_edges())
    # g = dgl.to_bidirected(g)
    # print('edges', g.num_edges())
    # b1 ,b2 , faces = slhelp.compute_hodge_matrix_2(g.num_nodes(),g.edges())
    # print( g, len(faces))
    # b2_to_nodes = np.abs(b1) @ np.abs(b2)
    # l1_to_nodes = b1 @ (np.transpose(b1) @ b1 + b2 @ np.transpose(b2))
    # print(f"b1:{b1.shape}, b2:{b2.shape}, b2_to_nodes{b2_to_nodes.shape}, l1_to_nodes{l1_to_nodes.shape}")
    # # print('b1',b1)
    # # print('b2',b2)
    # # print('b1row',b1[0])
    # # print('b2col',b2[:,0])
    # # print('b1col',b1[:,0])
    # # print('b2_to_nodes transpose', np.transpose(b2_to_nodes))
    # # print("l1_to_nodes", l1_to_nodes)
    # # print(np.transpose(b1) @ b1 + b2 @ np.transpose(b2))
    # # print('b1b1T',(b1@b1.T).shape)
    # # print(b1@b1.T)
    # # print(g.adj().to_dense())
    # L0, L1f= slSIM.compute_bunch_matrices(b1, b2) #using normalize matrices might try unnormalized     weird shapes
    # L1f_to_nodes = b1 @ L1f
    # L0_to_nodes = b1 @ L0
    # # L0 = torch.tensor(L0, dtype=torch.float32)  # convert hodge matrix to tensor format
    # # L1f = torch.tensor(L1f, dtype=torch.float32)
    # print('L0' ,L0.shape)
    # print('L1f', L1f.shape)
    # print('L0_to_nodes', L0_to_nodes.shape)
    # print("L1f_to_nodes" , L1f_to_nodes.shape)
    # print("b1", b1)
    # print("b2_to_nodes", b2_to_nodes)
    # print("l1_to_nodes", l1_to_nodes)
    # print('L0_to_nodes', L0_to_nodes)
    # print("L1f_to_nodes" , L1f_to_nodes)
    # relu = torch.nn.ReLU()
    # softmax = torch.nn.Softmax()
    # exit(1231)
    
    
    
    
    # # b1, b2_to_nodes, l1_to_nodes, L0, L1f
    # print(laplacian.shape, x.shape, type(g.ndata["pos_undirected"] ))
    # # g = dgl.to_bidirected(g) #doing this at the begining of the function now
    # b1 ,b2 , faces = slhelp.compute_hodge_matrix_2(g.num_nodes(),g.edges())
    # b2_to_nodes = np.abs(b1) @ np.abs(b2)
    # l1_to_nodes = b1 @ (np.transpose(b1) @ b1 + b2 @ np.transpose(b2))
    # L0, L1f= slSIM.compute_bunch_matrices(b1, b2) #using normalize matrices might try unnormalized     weird shapes
    # L1f_to_nodes = b1 @ L1f
    # L0_to_nodes = b1 @ L0
    # print(f"b1:{b1.shape}, b2:{b2.shape}, b2_to_nodes{b2_to_nodes.shape}, L0{L0.shape}, L1f{L1f.shape}, L0_to_nodes{L0_to_nodes.shape}, L1f_to_nodes{L1f_to_nodes.shape}")
    
    # #b1_compr == l1_to_nodes          
    # k = min(b1.shape[1] - 2, hidden_size)
    # b1_compr1  = single_val_decomp(k,b1,hidden_size, retry)
    # k = min(b2_to_nodes.shape[1] - 2, hidden_size)
    # b2_to_nodes1 = single_val_decomp(k, b2_to_nodes, hidden_size, retry)
    # k = min(l1_to_nodes.shape[1] - 2, hidden_size)
    # l1_to_nodes1 = single_val_decomp(k, l1_to_nodes, hidden_size, retry)
    # k = min(L0_to_nodes.shape[1] - 2, hidden_size)
    # L0_to_nodes1 = single_val_decomp(k, L0_to_nodes , hidden_size, retry)
    # k = min(L1f_to_nodes.shape[1] - 2, hidden_size)
    # L1f_to_nodes = single_val_decomp( k, L1f_to_nodes, hidden_size, retry)
    
    # k = min(b1.shape[1] - 2, hidden_size)
    # b1_compr3  = single_val_decomp(k,b1,hidden_size, retry)
    # k = min(b2_to_nodes.shape[1] - 2, hidden_size)
    # b2_to_nodes3 = single_val_decomp(k, b2_to_nodes, hidden_size, retry)
    # k = min(l1_to_nodes.shape[1] - 2, hidden_size)
    # l1_to_nodes3 = single_val_decomp(k, l1_to_nodes, hidden_size, retry)
    # k = min(L0_to_nodes.shape[1] - 2, hidden_size)
    # L0_to_nodes3 = single_val_decomp(k, L0_to_nodes , hidden_size, retry)
    # k = min(L1f_to_nodes.shape[1] - 2, hidden_size)
    # L1f_to_nodes3 = single_val_decomp( k, L1f_to_nodes, hidden_size, retry)
    
    # k = min(L0.shape[1] - 2, hidden_size)
    # L0 = eigen_decomposision(L0.shape[0], k, L0 , hidden_size, retry)
    # b1 = torch.tensor(b1,dtype=torch.float32 ) 
    # L0_to_nodes2 =torch.matmul(b1, L0)
    # k = min(L1f.shape[1] - 2, hidden_size)
    # L1f = eigen_decomposision(L1f.shape[0], k, L1f, hidden_size, retry)
    # L1f_to_nodes2 = torch.matmul(b1, L1f)
    
    # print(f"b1_compr:{b1_compr.shape}, \n b2_to_nodes{b2_to_nodes.shape}, \n l1_to_nodes{l1_to_nodes.shape}, \n L0_to_nodes{L0_to_nodes.shape}, \n L1f_to_nodes{L1f_to_nodes.shape}, \n  L0_to_nodes2{L0_to_nodes2.shape}, \n L1f_to_nodes2{L1f_to_nodes2.shape}")

    # with open('/home/shakir/simplical_complices_gcc/test.txt', 'a') as f:
    #     f.write(f"b1_compr:{b1_compr[0]}, \n b2_to_nodes{b2_to_nodes[0]}, \n l1_to_nodes{l1_to_nodes[0]}, \n L0_to_nodes{L0_to_nodes[0]}, \n L1f_to_nodes{L1f_to_nodes[0]}, \n  L0_to_nodes2{L0_to_nodes2[0]}, \n L1f_to_nodes2{L1f_to_nodes2[0]}\n")
    #     f.write(f"{g.adj().to_dense()}")
    #     f.write(f"{adj.toarray()}")
    #     f.write(f'{g.adj().to_dense()-adj.toarray()}')
    #     f.write(f"b1_compr:{b1_compr}, \n b2_to_nodes{b2_to_nodes}, \n l1_to_nodes{l1_to_nodes}, \n L0_to_nodes{L0_to_nodes}, \n L1f_to_nodes{L1f_to_nodes}, \n  L0_to_nodes2{L0_to_nodes2}, \n L1f_to_nodes2{L1f_to_nodes2}\n")

    # g.ndata["pos_undirected_b1_compr"] = b1_compr.float()
    # g.ndata["pos_undirected_b2_to_nodes"] = b2_to_nodes.float()
    # g.ndata["pos_undirected_l1_to_nodes"] = l1_to_nodes.float()
    # g.ndata["pos_undirected_L0_to_nodes"] = L0_to_nodes.float()
    # g.ndata["pos_undirected_L1f_to_nodes"] = L1f_to_nodes.float()
    # g.ndata["pos_undirected_L0_to_nodes2"] = L0_to_nodes2.float()
    # g.ndata["pos_undirected_L1f_to_nodes2"] = L1f_to_nodes2.float()
    # exit(1231)


    # #test1
    # import time
    # start_time = time.time()
    # print(time)
    # b1 ,b2 , faces = slhelp.compute_hodge_matrix_2(g.num_nodes(),g.edges())
    # print(1, time.time() - start_time)
    # start_time = time.time()
    # L0u, L1f= slSIM.compute_bunch_matrices(b1, b2) #using normalize matrices might try unnormalized     weird shapes
    # print('shape',L0u.shape, L1f.shape)
    # print(2, time.time() - start_time)
    # start_time = time.time()
    # k = min(L0u.shape[1] - 2, hidden_size)
    # L0u = eigen_decomposision(L0u.shape[0], k, L0u , hidden_size, retry)
    # print(3, time.time() - start_time)
    # start_time = time.time()
    # b1 = torch.tensor(b1,dtype=torch.float32 ) 
    # print(4, time.time() - start_time)
    # start_time = time.time()
    # L0u_to_nodes2 =torch.matmul(b1, L0u)
    # k = min(L1f.shape[1] - 2, hidden_size)
    # L1f = eigen_decomposision(L1f.shape[0], k, L1f, hidden_size, retry)
    # print(5, time.time() - start_time)
    # start_time = time.time()
    # L1f_to_nodes2 = torch.matmul(b1, L1f)
    
    # L0u_to_nodes2 = preprocessing.normalize(L0u_to_nodes2, norm="l2")
    # L0u_to_nodes2 = torch.from_numpy(L0u_to_nodes2.astype("float32"))
    # L1f_to_nodes2 = preprocessing.normalize(L1f_to_nodes2, norm="l2")
    # L1f_to_nodes2 = torch.from_numpy(L1f_to_nodes2.astype("float32"))
    # print(4, time.time() - start_time)
    # start_time = time.time()
    
    # g.ndata["pos_undirected"] = torch.cat((x.float(),L0u_to_nodes2.float(),L1f_to_nodes2.float()),dim=1)
    
    
    
    #test2
    # import time
    # start_time = time.time()
    # print(time)
    # b1 ,b2 , faces = slhelp.compute_hodge_matrix_2(g.num_nodes(),g.edges())
    # print(1, time.time() - start_time)
    # start_time = time.time()
    # L0u, L1f= slhelp.compute_unnomralized_bunch_matrices(b1, b2) #using normalize matrices might try unnormalized     weird shapes
    # print('shape',L0u.shape, L1f.shape)
    # print(2, time.time() - start_time)
    # start_time = time.time()
    # k = min(L0u.shape[1] - 2, hidden_size)
    # L0u = eigen_decomposision(L0u.shape[0], k, L0u , hidden_size, retry)
    # print(3, time.time() - start_time)
    # start_time = time.time()
    # b1 = torch.tensor(b1,dtype=torch.float32 ) 
    # print(4, time.time() - start_time)
    # start_time = time.time()
    # L0u_to_nodes2 =torch.matmul(b1, L0u)
    # k = min(L1f.shape[1] - 2, hidden_size)
    # L1f = eigen_decomposision(L1f.shape[0], k, L1f, hidden_size, retry)
    # print(5, time.time() - start_time)
    # start_time = time.time()
    # L1f_to_nodes2 = torch.matmul(b1, L1f)
    
    # L0u_to_nodes2 = preprocessing.normalize(L0u_to_nodes2, norm="l2")
    # L0u_to_nodes2 = torch.from_numpy(L0u_to_nodes2.astype("float32"))
    # L1f_to_nodes2 = preprocessing.normalize(L1f_to_nodes2, norm="l2")
    # L1f_to_nodes2 = torch.from_numpy(L1f_to_nodes2.astype("float32"))
    # print(4, time.time() - start_time)
    # start_time = time.time()
    
    # g.ndata["pos_undirected"] = torch.cat((x.float(),L0u_to_nodes2.float(),L1f_to_nodes2.float()),dim=1)
    
    
    # #test 3 #when using again make sure to ablsoulte value b1 and b2
    # b1 ,b2 , faces = slhelp.compute_hodge_matrix_2(g.num_nodes(),g.edges())
    # L0u, L1f= slhelp.compute_unnomralized_bunch_matrices(b1, b2) #using normalize matrices might try unnormalized     weird shapes
    # # print('shape',L0u.shape, L1f.shape)
    # k = min(L0u.shape[1] - 2, hidden_size)
    # L0u = eigen_decomposision(L0u.shape[0], k, L0u , hidden_size, retry)
    # b1 = torch.tensor(b1,dtype=torch.float32 ) 
    # L0u_to_nodes2 =torch.matmul(b1, L0u)
    # k = min(L1f.shape[1] - 2, hidden_size)
    # L1f = eigen_decomposision(L1f.shape[0], k, L1f, hidden_size, retry)
    # L1f_to_nodes2 = torch.matmul(b1, L1f)
    
    # L0u_to_nodes2 = preprocessing.normalize(L0u_to_nodes2, norm="l2")
    # L0u_to_nodes2 = torch.from_numpy(L0u_to_nodes2.astype("float32"))
    # L1f_to_nodes2 = preprocessing.normalize(L1f_to_nodes2, norm="l2")
    # L1f_to_nodes2 = torch.from_numpy(L1f_to_nodes2.astype("float32"))
    
    # g.ndata["pos_undirected"] = torch.cat((x.float(),L0u_to_nodes2.float(),L1f_to_nodes2.float()),dim=1)
    # # print('pos_undirected',g.ndata["pos_undirected"].shape)
    
    
    
    # #test 4
    # b1 ,b2 , faces = slhelp.compute_hodge_matrix_2(g.num_nodes(),g.edges())
    # b1, b2 = np.abs(b1), np.abs(b2)
    # # print(b1.shape)
    # k_left = min(b1.shape[0], hidden_size)
    # k_right = min(b1.shape[1], hidden_size)
    # import time
    # start_time = time.time()
    # b1_left, b1_right = single_val_decomp(k_left, k_right, b1 , hidden_size, retry)
    # print(1, time.time() - start_time)
    # start_time = time.time()
    # print(b2.shape)
    # k_left = min(b2.shape[0], hidden_size)
    # k_right = min(b2.shape[1], hidden_size)
    # b2_left, b2_right = single_val_decomp(k_left, k_right, b2 , hidden_size, retry)
    # print(2, time.time() - start_time)
    # start_time = time.time()
    # # print(b1)
    # # print(adj.shape)
    # # print(b1_left.shape, b1_right.shape, b2_left.shape, b2_right.shape)
    # b1 = torch.tensor(b1,dtype=torch.float32 ) 
    # b2 = torch.tensor(b2,dtype=torch.float32 ) 
    # b1_right_to = b1 @ b1_right
    # b2_left_to = b1 @ b2_left
    # b2_right_to = b1 @ b2 @ b2_right
    # # print(b1_left.shape, b1_right_to.shape, b2_left_to.shape, b2_right_to.shape)
    # b1_left = torch.tensor(preprocessing.normalize(b1_left, norm="l2"))
    # b1_right_to = torch.tensor(preprocessing.normalize(b1_right_to, norm="l2"))
    # b2_left_to = torch.tensor(preprocessing.normalize(b2_left_to, norm="l2"))
    # b2_right_to = torch.tensor(preprocessing.normalize(b2_right_to, norm="l2"))
    # # print(b1_left, b1_right_to, b2_left_to, b2_right_to)
    # # print(type(b1_left), type(b1_right_to), type(b2_left_to), type(b2_right_to))
    # g.ndata["pos_undirected"] = torch.cat((x.float(),b1_left.float(),b1_right_to.float(),b2_left_to.float(),b2_right_to.float()),dim=1)
    
    
    # # test 5
    # b1 ,b2 , faces = slhelp.compute_hodge_matrix_2(g.num_nodes(),g.edges())
    # b1, b2 = np.abs(b1), np.abs(b2)
    # b2_to = b1 @ b2
    # # print(b1.shape)
    # k = min(b1.shape[0], b1.shape[1], hidden_size)
    # pca = PCA(n_components = k)
    # import time
    # # start_time = time.time()
    # b1_compr = pca.fit_transform(b1)
    # # print(1, time.time() - start_time)
    # # start_time = time.time()
    # # print(b2_to.shape)
    # k = min(b2_to.shape[0], b2_to.shape[1], hidden_size)
    # pca = PCA(n_components = k)
    # b2_to_compr = pca.fit_transform(b2_to)
    # # print(2, time.time() - start_time)
    # # start_time = time.time()
    # # print(b1)
    # # print(adj.shape)
    # # print(b1_left.shape, b1_right.shape, b2_left.shape, b2_right.shape)
    # b1_compr = torch.tensor(b1_compr,dtype=torch.float32 ) 
    # b2_to_compr = torch.tensor(b2_to_compr,dtype=torch.float32 ) 

    # # print(b1_left.shape, b1_right_to.shape, b2_left_to.shape, b2_right_to.shape)
    # b1_compr = torch.tensor(preprocessing.normalize(b1_compr, norm="l2"))
    # b2_to_compr = torch.tensor(preprocessing.normalize(b2_to_compr, norm="l2"))
    # # print(b1_left, b1_right_to, b2_left_to, b2_right_to)
    # # print(type(b1_left), type(b1_right_to), type(b2_left_to), type(b2_right_to))
    # g.ndata["pos_undirected"] = torch.cat((x.float(),b1_compr.float(),b2_to_compr.float()),dim=1)
    
    # #test 6
    # b1 ,b2 , faces = slhelp.compute_hodge_matrix_2(g.num_nodes(),g.edges())
    # b1, b2 = torch.tensor(np.abs(b1)), torch.tensor(np.abs(b2))
    # b1_edg = b1.T @b1
    # b2_fac = b2.T@b2
    
    # import time
    # start_time = time.time()
    # k = min(b1_edg.shape[0]-2, hidden_size)    
    # b1_edg_compr =  eigen_decomposision_torch(n, k, b1_edg, hidden_size, retry)
    # print(1, time.time() - start_time)
    # start_time = time.time()
    # k = min(b2_fac.shape[0]-2, hidden_size)    
    # b2_fac_compr = eigen_decomposision_torch(n, k, b2_fac, hidden_size, retry)
    # print(2, time.time() - start_time)
    # start_time = time.time()
    # b1_edg_to = torch.nn.functional.normalize(b1 @ b1_edg)
    # b2_fac_to = torch.nn.functional.normalize(b1 @ b2 @ b2_fac)

    # # print(g.ndata["pos_undirected"].shape)
    # g.ndata["pos_undirected"] = torch.cat((x.float(),b1_edg_to.float(),b2_fac_to.float()),dim=1)
    # # print(g.ndata["pos_undirected"].shape)
    
    # #test 7
    # b1 ,b2 , faces = slhelp.compute_hodge_matrix_2(g.num_nodes(),g.edges())
    # b1, b2 = np.abs(b1), np.abs(b2)
    # # print(b1.shape,b2.shape)
    # b2_to = b1 @ b2
    # # print(b1.shape)
    # b1 = torch.tensor(b1)
    # b2_to = torch.tensor(b2_to)
    
    # kl = min(b1.shape[0], hidden_size)
    # kr = min(b1.shape[1], hidden_size)
    # # import time
    # # start_time = time.time()
    # b1_compr = pca_torch(kl, kr, b1, hidden_size, retry)
    # # print(1, time.time() - start_time)
    # # start_time = time.time()
    # # print(b2_to.shape)
    # kl = min(b2_to.shape[0], hidden_size)
    # kr = min(b2_to.shape[1], hidden_size)
    # b2_to_compr = pca_torch(kl, kr, b2_to, hidden_size, retry)
    # # print(2, time.time() - start_time)
    # # start_time = time.time()
    # # print(b1)
    # # print(adj.shape)
    # # print(b1_left.shape, b1_right.shape, b2_left.shape, b2_right.shape)

    # # print(b1_compr.shape, b2_to_compr.shape)
    # # print(b1_left, b1_right_to, b2_left_to, b2_right_to)
    # # print(type(b1_left), type(b1_right_to), type(b2_left_to), type(b2_right_to))
    # # print(g.ndata["pos_undirected"].shape)
    # g.ndata["pos_undirected"] = torch.cat((x.float(),b1_compr.float(),b2_to_compr.float()),dim=1)

    ###!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
    #test 8 use this one #comment out if too slow and not using multi features
    b1 ,b2 , faces = slhelp.compute_hodge_matrix_2(g.num_nodes(),g.edges())
    
    # b1 = torch.abs(torch.tensor(b1)).to_sparse().to(dtype=torch.float32)
    # b2 = torch.abs(torch.tensor(b2)).to_sparse().to(dtype=torch.float32)
    #b2_to = b1 @ b2

    b1, b2 = np.abs(b1), np.abs(b2)
    b2_to = b1 @ b2
    b1 = torch.tensor(b1)
    b2_to = torch.tensor(b2_to)
    
    kl = min(b1.shape[0], hidden_size)
    kr = min(b1.shape[1], hidden_size)
    # import time
    # start_time = time.time()
    b1_compr = pca_torch(kl, kr, b1, hidden_size, retry)
    # print(1, time.time() - start_time)
    # start_time = time.time()
    kl = min(b2_to.shape[0], hidden_size)
    kr = min(b2_to.shape[1], hidden_size)
    b2_to_compr = pca_torch(kl, kr, b2_to, hidden_size, retry)
    # print(2, time.time() - start_time)
    # start_time = time.time()

    g.ndata["pos_undirected_1"] = b1_compr.float()
    g.ndata["pos_undirected_2"] = b2_to_compr.float()
    
    # #test 9
    # b1 ,b2 , faces = slhelp.compute_hodge_matrix_2(g.num_nodes(),g.edges())
    # b1, b2 = np.abs(b1), np.abs(b2)
    # b2_to = b1 @ b2
    # b1 = torch.tensor(b1)
    # b2_to = torch.tensor(b2_to)
    
    # kl = min(b1.shape[0], hidden_size)
    # kr = min(b1.shape[1], hidden_size)
    # # import time
    # # start_time = time.time()
    # b1_compr = pca_fbpca(kl, kr, b1, hidden_size, retry)
    # # print(1, time.time() - start_time)
    # # start_time = time.time()
    # kl = min(b2_to.shape[0], hidden_size)
    # kr = min(b2_to.shape[1], hidden_size)
    # b2_to_compr = pca_fbpca(kl, kr, b2_to, hidden_size, retry)
    # # print(2, time.time() - start_time)
    # # start_time = time.time()

    # g.ndata["pos_undirected_1"] = b1_compr.float()
    # g.ndata["pos_undirected_2"] = b2_to_compr.float()
    
    # #test 10
    # b1 ,b2 , faces = slhelp.compute_hodge_matrix_2(g.num_nodes(),g.edges())
    # L0, L1f = slSIM.compute_bunch_matrices(b1, b2) #using normalize matrices might try unnormalized     weird shapes
    # b1 = torch.tensor(b1)
    # b2 = torch.tensor(b2)
    # L1f_to = b1 @ torch.tensor(L1f)
    # b2_to =  torch.abs(b1) @  torch.abs(b2) @  (torch.transpose(b2,0,1) @ b2)
    # b1, b2 = torch.abs(b1), torch.abs(b2)
    # L1f_to = b1 @ L1f

    # kl = min(b2_to.shape[0], hidden_size)
    # kr = min(b2_to.shape[1], hidden_size)
    # # import time
    # # start_time = time.time()
    # b2_to_compr = pca_torch(kl, kr, b2_to, hidden_size, retry)
    # # print(1, time.time() - start_time)
    # # start_time = time.time()
    # kl = min(L1f_to.shape[0], hidden_size)
    # kr = min(L1f_to.shape[1], hidden_size)
    # L1f_to_compr = pca_torch(kl, kr, L1f_to, hidden_size, retry)
    # # print(2, time.time() - start_time)
    # # start_time = time.time()

    # g.ndata["pos_undirected_1"] = b2_to_compr.float()
    # g.ndata["pos_undirected_2"] = L1f_to_compr.float()


    
    return g
