import gcc
import torch
import networkx as nx
from  torch_geometric.data import Data
import numpy as np
import torch.nn.functional as F
from simplex_lists.SIMBLOCKGNN import compute_bunch_matrices
from simplex_lists.incidence_matrix import get_faces, incidence_matrices
import os
import dgl

import scipy.sparse as sparse



def compute_hodge_matrix_2(num_nodes, edges):
    g = nx.Graph()
    g.add_nodes_from([i for i in range(num_nodes)])
    edge_index_ = torch.transpose(torch.stack(edges,dim=1),0,1)
    edge_index = [(edge_index_[0, i].item(), edge_index_[1, i].item()) for i in
                        range(np.shape(edge_index_)[1])]
    g.add_edges_from(edge_index)

    edge_to_idx = {edge: i for i, edge in enumerate(g.edges)}
    
    faces = get_faces(g)

    B1, B2 = incidence_matrices(g, sorted(g.nodes), sorted(g.edges), faces, edge_to_idx)

    return B1, B2, faces

def compute_hodge_matrix_3(num_nodes, edges):
    g = nx.Graph()
    g.add_nodes_from([i for i in range(num_nodes)])
    edge_index_ = torch.transpose(torch.stack(edges,dim=1),0,1)
    edge_index = [(edge_index_[0, i].item(), edge_index_[1, i].item()) for i in
                        range(np.shape(edge_index_)[1])]
    g.add_edges_from(edge_index)
    edge_to_idx = {edge: i for i, edge in enumerate(g.edges)}
    faces = get_faces(g)
    B1, B2 = incidence_matrices_3(g, sorted(g.nodes), sorted(g.edges), faces, edge_to_idx)
    return B1, B2, faces

def incidence_matrices_3(G, V, E, faces, edge_to_idx):
    """
    Returns incidence matrices B1 and B2

    :param G: NetworkX DiGraph
    :param V: list of nodes
    :param E: list of edges
    :param faces: list of faces in G

    Returns B1 (|V| x |E|) and B2 (|E| x |faces|)
    B1[i][j]: -1 if node is is tail of edge j, 1 if node is head of edge j, else 0 (tail -> head) (smaller -> larger)
    B2[i][j]: 1 if edge i appears sorted in face j, -1 if edge i appears reversed in face j, else 0; given faces with sorted node order
    """
    B1 = sparse.csr_matrix(nx.incidence_matrix(G, nodelist=V, edgelist=E, oriented=True))
    B2 = sparse.csr_matrix(np.zeros([len(E),len(faces)]))

    for f_idx, face in enumerate(faces): # face is sorted
        edges = [face[:-1], face[1:], [face[0], face[2]]]
        e_idxs = [edge_to_idx[tuple(e)] for e in edges]

        B2[e_idxs[:-1], f_idx] = 1
        B2[e_idxs[-1], f_idx] = -1
    return B1, B2

def save(file_path,name, B1, B2, faces):
    print(file_path,name)
    state = {
        "name": name,
        "B1": B1,
        "B2": B2,
        "faces": faces
    }
    save_file = os.path.join(file_path, name+".pt")
    if not os.path.isdir(file_path):
        os.makedirs(file_path)
    torch.save(state, save_file,pickle_protocol=4)
        
        
def node_based(dataset,name):
    g = dgl.to_bidirected(dataset.graphs[0])
    b1,b2,faces = compute_hodge_matrix_2(g.num_nodes(),g.edges())
    print(faces)
    print(b1)
    print(b2)
    save('/home/shakir/simplical_complices_gcc/prefaces',name,b1,b2,faces )
    

def graph_based(dataset,name):
    for g in range(len(dataset.graphs)):
        g2 = dgl.to_bidirected(dataset.graphs[g])
        b1,b2,faces = compute_hodge_matrix_2(g2.num_nodes(),g2.edges())
        save('/home/shakir/simplical_complices_gcc/prefaces/'+ name ,name+str(g),b1,b2,faces )
    
    
def pretrain_based(graphlist,name):
    for g in range(5,0,-1):
        g2 = dgl.to_bidirected(graphlist[0][g])
        print(g2.num_nodes(),g2.num_edges(),flush=True)
        b1,b2,faces = compute_hodge_matrix_2(g2.num_nodes(),g2.edges())
        save('/home/shakir/simplical_complices_gcc/prefaces/'+ name ,name+str(g),b1,b2,faces)    

    


def load_data_from_gcc(pos_feat, edge_list):
    return Data(x = pos_feat, edge_index=edge_list)

# def save_blocks():
#     # random_edge_num = 2500
#     # indices = np.random.choice((data.edge_index).size(1), (random_edge_num,), replace=False)
#     # indices = np.sort(indices)
#     # sample_data_edge_index = data.edge_index[:, indices]
#     # boundary_matrix0_, boundary_matrix1_ = compute_hodge_matrix(data, sample_data_edge_index)
    
#     boundary_matrix0_, boundary_matrix1_ = compute_hodge_matrix(data, sample_data_edge_index)

    
#     L0u, L1f = compute_bunch_matrices(boundary_matrix0_, boundary_matrix1_)
#     L0u = torch.tensor(L0u, dtype=torch.float32)  # convert hodge matrix to tensor format
#     L1f = torch.tensor(L1f, dtype=torch.float32)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     boundary_matrics=[L0u.to(device), L1f.to(device)]
#     L0, L1 = boundary_matrics
    
#     # L0_r = torch.matrix_power(L0, 2)
#     # L1_r = torch.matrix_power(L1, 2)
#     # relation_embedded = torch.einsum('xd, dy -> xy', torch.matmul(L0_r, self.weights_L_0), torch.matmul(L1_r, self.weights_L_1).transpose(0, 1))
#     # relation_embedded_ = torch.matmul(self.weights_off_diagonal, relation_embedded)
#     # upper_block = torch.cat([L0_r, relation_embedded_], dim=1)
#     # lower_block = torch.cat([torch.transpose(relation_embedded_, 0, 1), L1_r], dim=1)
#     # sim_block = torch.cat([upper_block, lower_block], dim=0)
#     # sim_block = F.softmax(F.relu(sim_block), dim = 1)
    
    
#     # self.weights_sim = nn.Parameter(torch.FloatTensor(int(boundary_matrix_size*2), dimension))
#     # self.embeddings_sim = nn.Parameter(torch.FloatTensor(data.x.size(1), int(boundary_matrix_size*2)))
#     # self.weights_off_diagonal = nn.Parameter(torch.FloatTensor(int(boundary_matrix_size), int(boundary_matrix_size)))
#     # self.weights_L_0 = nn.Parameter(torch.FloatTensor(int(boundary_matrix_size), 32))
#     # self.weights_L_1 = nn.Parameter(torch.FloatTensor(int(boundary_matrix_size), 32))
#     # # reset parameters
#     # nn.init.kaiming_uniform_(self.weights_sim, mode = 'fan_out', a = math.sqrt(5))
#     # nn.init.kaiming_uniform_(self.embeddings_sim, mode='fan_out', a=math.sqrt(5))
#     # nn.init.kaiming_uniform_(self.weights_off_diagonal, mode='fan_out', a=math.sqrt(5))
#     # nn.init.kaiming_uniform_(self.weights_L_0, mode='fan_out', a=math.sqrt(5))
#     # nn.init.kaiming_uniform_(self.weights_L_1, mode='fan_out', a=math.sqrt(5))

# if __name__ == "__main__":

