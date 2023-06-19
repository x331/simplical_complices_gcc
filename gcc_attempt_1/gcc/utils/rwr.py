import dgl 
import torch

#returns a trace of a random walk with restarts for a given starting node and only works for homogenous graphs       
def rwr_for_one_node(graph, nodes, restart_prob, length):
    rwr = None
    run = dgl.sampling.random_walk(
        g = graph,
        nodes=nodes,
        restart_prob=restart_prob,
        length=length,
    )[0][0]
    endOfRun = (run == -1).nonzero(as_tuple=False)[0]
    rwr = torch.clone(run[:endOfRun])
    while rwr.size(0) < length:
        run = dgl.sampling.random_walk(
            g = graph,
            nodes=nodes,
            restart_prob=restart_prob,
            length=length,
        )[0][0]
        endOfRun = (run == -1).nonzero(as_tuple=False)[0]
        rwr = torch.cat((rwr,run[:endOfRun]), dim=0)
    return rwr[:length]
    

    
    
