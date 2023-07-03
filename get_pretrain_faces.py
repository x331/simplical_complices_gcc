import dgl
from gcc.datasets.graph_dataset import GraphClassificationDataset
from simplex_lists.helpers import pretrain_based, graph_based


if __name__ == "__main__":

    gs = dgl.data.utils.load_graphs("/home/shakir/simplical_complices_gcc/data/small.bin")
    pretrain_based(gs,'dgl')
    # train_dataset = GraphClassificationDataset('imdb-multi')
    # graph_based(train_dataset,'imdb-multi')