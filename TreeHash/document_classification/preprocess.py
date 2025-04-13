from scipy.sparse import csr_matrix
from tqdm import tqdm
import numpy as np
import itertools
from itertools import chain
def preprocess(dataset):
    path = "./data/"
    data_path = path + dataset
    def build_adjacency(dataset,node_num):
        data_path = path + dataset
        file = open(data_path + '/graph.txt', "r")
        contents = file.readlines()
        edge_index = []
        for content in contents:
            edge_index.extend([[int(content.split("\t")[0]), int(content.split("\t")[1].split('\n')[0])]])
            edge_index.extend([[int(content.split("\t")[1].split('\n')[0]), int(content.split("\t")[0])]])
        edge_index = list(k for k, _ in itertools.groupby(sorted(edge_index)))
        edge_index = np.asarray(edge_index)
        adjacency = csr_matrix((np.ones(len(edge_index)), (edge_index[:, 0], edge_index[:, 1])),
                               shape=(node_num, node_num), dtype="float32")
        adjacency.setdiag(1)
        return adjacency

    def get_feature(path, delimiter):
        fp = open(path, 'r', encoding='utf-8')
        content = fp.read()
        fp.close()
        rowlist = content.splitlines()
        recordlist = [row.split(delimiter) for row in rowlist if row.strip()]
        recordArray = np.delete(np.array(recordlist), -1, axis=1).astype(int)
        return recordArray

    if dataset in['dblp','NELL']:
        from torch_geometric.datasets import Planetoid, CitationFull, NELL
        if dataset =='dblp':
            data = CitationFull(root=path, name="DBLP")[0]
            feature = csr_matrix(np.array(data.x))
        if dataset =='NELL':
            data = NELL(root=data_path)[0]
            feature = data.x.to_scipy().tocsr()

        label = data.y.numpy()
        node_num = len(label)
        row = data.edge_index[0].tolist()
        col = data.edge_index[1].tolist()
        d = len(col) * [1]
        csr_graph = csr_matrix((d, (row, col)), shape=(node_num, node_num))
        csr_graph.setdiag(1)
        csr_graph = csr_graph + csr_graph.T.multiply(csr_graph.T > csr_graph) - csr_graph.multiply(
            csr_graph.T > csr_graph)

    else:
        feature = csr_matrix(get_feature(data_path + '/feature.txt', '\t'))
        label = []
        file = open(data_path + '/group.txt', 'r')
        contents = file.readlines()
        for content in contents:
            label.append(int(content.split('\t')[1]))
        label = np.array(label).T
        node_num = len(label)
        csr_graph = build_adjacency(dataset, node_num)

    if dataset == 'NELL':
        tree_depth = 2
    else:
        tree_depth = 3
    tree = []
    for n in range(0, node_num):
        level_nodes = [n]
        nodes = [[n]]
        temp = [n]
        multipy_list = [1, 2, 2]
        for d in range(1, tree_depth):
            level_nodes = list(map(lambda x: list(csr_graph[x].indices), level_nodes))
            nodes.append(level_nodes[0]*multipy_list[d])
        nodes = list(chain.from_iterable(nodes))
        temp.append(nodes)
        tree.append(temp)
    return tree, feature, label


