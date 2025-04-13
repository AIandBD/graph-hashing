import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm
from javalang.ast import Node
from ast import *
from tqdm import tqdm
import itertools
from scipy.sparse import csr_matrix
from collections import Counter
from tree import ASTNode, SingleNode, BlockNode



class Tree2vec:
    def __init__(self, root, dataset):
        self.root = root
        self.dataset = dataset

    def code2ast(self, output_file, option):
        path = self.root + '/' + self.dataset + '/' + output_file

        if os.path.exists(path) and option == 'existing':
            source = pd.read_pickle(path)

        else:
            if self.dataset == 'POJ104':
                from pycparser import c_parser
                parser = c_parser.CParser()
                source = pd.read_pickle(self.root + '/' + self.dataset + '/programs.pkl')
                source.columns = ['id', 'code', 'label']
                tqdm.pandas(desc="code2ast")
                source['code'] = source['code'].progress_apply(parser.parse)

            elif self.dataset in ['GCJ', 'Java250']:
                import javalang
                source = pd.read_pickle(
                    self.root + '/' + self.dataset + '/' + 'programs.pkl')
                source.columns = ['id', 'code', 'label']
                tqdm.pandas(desc="code2ast")
                source['code'] = source['code'].progress_apply(javalang.parse.parse)

        source.to_pickle(path)
        return source

    def tree2bracket(self):
        trees = pd.read_pickle(self.root + '/' + self.dataset + '/' + "ast.pkl")
        def get_gcj_token(node):
            token = ''
            if isinstance(node, str):
                token = node
            elif isinstance(node, set):
                token = 'Modifier'
            elif isinstance(node, Node):
                token = node.__class__.__name__
            return token

        def get_gcj_children(root):
            if isinstance(root, Node):
                children = root.children
            elif isinstance(root, set):
                children = list(root)
            else:
                children = []

            def expand(nested_list):
                for item in nested_list:
                    if isinstance(item, list):
                        for sub_item in expand(item):
                            yield sub_item
                    elif item:
                        yield item

            return list(expand(children))

        def get_oj_sequences(node, sequence):
            current = SingleNode(node)
            sequence.append(current.get_token())
            for _, child in node.children():
                get_oj_sequences(child, sequence)
            if current.get_token().lower() == 'compound':
                sequence.append('End')

        def get_gcj_sequences(node, sequence):
            token, children = get_gcj_token(node), get_gcj_children(node)
            sequence.append(token)
            for child in children:
                get_gcj_sequences(child, sequence)
            if token in ['ForStatement', 'WhileStatement', 'DoStatement', 'SwitchStatement', 'IfStatement']:
                sequence.append('End')

        def trans_to_sequences(ast):
            sequence = []
            if self.dataset == 'POJ104':
                get_oj_sequences(ast, sequence)
            else:
                get_gcj_sequences(ast, sequence)
            return sequence

        corpus = trees['code'].apply(trans_to_sequences)
        str_corpus = [' '.join(c) for c in corpus]
        trees['code'] = pd.Series(str_corpus, index=corpus.index)
        trees.to_csv(self.root + '/' + self.dataset + '/' + 'programs_ns.tsv')
        from gensim.models.word2vec import Word2Vec
        size = 300
        w2v = Word2Vec(corpus, vector_size=size, workers=16, sg=1, min_count=3)
        w2v.save(self.root + '/' + self.dataset + '/' + 'node_w2v_' + str(size))
        word2vec = Word2Vec.load(self.root + '/' + self.dataset + '/' + 'node_w2v_' + str(size)).wv
        vocab = word2vec.key_to_index

        file = open(self.root + '/' + self.dataset + '/' +'vocab.txt', 'w')
        file.write(str(vocab))
        max_token = word2vec.vectors.shape[0]

        def get_oj_blocks(node, block_seq):
            children = node.children()
            name = node.__class__.__name__
            if name in ['FuncDef', 'If', 'For', 'While', 'DoWhile']:
                block_seq.append(ASTNode(node))
                if name != 'For':
                    skip = 1
                else:
                    skip = len(children) - 1

                for i in range(skip, len(children)):
                    child = children[i][1]
                    if child.__class__.__name__ not in ['FuncDef', 'If', 'For', 'While', 'DoWhile', 'Compound']:
                        block_seq.append(ASTNode(child))
                    get_oj_blocks(child, block_seq)
            elif name == 'Compound':
                block_seq.append(ASTNode(name))
                for _, child in node.children():
                    if child.__class__.__name__ not in ['If', 'For', 'While', 'DoWhile']:
                        block_seq.append(ASTNode(child))
                    get_oj_blocks(child, block_seq)
                block_seq.append(ASTNode('End'))
            else:
                for _, child in node.children():
                    get_oj_blocks(child, block_seq)

        def get_gcj_blocks(node, block_seq):
            name, children = get_gcj_token(node), get_gcj_children(node)
            logic = ['SwitchStatement', 'IfStatement', 'ForStatement', 'WhileStatement', 'DoStatement']
            if name in ['MethodDeclaration', 'ConstructorDeclaration']:  
                block_seq.append(BlockNode(node))
                body = node.body
                if body:
                    for child in body:
                        if get_gcj_token(child) not in logic and not hasattr(child, 'block'):
                            block_seq.append(BlockNode(child))
                        else:
                            get_gcj_blocks(child, block_seq)
            elif name in logic:
                block_seq.append(BlockNode(node))
                for child in children[1:]:
                    token = get_gcj_token(child)
                    if not hasattr(node, 'block') and token not in logic + ['BlockStatement']:
                        block_seq.append(BlockNode(child))
                    else:
                        get_gcj_blocks(child, block_seq)
                    block_seq.append(BlockNode('End'))
            elif name == 'BlockStatement' or hasattr(node, 'block'):
                block_seq.append(BlockNode(name))
                for child in children:
                    if get_gcj_token(child) not in logic:
                        block_seq.append(BlockNode(child))
                    else:
                        get_gcj_blocks(child, block_seq)
            else:
                for child in children:
                    get_gcj_blocks(child, block_seq)

        def tree_to_index(node):
            token = node.token
            result = [vocab[token] if token in vocab else max_token]
            children = node.children
            for child in children:
                result.append(tree_to_index(child))
            return result

        def trans2seq(r):
            blocks = []
            if self.dataset == 'POJ104':
                get_oj_blocks(r, blocks)
            else:
                get_gcj_blocks(r, blocks)
            tree = []
            for b in blocks:
                btree = tree_to_index(b)
                tree.append(btree)
            return tree

        trees = pd.read_pickle(self.root + '/' + self.dataset + '/' + 'ast.pkl')
        trees['code'] = trees['code'].apply(trans2seq)

        def add_rootnode(tree):
            rootnode = [0]
            rootnode.extend(tree)
            return rootnode

        trees['code'] = trees['code'].apply(add_rootnode)
        trees.to_pickle(self.root + '/' + self.dataset + '/' + 'tree_bracket.pkl')
        return trees

    def bracket2csr(self):
        def fun(Tree, row, col, data, row_id):
            global col_id
            if (len(Tree) < 1):
                return
            row.extend([col_id - 1])
            col.extend([col_id - 1])
            data.extend([Tree[0]])
            for i in range(1, len(Tree)):
                if (len(Tree) == 1):
                    row.extend([row_id])
                    col.extend([col_id])
                    data.extend([Tree[0]])
                    return
                else:
                    row.extend([row_id])
                    col.extend([col_id])
                    data.extend([Tree[i][0]])
                    col_id = col_id + 1
                    row_id = col[-1] - 1
                    fun(Tree[i], row, col, data, row_id + 1)
            return row, col, data

        def tomatrix(Tree):
            global col_id
            col_id = 1
            row, col, data = fun(Tree, [], [], [], 0)
            csr = csr_matrix((data, (row, col)))
            return csr

        source = pd.read_pickle(self.root + '/' + self.dataset + '/' + 'tree_bracket.pkl')
        tqdm.pandas(desc="transform to csr")
        source['code'] = source['code'].progress_apply(tomatrix)
        self.sources = source
        source.to_pickle(self.root + '/' + self.dataset + '/' + 'csr_matrix.pkl')


    def run(self):
        print('code to ast...generated ast.pkl ')
        self.code2ast(output_file='ast.pkl', option='existing')
        print('ast to bracket...generated vocab.txt and tree_bracket.pkl')
        self.tree2bracket()
        print('bracket to csr matrix...generated csr_matrix.pkl')
        self.bracket2csr()

#datasets: POJ104, GCJ, Java250
t2v = Tree2vec(root='data', dataset='GCJ')
t2v.run()


