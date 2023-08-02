from functools import reduce
from operator  import xor

import networkx as nx

from .bitvec import _BITVEC_EXPR, _expr_normal_form


def _follow_edges(G: nx.DiGraph, u):
    v, inv = u, False
    while True:
        try:
            _, v, data = next(iter(G.edges(v, data=True)))
            inv ^= data['inverse']
        except StopIteration:
            break
    return v, inv


class partial_assignment:
    def __init__(self, graph: nx.DiGraph):
        self.__PAG = self.__normalize_graph(graph) # PAG = Partial Assignment Graph

    def __normalize_graph(self, G: nx.DiGraph):
        for nodes in nx.simple_cycles(G):
            edges = list(zip(nodes, nodes[1:]+[nodes[0]]))
            conflict = reduce(xor, [G.get_edge_data(*e)['inverse'] for e in edges])
            if conflict:
                raise RuntimeError('Partial Assignment Graph contains conflicting cycle!')
            # resolve cycle by deleting an edge
            G.remove_edge(*edges[0])

        # resolve chained references as far as possible
        # TODO: This could be more efficient…
        for node in G.nodes:
            u, inv = _follow_edges(G, node)
            if u != node:
                v, = G.neighbors(node)
                G.remove_edge(node, v)
                G.add_edge(node, u, inverse=inv)

        # TODO: Flip edges from lower to higher indices
        # ??? Is the structure now a tree?
        return G

    def to_expression(self):
        n = self.__PAG.number_of_nodes()-1
        nodes = [f'x{i}' for i in range(n)]
        expr = ''
        for u in nodes:
            try:
                _, v, data = next(iter(self.__PAG.edges(u, data=True)))
                if v == '1':
                    expr += '0' if data['inverse'] else '1'
                else:
                    expr += f'[{"!" if data["inverse"] else ""}{v[1:]}]'
            except StopIteration:
                expr += '*'
        return expr

    @classmethod
    def from_expression(cls, expr: str):
        tokens = list(_BITVEC_EXPR.findall(expr))
        G = nx.DiGraph()
        G.add_node('1')
        for i, token in enumerate(tokens):
            xi = f'x{i}'
            G.add_node(xi)
            if token == '*':
                continue
            if token == '0':
                G.add_edge(xi, '1', inverse=True); continue
            if token == '1':
                G.add_edge(xi, '1', inverse=False); continue
            xj = 'x' + token.strip('[!]')
            G.add_edge(xi, xj, inverse='!' in token)
        return cls(G)