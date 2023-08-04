import re
from functools import reduce
from operator  import methodcaller, xor

import networkx as nx
import numpy    as np

from .       import qubo
from ._misc  import make_upper_triangle
from .bitvec import _BITVEC_EXPR


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

    __NODE_NAME_PATTERN  = re.compile(r'(x(0$|([1-9]\d*)))|1$')

    def __init__(self, graph: nx.DiGraph):
        # check if `graph` is a valid PAG (=Partial Assignment Graph)
        assert all(self.__NODE_NAME_PATTERN.match(u) is not None for u in graph.nodes), \
            '`graph` contains invalid node names'
        self.__PAG = self.__normalize_graph(graph)
        self.__dirty = False

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
                # ensure that higher indices always point to lower indices
                if u != '1' and int(node[1:]) < int(u[1:]):
                    # reverse edge
                    G.add_edge(u, node, inverse=inv)
                    # flip all edges incident to u
                    for w, _, data in list(G.in_edges(u, data=True)):
                        G.remove_edge(w, u)
                        G.add_edge(w, node, **data)
                else:
                    G.add_edge(node, u, inverse=inv)
        return G

    def __dirty(func):
        def go(self, *args, **kwargs):
            self.__dirty = True
            return func(self, *args, **kwargs)
        return go

    def __assert_normalized(func):
        def go(self, *args, **kwargs):
            dirty = getattr(self, '__dirty', True)
            if dirty:
                self.__PAG = self.__normalize_graph(self.__PAG)
                self.__dirty = False
            return func(self, *args, **kwargs)
        return go

    @__assert_normalized
    def __repr__(self):
        s = ''
        const_zero, const_one = [], []
        for u, _, data in self.__PAG.in_edges('1', data=True):
            (const_zero if data['inverse'] else const_one).append(u)
        if const_zero: s += ', '.join(const_zero) + ' = 0;\n'
        if const_one:  s += ', '.join(const_one)  + ' = 1;\n'
        for v in self.__PAG.nodes():
            if v == '1': continue
            us_pos, us_neg = [], []
            for u, _, data in self.__PAG.in_edges(v, data=True):
                (us_neg if data['inverse'] else us_pos).append(u)
            if us_pos: s += ', '.join(us_pos) + ' = '   + v + ';\n'
            if us_neg: s += ', '.join(us_neg) + ' != ' + v + ';\n'
        return s[:-1] # remove trailing newline

    @property
    def size(self):
        return self.__PAG.number_of_nodes()-1

    @property
    @__assert_normalized
    def num_free(self):
        nodes_with_out_edges = { u for u, _ in self.__PAG.edges }
        return self.__PAG.number_of_nodes()-len(nodes_with_out_edges)-1

    @__dirty
    def assign_constant(self, u: int, const):
        const_ = str(int(const))
        assert const_ in '01', '`const` must be 0 or 1'
        self.__PAG.remove_edges_from(list(self.__PAG.out_edges(f'x{u}')))
        self.__PAG.add_edge(f'x{u}', '1', inverse=const_=='0')

    @__dirty
    def assign_index(self, u: int, v: int, inverse=False):
        self.__PAG.remove_edges_from(list(self.__PAG.out_edges(f'x{u}')))
        self.__PAG.add_edge(f'x{u}', f'x{v}', inverse=inverse)

    @__assert_normalized
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

    @classmethod
    def from_string(cls, s: str, n: int=None):
        assignments = s.split(';')
        for assignment in assignments:
            left, right = assignment.split('=')
            # TODO
            raise NotImplementedError()

    @classmethod
    def infer(cls, x: np.ndarray):
        raise NotImplementedError() # <- Nico

    @classmethod
    def simplify_expression(cls, expr: str):
        pa = cls.from_expression(expr)
        return pa.to_expression()

    @__assert_normalized
    def to_matrix(self):
        # construct transformation matrix
        T = np.zeros((self.size, self.num_free+1))
        one = T.shape[1]-1 # index of constant 1
        j = 0
        for i in range(self.size):
            u = f'x{i}'
            try:
                _, v, data = next(iter(self.__PAG.edges(u, data=True)))
            except StopIteration:
                # no outgoing edges -> free variable
                T[i, j] = 1.0
                j += 1
                continue
            if v == '1':
                if not data['inverse']:
                    T[i, one] = 1.0
            else:
                l = int(v[1:])
                if data['inverse']:
                    T[i, l] = -1.0
                    T[i, one] = 1.0
                else:
                    T[i, l] = 1.0
        return T

    @__assert_normalized
    def apply(self, arg: qubo | np.ndarray):
        if isinstance(arg, qubo):
            Q = arg
            assert Q.n == self.size, 'Size of partial assignment does not match QUBO size'
            T = self.to_matrix()
            m = make_upper_triangle(T.T @ Q.m @ T)
            # eliminate constant 1 from matrix (last row and column)
            offset = m[-1, -1]
            return qubo(np.diag(m[:-1, -1]) + m[:-1, :-1]), offset
        else:
            assert arg.shape[-1] == self.num_free
            T = self.to_matrix()
            return T @ np.concatenate((arg, np.ones((*arg.shape[:-1], 1))), axis=-1)