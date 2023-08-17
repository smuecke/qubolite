import re
from functools import reduce
from itertools import combinations, groupby, repeat
from operator  import methodcaller, xor

import networkx as nx
import numpy    as np

from .       import qubo
from ._misc  import get_random_state, make_upper_triangle, to_shape


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
    """This class represents bit vectors of a fixed size ``n`` where a number of
    bits at certain positions are either fixed to a constant value (0 or 1), or 
    tied to the value of another bit (or its inverse). The bits that are not
    fixed or tied are called *free variables*.

    The preferred way to instantiate a partial assignment is through the str
    argument or through a bit vector expression (see :meth:`assignment.partial_assignment.from_expression`).
    However, you can specify a partial assignment graph using the ``graph`` argument.

    Args:
        s (str, optional): String representation of a partial assignment; see 
            examples below for the format.
        n (int, optional): The minimum number of bits of the bit vector; e.g.,
            if only ``x2 = 1`` is specified, by setting ``n=5``, the partial 
            assignment will have 5 bits (i.e., ``**1**``). If None (default), 
            use the highest bit index to determine the size. If the highest 
            index is greater than ``n``, then it will be used instead of ``n``.
        graph (networkx.DiGraph, optional): Directed graph representing a partial variable
            assignment. The nodes must be labeled ``"x0"``, ``"x1"``, etc., up to
            some ``n-1`` for all bit variables. Additionally, there must be a
            node labeled ``"1"``. An edge from ``"x3"`` to ``"1"`` means that
            bit 3 is fixed to 1, and an edge from ``"x5"`` to ``"x4"`` means that
            bit 5 is tied to bit 4. Every edge must have a boolean attribute
            ``inverse`` which indicates if the relation holds inversely, i.e.,
            an edge from ``"x3"`` to ``"x4"`` with ``inverse=True`` means that
            bit 3 is always the opposite of bit 4. The preferred way to create
            instances is through ``from_expression`` or ``from_string``. Only
            use the constructor if you know what you are doing. If specified,
            ``s`` and ``n`` will be ignored.

    Examples:
        The string representation of a partial assignment consists of a list of 
        bit assignments separated by semicola (``;``). A bit assignment consists 
        of a variable or a comma-separated list of bit variables followed by ``=`` 
        or ``!=`` and then a single bit variable or ``0`` or ``1``. A bit 
        variable consists of the letter ``x`` followed by a non-negative integer 
        denoting the bit index. Additionally you can specify ranges of 
        consecutive bit variables like ``x{<i>-<j>}``, where ``<i>`` and ``<j>``
        are the start and stop index (inclusive) respectively.
        The following lines are all valid strings:

            x0 = 1
            x2, x3, x5 != x8; x6 = 0
            x4=x3;x10=1
            x{2-6}, x8 = x0; x7 != x0

        The partial assignment can then be instantiated like this:

        >>> PA = partial_assignment('x4!=x5; x1=0', n=10)
        >>> PA
        x1 = 0; x5 != x4
        >>> PA.to_expression()
        *0***[!4]****
    """

    __BITVEC_EXPR = re.compile(r'[01*]|\[!?\d+\]|\{\d+\}')
    __NODE_NAME_PATTERN = re.compile(r'(x(0$|([1-9]\d*)))|1$')

    def __init__(self, s: str=None, n: int=None, *, graph: nx.DiGraph=None):
        if graph is not None:
            # check if `graph` is a valid PAG (=Partial Assignment Graph)
            assert all(self.__NODE_NAME_PATTERN.match(u) is not None for u in graph.nodes), \
                '`graph` contains invalid node names'
            self.__PAG = graph
        else:
            self.__PAG = self.__from_string(s, n)
        self.__dirty = True
        self.grouping_limit = 10

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
        if const_zero: s += ', '.join(const_zero) + ' = 0; '
        if const_one:  s += ', '.join(const_one)  + ' = 1; '
        for v in self.__PAG.nodes():
            if v == '1': continue
            us_pos, us_neg = [], []
            for u, _, data in self.__PAG.in_edges(v, data=True):
                (us_neg if data['inverse'] else us_pos).append(u)
            if us_pos: s += ', '.join(us_pos) + ' = '   + v + '; '
            if us_neg: s += ', '.join(us_neg) + ' != ' + v + '; '
        return s[:-2] # remove trailing separator

    def __from_string(self, s: str, n: int=None):
        def unpack_ranges(nodes):
            # handle expressions like 'x{1-5}'
            for node in nodes:
                if '{' in node:
                    start, stop = node[2:-1].split('-')
                    yield from [f'x{i}' for i in range(int(start), int(stop)+1)]
                else:
                    yield node
        PAG = nx.DiGraph()
        PAG.add_node('1')
        assignments = s.split(';')
        for assignment in assignments:
            try:
                left, right = assignment.split('!=')
                inv = True
            except ValueError:
                if assignment.strip() == '':
                        continue
                left, right = assignment.split('=')
                inv = False
            nodes_left = map(methodcaller('strip'), left.split(','))
            node_right = right.strip()
            if node_right == '0':
                v = '1'
                inv = True
            else:
                v = node_right
            PAG.add_edges_from([(u, v, {'inverse': inv})
                                for u in unpack_ranges(nodes_left)])
        # add potentially missing nodes
        n_ = 1 + int(max([u for u in PAG.nodes() if u!='1'],
                         key=lambda x: int(x[1:]))[1:])
        n = n_ if n is None else max(n, n_)
        PAG.add_nodes_from([f'x{i}' for i in range(n)])
        return PAG

    @property
    def size(self):
        """Total number of bits described by the partial assignment, including
        fixed/tied and free bits.

        Returns:
            int: Number of bits.
        """
        return self.__PAG.number_of_nodes()-1

    @property
    @__assert_normalized
    def free(self):
        free_nodes = set(range(self.size)).difference([int(u[1:]) for u, _ in self.__PAG.edges])
        return np.fromiter(sorted(free_nodes), dtype=int)

    @property
    @__assert_normalized
    def num_free(self):
        nodes_with_out_edges = { u for u, _ in self.__PAG.edges }
        return self.__PAG.number_of_nodes()-len(nodes_with_out_edges)-1

    @property
    @__assert_normalized
    def fixed(self):
        nodes_with_out_edges = { int(u[1:]) for u, _ in self.__PAG.edges }
        return np.fromiter(sorted(nodes_with_out_edges), dtype=int)
    
    @property
    @__assert_normalized
    def num_fixed(self):
        nodes_with_out_edges = { u for u, _ in self.__PAG.edges }
        return len(nodes_with_out_edges)

    @__dirty
    def assign_constant(self, u: int, const):
        """Fix a bit at the given index to a constant 0 or 1. If the index is
        larger than the current size, add all intermediate bits.

        Args:
            u (int): Bit index.
            const (int or str): Constant value; must be 0 or 1.
        """
        const_ = str(int(const))
        assert const_ in '01', '`const` must be 0 or 1'
        self.__PAG.add_nodes_from([f'x{i}' for i in range(self.size, u)])
        self.__PAG.remove_edges_from(list(self.__PAG.out_edges(f'x{u}')))
        self.__PAG.add_edge(f'x{u}', '1', inverse=const_=='0')

    @__dirty
    def assign_index(self, u: int, v: int, inverse=False):
        """Tie a bit to another. If either of the indices is larger than the
        current size, add all intermediate bits.

        Args:
            u (int): Bit index to tie to another index
            v (int): Second bit index  that ``u`` is tied to.
            inverse (bool, optional): Indicates if the first bit is tied
                inversely to the second. Defaults to False.
        """
        self.__PAG.add_nodes_from([f'x{i}' for i in range(self.size, max(u, v))])
        self.__PAG.remove_edges_from(list(self.__PAG.out_edges(f'x{u}')))
        self.__PAG.add_edge(f'x{u}', f'x{v}', inverse=inverse)

    @__assert_normalized
    def to_expression(self):
        n = self.__PAG.number_of_nodes()-1
        nodes = [f'x{i}' for i in range(n)]
        tokens = []
        for u in nodes:
            try:
                _, v, data = next(iter(self.__PAG.edges(u, data=True)))
                if v == '1':
                    tokens.append('0' if data['inverse'] else '1')
                else:
                    tokens.append(f'[{"!" if data["inverse"] else ""}{v[1:]}]')
            except StopIteration:
                tokens.append('*')
        # group tokens
        expr = ''
        for token, g in groupby(tokens):
            rep = len(list(g))
            if rep >= self.grouping_limit:
                expr += f'{token}{{{rep}}}'
            else:
                expr += token*rep
        return expr

    @classmethod
    def _ungrouped_tokens(cls, expr):
        for token in cls.__BITVEC_EXPR.findall(expr):
            if token.startswith('{'):
                rep = int(token[1:-1])
                assert rep >= 1, 'repetition numbers must be >= 1'
                yield from repeat(last_token, rep-1)
            else:
                yield token
                last_token = token

    @classmethod
    def from_expression(cls, expr: str):
        """Generate a partial assignment from a string containing a bit vector
        expression. Such an expression consists of a sequence of these tokens: ``0``
        - a constant 0, ``1`` - a constant 1, ``*`` - all combinations of 0 and 1,
        ``[i]`` - the same as the bit at index i, ``[!i]`` - the inverse of the bit
        at index i.

        The last two tokens are called references, with ``i`` being their pointing
        index (counting from 0), where ``i`` refers to the i-th token of the
        bitvector expression itself. Note that a ``RuntimeError`` is raised if there
        is a circular reference chain.

        If a token is repeated, you can use specify a number of repetitions in
        curly braces, e.g., ``*{5}`` is the same as ``*****``. This also works with
        references.

        Args:
            expr (str): Bit vector expression.

        Returns:
            partial_assignment: The partial assignment described by the
                expression.

        Examples:
            This function is useful for generating arrays of bit vectors
            with a prescribed structure. For instance, "all bit vectors
            of length 4 that start with 1 and where the last two bits are
            the same" can be expressed as

            >>> PA = from_expression('1**[2]')
            >>> PA
            x0 = 1; x3 = x2
            >>> PA.all()
            array([[1., 0., 0., 0.],
            ...    [1., 1., 0., 0.],
            ...    [1., 0., 1., 1.],
            ...    [1., 1., 1., 1.]])
        """
        G = nx.DiGraph()
        G.add_node('1')
        for i, token in enumerate(cls._ungrouped_tokens(expr)):
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
        return cls(graph=G)

    @classmethod
    def infer(cls, X: np.ndarray):
        X_ = X.reshape(-1, X.shape[-1])
        N, n = X.shape
        assert N > 0, 'At least one example is required!'
        S = ['*'] * n
        # find constants
        col_sum = np.sum(X, axis=0)
        for i, s in enumerate(col_sum):
            if   s == 0: S[i] = '0'
            elif s == N: S[i] = '1'
        # find correspondences
        free = np.asarray([i for i in range(n-1, -1, -1) if S[i]=='*'], dtype=int)
        col_eq = (X_[:,np.newaxis,:]==X_[...,np.newaxis]).sum(0)
        for i, j in combinations(free, r=2):
            if col_eq[i, j] == 0:
                S[i] = f'[!{j}]'
            elif col_eq[i, j] == N:
                S[i] = f'[{j}]'
        return cls.from_expression(''.join(S))

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
    def apply(self, Q: qubo.__class__):
        """Apply this partial assignment either to a ``qubo`` instance. The
        result is a new, smaller instance where the variables fixed by this
        partial assignment are made implicit. E.g., if this assignment
        corresponds to the expression ``*1**[0]`` and we apply it to a ``qubo``
        of size 5, the resulting ``qubo`` will have size 3 (which is ``num_free``
        of this assignment), and its variables correspond to the ``*`` in the
        expression.

        Args:
            expr (str): Bit vector expression.
            x (np.ndarray): Bit vector or array of bit vectors of shape ``(..., m)``
                where ``m`` is the number of ``*`` tokens in ``expr``.

        Returns:
            np.ndarray: Bit vector(s) of shape ``(..., n)`` where ``n`` is the
                number of tokens in ``expr``.
        """
        Q = Q
        assert Q.n == self.size, 'Size of partial assignment does not match QUBO size'
        T = self.to_matrix()
        m = make_upper_triangle(T.T @ Q.m @ T)
        # eliminate constant 1 from matrix (last row and column)
        offset = m[-1, -1]
        return qubo(np.diag(m[:-1, -1]) + m[:-1, :-1]), offset
            

    def expand(self, x: np.ndarray):
        """Fill the free variables of this partial assignments with bits
        provided by ``x``.

        Args:
            x (np.ndarray): Bits to fill the free variables of this partial
                assignment with. Must have shape ``(m?, n)``, where ``n`` is the
                number of free variables of this partial assignment, as given by
                the property ``num_free``.

        Returns:
            np.ndarray: (Array of) bit vector(s) expanded by the partial
                assignment. The shape is ``(m?, s)``, where ``s`` is the size
                of this partial assignment as given by the property ``size``.
        """
        *r, k = x.shape
        assert k == self.num_free, 'Dimension of `x` does not match free variables in expression'
        z = np.empty((*r, self.size))
        ix = 0
        for i, token in enumerate(self._ungrouped_tokens(self.to_expression())):
            if token in '01':
                z[..., i] = float(token)
            elif token.startswith('[!'):
                j = int(token[2:-1])
                z[..., i] = 1-z[..., j]
            elif token.startswith('['):
                j = int(token[1:-1])
                z[..., i] = z[..., j]
            else:
                z[..., i] = x[..., ix]
                ix += 1
        return z

    @__assert_normalized
    def random(self, size, random_state=None):
        rng = get_random_state(random_state)
        rand = rng.random((*to_shape(size), self.num_free))<0.5
        return self.expand(rand)
    
    def all(self):
        """Generate all vectors matching this given partial assignment.

        Returns:
            numpy.ndarray: Array containing all bit vectors that match this
                partial assignment. If ``n`` is the size of this assignment and
                ``m`` the number of free variables, the resulting shape will be
                ``(2**m, n)``.
        """
        m = self.num_free
        all_ = np.arange(1<<m)[:, np.newaxis] & (1<<np.arange(m)) > 0
        return self.expand(all_)
    
    def match(self, x: np.ndarray):
        """Check if a given bit vector or array of bit vectors matches this
        partial assignment, i.e., if it represents a valid realization.

        Args:
            x (np.ndarray): Bit vector or array of bit vectors of shape ``(m?, n)``.

        Returns:
            Boolean ``np.ndarray`` of shape ``(m?,)`` or single Boolean value
            indicating if the bit vectors match this partial assignment.
        """
        out_shape = x.shape[:-1] if x.ndim > 1 else (1,)
        out = np.ones(out_shape, dtype=bool)
        if x.shape[-1] != self.size:
            return out & False if x.ndim > 1 else False
        for i, token in enumerate(self._ungrouped_tokens(self.to_expression())):
            if token == '0':
                out &= x[..., i] == 0
            elif token == '1':
                out &= x[..., i] == 1
            elif token != '*':
                j = int(token.strip('[!]'))
                if '!' in token: # inverse
                    out &= x[..., i] != x[..., j]
                else:
                    out &= x[..., i] == x[..., j]
        return out if x.ndim > 1 else out[0]