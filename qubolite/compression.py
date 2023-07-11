from functools import partial

import numpy as np
import portion as P

from . import qubo
from .bounds import (
    lb_roof_dual,
    lb_negative_parameters,
    ub_local_descent,
    ub_sample)
from ._heuristics import MatrixOrder, HEURISTICS
from ._misc import get_random_state


def _compute_final_change(matrix_order, heuristic=None, decision='heuristic',
                          npr=None, **kwargs):
    if decision == 'random':
        row_indices, column_indices = np.where(np.invert(np.isclose(matrix_order.matrix, 0)))
        try:
            random_index = npr.integers(row_indices.shape[0])
            i, j = row_indices[random_index], column_indices[random_index]
        except ValueError:
            i, j = 0, 0
        change = _compute_index_change(matrix_order, i, j,
                                       heuristic=heuristic,
                                                  ** kwargs)
    elif decision == 'heuristic':
        order_indices = matrix_order.dynamic_range_impact()
        indices = matrix_order.to_matrix_indices(order_indices, matrix_order.matrix.shape[0])
        changes = [_compute_index_change(matrix_order, x[0], x[1],
                                         heuristic=heuristic,
                                         **kwargs) for x in indices]
        drs = [_dynamic_range_change(x[0], x[1], changes[index],
                                     matrix_order) for index, x in enumerate(indices)]
        if np.any(drs):
            index = np.argmax(drs)
            i, j = indices[index]
            change = changes[index]
        else:
            row_indices, column_indices = np.where(np.invert(np.isclose(matrix_order.matrix, 0)))
            try:
                random_index = npr.integers(row_indices.shape[0])
                i, j = row_indices[random_index], column_indices[random_index]
            except ValueError:
                i, j = 0, 0
            change = _compute_index_change(matrix_order, i, j,
                                           heuristic=heuristic,
                                           **kwargs)
    else:
        raise NotImplementedError
    return i, j, change


def _compute_pre_opt_bounds(Q, i, j, **kwargs):
    lower_bound = {
        'roof_dual': lb_roof_dual,
        'min_sum': lb_negative_parameters
    }[kwargs.get('lower_bound', 'roof_dual')]
    upper_bound = {
        'local_descent': ub_local_descent,
        'sample': ub_sample
    }[kwargs.get('upper_bound', 'local_descent')]
    lower_bound = partial(lower_bound, **kwargs.get('lower_bound_kwargs', {}))
    upper_bound = partial(upper_bound, **kwargs.get('upper_bound_kwargs', {}))
    change_diff = kwargs.get('change_diff', 1e-08)
    Q = qubo(Q)
    if i != j:
        # Define sub-qubos
        Q_00, c_00, _ = Q.clamp({i: 0, j: 0})
        Q_01, c_01, _ = Q.clamp({i: 0, j: 1})
        Q_10, c_10, _ = Q.clamp({i: 1, j: 0})
        Q_11, c_11, _ = Q.clamp({i: 1, j: 1})
        # compute bounds
        upper_00 = upper_bound(Q_00) + c_00
        upper_01 = upper_bound(Q_01) + c_01
        upper_10 = upper_bound(Q_10) + c_10
        lower_11 = lower_bound(Q_11) + c_11
        upper_or = min(upper_00, upper_01, upper_10)

        lower_00 = lower_bound(Q_00) + c_00
        lower_01 = lower_bound(Q_01) + c_01
        lower_10 = lower_bound(Q_10) + c_10
        upper_11 = upper_bound(Q_11) + c_11
        lower_or = min(lower_00, lower_01, lower_10)

        suboptimal = lower_11 > min(upper_00, upper_01, upper_10)
        optimal = upper_11 < min(lower_00, lower_01, lower_10)
        upper_bound = float('inf') if suboptimal else lower_or - upper_11 - change_diff
        lower_bound = -float('inf') if optimal else upper_or - lower_11 + change_diff
    else:
        # Define sub-qubos
        Q_0, c_0, _ = Q.clamp({i: 0})
        Q_1, c_1, _ = Q.clamp({i: 1})
        # Compute bounds
        upper_0 = upper_bound(Q_0) + c_0
        lower_1 = lower_bound(Q_1) + c_1

        lower_0 = lower_bound(Q_0) + c_0
        upper_1 = upper_bound(Q_1) + c_1
        suboptimal = lower_1 > upper_0
        optimal = upper_1 < lower_0
        upper_bound = float("inf") if suboptimal else lower_0 - upper_1 - change_diff
        lower_bound = -float("inf") if optimal else upper_0 - lower_1 + change_diff
    return lower_bound, upper_bound


def _compute_pre_opt_bounds_all(Q, **kwargs):
    indices = np.triu_indices(Q.shape[0])
    bounds = np.zeros((len(indices[0]), 2))
    for index, index_pair in enumerate(zip(indices[0], indices[1])):
        i, j = index_pair[0], index_pair[1]
        res = _compute_pre_opt_bounds(Q, i=i, j=j, **kwargs)
        if isinstance(res[0], qubo):
            print(f'Optimal configuration is found, clamped QUBO should be returned!')
            # return res
        else:
            bounds[index, 0] = res[0]
            bounds[index, 1] = res[1]
    return bounds


def _dynamic_range_change(i, j, change, matrix_order):
    old_dynamic_range = matrix_order.dynamic_range
    matrix = matrix_order.update_entry(i, j, change, True)
    new_dynamic_range = qubo(matrix).dynamic_range()
    dynamic_range_diff = old_dynamic_range - new_dynamic_range
    return dynamic_range_diff


def _check_to_next_increase(matrix_order, change, i, j):
    current_entry = matrix_order.matrix[i, j]
    new_entry = current_entry + change
    lower_index = np.searchsorted(matrix_order.unique, new_entry, side='right')
    lower_entry = matrix_order.unique[lower_index - 1]
    min_dis = matrix_order.min_distance
    lower_interval = P.open(lower_entry - min_dis, lower_entry + min_dis)
    try:
        upper_entry = matrix_order.unique[lower_index]
        upper_interval = P.open(upper_entry - min_dis, upper_entry + min_dis)
        forbidden_interval = lower_interval | upper_interval
    except IndexError:
        forbidden_interval = lower_interval
    possible_interval = P.openclosed(-P.inf, new_entry)
    difference = possible_interval.difference(forbidden_interval)
    difference = difference | P.singleton(lower_entry)
    return difference.upper - current_entry


def _check_to_next_decrease(matrix_order, change, i, j):
    current_entry = matrix_order.matrix[i, j]
    new_entry = current_entry + change
    upper_index = np.searchsorted(matrix_order.unique, new_entry, side='left')
    upper_entry = matrix_order.unique[upper_index]
    min_dis = matrix_order.min_distance
    upper_interval = P.open(upper_entry - min_dis, upper_entry + min_dis)
    try:
        lower_entry = matrix_order.unique[upper_index - 1]
        lower_interval = P.open(lower_entry - min_dis, lower_entry + min_dis)
        forbidden_interval = lower_interval | upper_interval
    except IndexError:
        forbidden_interval = upper_interval
    possible_interval = P.openclosed(new_entry, P.inf)
    difference = possible_interval.difference(forbidden_interval)
    difference = difference | P.singleton(upper_entry)
    return difference.lower - current_entry


def _compute_index_change(matrix_order, i, j, heuristic=None, **kwargs):
    # Decide whether to increase or decrease
    increase = heuristic.decide_increase(matrix_order, i, j)
    # Bounds on changes based on reducing the dynamic range
    dyn_range_change = heuristic.compute_change(matrix_order, i, j, increase)
    # Bounds on changes based on preserving the optimum
    if increase:
        _, pre_opt_change = _compute_pre_opt_bounds(matrix_order.matrix, i, j, **kwargs)
    else:
        pre_opt_change, _ = _compute_pre_opt_bounds(matrix_order.matrix, i, j, **kwargs)
    set_to_zero = heuristic.set_to_zero()
    if increase:
        change = min(pre_opt_change, dyn_range_change)
        if change < 0 or np.isclose(change, 0):
            change = 0
        elif 0 > matrix_order.matrix[i, j] > - change and set_to_zero:
            change = - matrix_order.matrix[i, j]
        else:
            change = _check_to_next_increase(matrix_order, change, i, j)
    else:
        change = max(pre_opt_change, dyn_range_change)
        if change > 0 or np.isclose(change, 0):
            change = 0
        elif 0 < matrix_order.matrix[i, j] < - change and set_to_zero:
            change = - matrix_order.matrix[i, j]
        else:
            change = _check_to_next_decrease(matrix_order, change, i, j)
    return change


def compress_parameters(Q: qubo,
                        iterations=100,
                        callback=None,
                        heuristic='greedy0',
                        random_state=None,
                        decision='heuristic',
                        **kwargs):
    try:
        heuristic = HEURISTICS[heuristic]
    except KeyError:
        raise ValueError(f'Unknown heuristic "{heuristic}", available are "greedy0", "greedy" and "order"')
    npr = get_random_state(random_state)
    Q_copy = Q.copy()
    matrix_order = MatrixOrder(Q_copy.m)
    stop_update = False
    matrix_order.matrix = np.round(matrix_order.matrix, decimals=8)
    for it in range(iterations):
        if not stop_update:
            i, j, change = _compute_final_change(matrix_order, heuristic=heuristic, npr=npr,
                                                 decision=decision, **kwargs)
            stop_update = matrix_order.update_entry(i, j, change)
            if callback is not None:
                callback(i, j, change, matrix_order, it)
        else:
            break
    return qubo(matrix_order.matrix)
