import numpy as np

from . import qubo
from .bounds import lb_roof_dual, lb_negative_parameters, ub_local_search, ub_sample
from .dr_heuristics import ReduceHeuristic, MatrixOrder, Greedy


def decide_index(matrix_order, heuristic=None, bound_dict=None, npr=None, set_to_zero=True,
                 change_tol=1e-08):
    if npr is None:
        npr = np.random.RandomState()
    if heuristic is None:
        row_indices, column_indices = np.where(np.invert(np.isclose(matrix_order.matrix, 0)))
        try:
            random_index = npr.randint(row_indices.shape[0])
            i, j = row_indices[random_index], column_indices[random_index]
        except ValueError:
            i, j = 0, 0
    elif isinstance(heuristic, ReduceHeuristic):
        order_indices = matrix_order.dynamic_range_impact()
        indices = matrix_order.to_matrix_indices(order_indices, matrix_order.matrix.shape[0])
        drs = [dynamic_range_change(x[0], x[1],
                                    compute_final_change(matrix_order, x[0], x[1],
                                                         bound_dict=bound_dict,
                                                         heuristic=heuristic,
                                                         change_tol=change_tol,
                                                         set_to_zero=set_to_zero),
                                    matrix_order) for x in indices]
        if np.any(drs):
            index = np.argmax(drs)
            i, j = indices[index]
        else:
            row_indices, column_indices = np.where(np.invert(np.isclose(matrix_order.matrix, 0)))
            try:
                random_index = npr.randint(row_indices.shape[0])
                i, j = row_indices[random_index], column_indices[random_index]
            except ValueError:
                i, j = 0, 0
    else:
        raise NotImplementedError
    return i, j


def compute_pre_opt_bound(Q, i, j, increase=True, bound_dict=None):
    if bound_dict is None:
        bound_dict = {'upper_bound': 'local_search', 'lower_bound': 'roof_dual',
                      'change_diff': 1e-08, 'upper_bound_args': None, 'lower_bound_args': None}
    if bound_dict['lower_bound'] == "roof_dual":
        lower_bound = lb_roof_dual
    elif bound_dict['lower_bound'] == "min_sum":
        lower_bound = lb_negative_parameters
    else:
        raise NotImplementedError
    if bound_dict['upper_bound'] == "local_search":
        upper_bound = ub_local_search
    elif bound_dict['upper_bound'] == "sample":
        upper_bound = ub_sample
    else:
        raise NotImplementedError
    change_diff = bound_dict['change_diff']
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
        if increase:
            if suboptimal:
                bound = float("inf")
            else:
                bound = lower_or - upper_11 - change_diff
        else:
            if optimal:
                bound = - float("inf")
            else:
                bound = upper_or - lower_11 + change_diff
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
        if increase:
            if suboptimal:
                bound = float("inf")
            else:
                bound = lower_0 - upper_1 - change_diff
        else:
            if optimal:
                bound = -float("inf")
            else:
                bound = upper_0 - lower_1 + change_diff
    return bound


def dynamic_range_change(i, j, change, matrix_order):
    # TODO: dynamic range change can be improved
    old_dynamic_range = matrix_order.dynamic_range
    matrix = matrix_order.update_entry(i, j, change, True)
    new_dynamic_range = qubo(matrix).dynamic_range()
    dynamic_range_diff = old_dynamic_range - new_dynamic_range
    return dynamic_range_diff


def compute_final_change(matrix_order, i, j, bound_dict=None, heuristic=None, change_tol=1e-08,
                         set_to_zero=True):
    # Decide whether to increase or decrease
    increase = heuristic.decide_increase(matrix_order, i, j)
    # Bounds on changes based on reducing the dynamic range
    dyn_range_change = heuristic.compute_change(matrix_order, i, j, increase)
    # Bounds on changes based on preserving the optimum
    pre_opt_change = compute_pre_opt_bound(matrix_order.matrix, i, j, increase, bound_dict=bound_dict)
    if increase:
        change = min(pre_opt_change, dyn_range_change)
        if change < 0 or np.isclose(change, 0, atol=change_tol):
            change = 0
        elif 0 > matrix_order.matrix[i, j] > - change and set_to_zero:
            change = - matrix_order.matrix[i, j]
    else:
        change = max(pre_opt_change, dyn_range_change)
        if change > 0 or np.isclose(change, 0, atol=change_tol):
            change = 0
        elif 0 < matrix_order.matrix[i, j] < - change and set_to_zero:
            change = - matrix_order.matrix[i, j]
    return change


def reduce_dr(Q: qubo, iterations=100, callback=None, set_to_zero=True, heuristic=None, npr=None,
              bound_dict=None, change_tol=1e-08):
    if heuristic is None:
        heuristic = Greedy()
    Q_copy = Q.copy()
    matrix_order = MatrixOrder(Q_copy.m)
    stop_update = False
    for it in range(iterations):
        if not stop_update:
            i, j = decide_index(matrix_order, heuristic=heuristic, bound_dict=bound_dict, npr=npr,
                                set_to_zero=set_to_zero, change_tol=change_tol)
            change = compute_final_change(matrix_order, i, j, bound_dict=None, heuristic=heuristic,
                                          change_tol=change_tol, set_to_zero=set_to_zero)
            stop_update = matrix_order.update_entry(i, j, change)
            if callback is not None:
                callback(i, j, change, matrix_order, it)
        else:
            break
    return qubo(matrix_order.matrix)
