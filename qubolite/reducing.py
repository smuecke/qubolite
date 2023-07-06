import numpy as np
from numpy import newaxis as na
from .dr_heuristics import ReduceHeuristic
from . import qubo
from .bounds import lb_roof_dual, lb_negative_parameters, ub_local_search, ub_sample
from .solving import brute_force


def decide_index(unique, heuristic=None, lower_bound_method="roof_dual",
                 upper_bound_method="gradient", npr=None, set_to_zero=True,
                 change_tol=1e-08, change_diff=1e-08):
    if npr is None:
        npr = np.random.RandomState()
    if heuristic is None:
        row_indices, column_indices = np.where(np.invert(np.isclose(unique.matrix, 0)))
        try:
            random_index = npr.randint(row_indices.shape[0])
            i, j = row_indices[random_index], column_indices[random_index]
        except ValueError:
            i, j = 0, 0
    elif isinstance(heuristic, ReduceHeuristic):
        unique_indices = unique.dynamic_range_impact()
        indices = unique.to_matrix_indices(unique_indices, unique.qubo.n)
        drs = [dynamic_range_change(x[0], x[1],
                                    compute_final_change(unique, x[0], x[1],
                                                         lower_bound_method=lower_bound_method,
                                                         upper_bound_method=upper_bound_method,
                                                         heuristic=heuristic,
                                                         change_tol=change_tol,
                                                         change_diff=change_diff,
                                                         set_to_zero=set_to_zero),
                                    unique) for x in indices]
        if np.any(drs):
            index = np.argmax(drs)
            i, j = indices[index]
        else:
            row_indices, column_indices = np.where(np.invert(np.isclose(unique.matrix, 0)))
            try:
                random_index = npr.randint(row_indices.shape[0])
                i, j = row_indices[random_index], column_indices[random_index]
            except ValueError:
                i, j = 0, 0
    else:
        raise NotImplementedError
    return i, j


def compute_opt_pre_bound(Q, i, j, increase=True, lower_bound_method="roof_dual",
                          upper_bound_method="local_search", change_diff=1e-08):
    if lower_bound_method == "roof_dual":
        lower_bound = lb_roof_dual
    elif lower_bound_method == "min_sum":
        lower_bound = lb_negative_parameters
    else:
        raise NotImplementedError
    if upper_bound_method == "local_search":
        upper_bound = ub_local_search
    elif upper_bound_method == "sample":
        upper_bound = ub_sample
    else:
        raise NotImplementedError
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


def compute_final_change(matrix_order, i, j, lower_bound_method="roof_dual",
                         upper_bound_method="local_search",
                         heuristic=None, change_tol=1e-08, change_diff=1e-08, set_to_zero=True):
    # Decide whether to increase or decrease
    increase = heuristic.decide_increase(matrix_order, i, j)
    # Bounds on changes based on reducing the dynamic range
    dyn_range_change = heuristic.compute_change(matrix_order, i, j, increase)
    # Bounds on changes based on maintaining the optimum
    main_opt_change = compute_opt_pre_bound(matrix_order.qubo, i, j, increase,
                                            lower_bound_method=lower_bound_method,
                                            upper_bound_method=upper_bound_method,
                                            change_diff=change_diff)
    if increase:
        change = min(main_opt_change, dyn_range_change)
        if change < 0 or np.isclose(change, 0, atol=change_tol):
            change = 0
        elif 0 > matrix_order.matrix[i, j] > - change and set_to_zero:
            change = - matrix_order.matrix[i, j]
    else:
        change = max(main_opt_change, dyn_range_change)
        if change > 0 or np.isclose(change, 0, atol=change_tol):
            change = 0
        elif 0 < matrix_order.matrix[i, j] < - change and set_to_zero:
            change = - matrix_order.matrix[i, j]
    return change


def reduce_dr(Q: qubo, iterations=100):
    return NotImplemented
