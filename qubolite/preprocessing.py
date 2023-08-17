from collections import namedtuple
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
from ._misc       import get_random_state
from .assignment  import partial_assignment

################################################################################
# Dynamic Range Compression                                                    #
################################################################################


def _get_random_index_pair(matrix_order, npr):
    row_indices, column_indices = np.where(np.invert(np.isclose(matrix_order.matrix, 0)))
    try:
        random_index = npr.integers(row_indices.shape[0])
        i, j = row_indices[random_index], column_indices[random_index]
    except ValueError:
        i, j = 0, 0
    return i, j

def _compute_change(matrix_order, npr, heuristic=None, decision='heuristic', **bound_params):
    if decision == 'random':
        i, j = _get_random_index_pair(matrix_order, npr)
        change = _compute_index_change(matrix_order, i, j, heuristic=heuristic, **bound_params)
    elif decision == 'heuristic':
        order_indices = matrix_order.dynamic_range_impact()
        indices = matrix_order.to_matrix_indices(order_indices, matrix_order.matrix.shape[0])
        changes = [_compute_index_change(matrix_order, x[0], x[1],
                                         heuristic=heuristic,
                                         **bound_params) for x in indices]
        drs = [_dynamic_range_change(x[0], x[1], changes[index],
                                     matrix_order) for index, x in enumerate(indices)]
        if np.any(drs):
            index = np.argmax(drs)
            i, j = indices[index]
            change = changes[index]
        else:
            i, j = _get_random_index_pair(matrix_order, npr)
            change = _compute_index_change(matrix_order, i, j,
                                           heuristic=heuristic,
                                           **bound_params)
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

def reduce_dynamic_range(
        Q: qubo,
        iterations=100,
        heuristic='greedy0',
        random_state=None,
        decision='heuristic',
        callback=None,
        **kwargs):
    """Iterative procedure for reducing the dynammic range of a given QUBO, while preserving an
    optimum. For this, at every step we choose a specific QUBO weight and change it according to
    some heuristic.

    Args:
        Q (qubolite.qubo): QUBO
        iterations (int, optional): Number of iterations. Defaults to 100.
        heuristic (str, optional): Used heuristic for computing weight change. Possible heuristics
            are 'greedy0', 'greedy' and 'order'. Defaults to 'greedy0'.
        random_state (optional): A numerical or lexical seed, or a NumPy random generator.
            Defaults to None.
        decision (str, optional): Method for deciding which QUBO weight to change next.
            Possibilities are 'random' and 'heuristic'. Defaults to 'heuristic'.
        callback (optional): Callback function which obtains the following inputs after each step:
            i (int), j (int) , change (float), current matrix order (MatrixOrder), current
            iteration (int). Defaults to None.
        **kwargs (optional): Keyword arguments for determining the upper and lower bound
            computations of the optimal QUBO value.
    Keyword Args:
        change_diff (float): Distance to optimum for avoiding numerical madness. Defaults to 1e-8.
        upper_bound (str): Method for upper bound, possibilities are 'local_descent' and 'sample'.
            Defaults to 'local_descent'.
        lower_bound (str): Method for lower bound, possibilities are 'roof_dual' and 'min_sum'.
            Defaults to 'roof_dual'.
        upper_bound_kwargs (dict): Additional keyword arguments for upper bound method.
        lower_bound_kwargs (dict): Additional keyword arguments for lower bound method.
    Returns:
        qubolite.qubo: Compressed QUBO with reduced dynamic range.
    """
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
            i, j, change = _compute_change(matrix_order, heuristic=heuristic, npr=npr,
                                           decision=decision, **kwargs)
            stop_update = matrix_order.update_entry(i, j, change)
            if callback is not None:
                callback(i, j, change, matrix_order, it)
        else:
            break
    return qubo(matrix_order.matrix)


################################################################################
# QPRO+ Algorithm                                                              #
################################################################################

def _calculate_Dplus_and_Dminus(Q: np.ndarray):
    """Calculates a bound for each variable of the possible impact of the variable.

    Args:
        Q (np.ndarray): Array that contains the QUBO

    Returns:
        np.ndarray: A 2d array, where the first column contains the positive bounds and the second
            column the negative bounds
    """
    Q_plus = np.multiply(Q, Q > 0)
    np.fill_diagonal(Q_plus, 0)
    row_sums_plus = np.sum(Q_plus, axis = 0)
    coulumn_sums_plus = np.sum(Q_plus, axis = 1)
    d_plus = row_sums_plus + coulumn_sums_plus
    Q_minus = np.multiply(Q, Q < 0)
    np.fill_diagonal(Q_minus, 0)
    row_sums_minus = np.sum(Q_minus, axis = 0)
    coulumn_sums_minus = np.sum(Q_minus, axis = 1)
    d_minus = row_sums_minus + coulumn_sums_minus
    return np.vstack((d_minus, d_plus))

def _reduceQ(Q: np.ndarray, assignment: tuple, D_list: np.ndarray, indices: list):
    """Given an assignment, updates the Q matrix so that the QUBO stays
    equivalent but the assigned variable can be removed.

    Args:
        Q (np.ndarray): containing the QUBO
        assignment (tuple): first element is the variabe index, second elment is the assignment,
            i.e. 0 or 1
        D_list (np.ndarray): containing bounds as calculated by calculate_Dplus_and_Dminus

    Returns:
        np.ndarray: updated Q
        np.ndarray: updated D_list
        list: updated indices
    """
    #does change input D_list
    #value is assumed to be either 0 or 1
    i = assignment[0]
    value = assignment[1]
    if value == 1:
        Q[indices, indices] += Q[indices, i] + Q[i, indices]
    D_list = _D_list_remove(Q, D_list, i)
    indices.remove(i) # drop node i
    return Q, D_list, indices

def _D_list_remove(Q: np.ndarray, D_list: np.ndarray, i: int):
    """Removes influence of variable i in D_list. Used in reduceQ2_5 and reduceQ2_6

    Args:
        Q (np.ndarray): containing the QUBO
        D_list (np.ndarray): containing bounds as calculated by calculate_Dplus_and_Dminus
        i (int): variable whose values are to be removed

    Returns:
        np.ndarray: updated D_list
    """
    d_ij = Q[:, i] + Q[i, :]
    d_plus = d_ij > 0
    d_minus = d_ij < 0
    D_list[1, d_plus] -= d_ij[d_plus]
    D_list[0, d_minus] -= d_ij[d_minus]
    return D_list

def _D_list_correct_i(
        new_i_row_column: np.ndarray,
        D_list: np.ndarray,
        i: int,
        h:int,
        indices: list):
    """Corrects the entry of the i-th variable in D_list in reduceQ2_5 and reduceQ2_6

    Args:
        new_i_row_column (np.ndarray): containing the updated row and column of variable i
        D_list (np.ndarray): containing bounds as calculated by calculate_Dplus_and_Dminus
        i (int): variable whose value is corrected
        h (int): variable that is removed by 2.5 or 2.6
        indices (list): of variables that have not been assigned yet

    Returns:
        np.ndarray: updated D_list
    """
    #add new elements d_hj
    new_i_row_column[i] = 0
    new_i_row_column[h] = 0
    positive = new_i_row_column > 0
    negative = new_i_row_column < 0
    D_list[1, positive] += new_i_row_column[positive]
    D_list[0, negative] += new_i_row_column[negative]
    #fix D_i^+ and D_i^-
    D_list[1, i] = np.sum(new_i_row_column[indices][positive[indices]])
    D_list[0, i] = np.sum(new_i_row_column[indices][negative[indices]])
    return D_list

def _reduceQ2_5(Q: np.ndarray, assignment: tuple, D_list: np.ndarray, indices: list):
    """
    Implements updates according to rule 2.5, i.e. assumes x_h = 1 - x_i and updates QUBO
    accordingly

    Args:
        Q (np.ndarray): containing the QUBO
        assignment (tuple): first element is the variabe h, second elment is the variable h
        D_list (np.ndarray): containing bounds as calculated by calculate_Dplus_and_Dminus
        indices (list): of variables that have not been assigned yet

    Returns:
        np.ndarray: updated Q
        np.ndarray: updated D_list
        list: updated indices
    """
    #x_h = 1 - x_i
    i = assignment[1]
    h = assignment[0]
    c_i = Q[i,i]
    c_h = Q[h,h]
    #D_list update
    #remove h and i from all calculations
    D_list = _D_list_remove(Q, D_list, i)
    D_list = _D_list_remove(Q, D_list, h)
    new_i_row_column = (Q[:, i] + Q[i, :]) - (Q[:, h] + Q[h, :])
    Q[:i, i] = new_i_row_column[:i]
    Q[i, i+1:] = new_i_row_column[i+1:]
    Q[indices, indices] += (Q[indices, h] + Q[h, indices])
    Q[i,i] = c_i - c_h
    #add new elements d_hj and fix D_i^+ and D_i^-
    D_list = _D_list_correct_i(new_i_row_column, D_list, i, h, indices)
    #remove variable x_h from matrix by deleting row and column h
    indices.remove(h)
    return Q, D_list, indices

def _reduceQ2_6(
        Q: np.ndarray,
        assignment: tuple,
        D_list: np.ndarray,
        indices:list):
    """
    Implements updates according to rule 2.6, i.e. assumes x_h = x_i and updates QUBO accordingly

    Args;
        Q (np.ndarray): containing the QUBO
        assignment (tuple): first element is the variabe h, second elment is the variable h
        D_list (np.ndarray): containing bounds as calculated by calculate_Dplus_and_Dminus
        indices (list): variables that have not been assigned yet

    Returns:
        np.ndarray: updated Q
        np.ndarray: updated D_list
        np.ndarray: updated indices
    """
    #x_h = x_i
    i = assignment[1]
    h = assignment[0]
    c_i = Q[i,i]
    c_h = Q[h,h]
    d_hi = Q[h, i]
    #D_list update
    #remove h and i from all calculations
    D_list = _D_list_remove(Q, D_list, i)
    D_list = _D_list_remove(Q, D_list, h)
    #calculate new x_i values as d_ij + d_hj
    new_i_row_column = (Q[:, i] + Q[i, :]) + (Q[:, h] + Q[h, :])
    Q[:i, i] = new_i_row_column[:i]
    Q[i, i+1:] = new_i_row_column[i+1:]
    Q[i,i] = c_i + c_h + d_hi
    #add new elements d_hj and fix D_i^+ and D_i^-
    D_list = _D_list_correct_i(new_i_row_column, D_list, i, h, indices)
    #remove variable x_h from matrix by deleting row and column h
    indices.remove(h)
    return Q, D_list, indices

def _assign_1(
        Qmatrix: np.ndarray,
        D_list: np.ndarray,
        indices: list,
        assignments: dict,
        last_assignment: int,
        c_0: int,
        i: int,
        c_i: int):
    assignments["x_" + str(i)] = 1 #store what assignment is being made
    last_assignment = i
    Qmatrix, D_list, indices = _reduceQ(Qmatrix, (i, 1), D_list, indices)
    c_0 += c_i
    return Qmatrix, D_list, indices, assignments, last_assignment, c_0

def _assign_0(
        Qmatrix: np.ndarray,
        D_list: np.ndarray,
        indices: list,
        assignments: dict,
        last_assignment: int, i: int):
    assignments["x_" + str(i)] = 0 #store what assignment is beeing made
    last_assignment = i
    Qmatrix, D_list, indices = _reduceQ(Qmatrix, (i, 0), D_list, indices)
    return Qmatrix, D_list, indices, assignments, last_assignment

def _apply_rule2_5(
        Qmatrix: np.ndarray,
        D_list: np.ndarray,
        indices: list,
        assignments: dict,
        last_assignment: int,
        c_0: int,
        i: int,
        h: int,
        c_h: int):
    assignments["x_" + str(i)] = f'[!{h}]' #store what assignment is being made
    assignments["x_" + str(h)] = f'[!{i}]' #store what assignment is being made
    last_assignment = i+1
    Qmatrix, D_list, indices = _reduceQ2_5(Qmatrix, (h, i), D_list, indices)
    c_0 += c_h
    return Qmatrix, D_list, indices, assignments, last_assignment, c_0

def _apply_rule2_6(
        Qmatrix: np.ndarray,
        D_list: np.ndarray,
        indices: list,
        assignments: dict,
        last_assignment: int,
        i: int, h: int):
    assignments["x_" + str(i)] = f'[{h}]'  #store what assignment is being made
    assignments["x_" + str(h)] = f'[{i}]'  #store what assignment is being made
    last_assignment = i+1
    Qmatrix, D_list, indices = _reduceQ2_6(Qmatrix, (h, i), D_list, indices)
    return Qmatrix, D_list, indices, assignments, last_assignment

def qpro_plus(Q: qubo):
    """Implements the routine applying rules described in
    `Glover et al., 2018<https://www.sciencedirect.com/science/article/pii/S0377221717307567>`__
    for reducing the QUBO size by applying logical implications.

    Args:
        Q (qubo): QUBO instance to be reduced.

    Returns:
        Instance of :class:`assignment.partial_assignment` representing the
        reduction. See example.

    Example:
        >>> import qubolite as ql
        >>> Q = ql.qubo.random(32, density=0.2, random_state='example')
        >>> PA = ql.preprocessing.qpro_plus(Q)
        >>> print(f'{PA.num_fixed} variables were eliminated!')
        9 variables were eliminated!
        >>> Q_reduced, offset = PA.apply(Q)
        >>> Q_reduced.n
        23
        >>> x = ql.bitvec.from_string('10011101011011110001011')
        >>> Q_reduced(x)+offset
        -0.5215481745331401
        >>> Q(PA.expand(x))
        -0.5215481745331385

    """
    #assumes QUBO to be upper triangluar
    #paper assumes maximazation hence we need to flip the sign for minimization
    m = -Q.m
    assignments = dict() # dict of assignments
    hList = list()
    c_0 = 0 # offset to restore original energy function
    #calculate D_i^+ and D_i^- for each variable
    D_list = _calculate_Dplus_and_Dminus(m)
    last_assignment = 0 #index of the most recent variable that has been assigned
    #variables that have not been assigned yet
    indices = list(range(Q.n))

    while (last_assignment != -1):
        last_assignment = -1
        #apply rules
        hList.clear()
        for i in indices.copy():
            #apply rule 1.0 and 2.0
            c_i = m[i,i]
            if c_i + D_list[0, i] >= 0: #rule 1.0
                #x_i = 1
                m, D_list, indices, _, last_assignment, c_0 = _assign_1(
                    m, D_list, indices, assignments, last_assignment, c_0, i, c_i)
            elif c_i + D_list[1, i] <= 0: #rule 2.0
                #x_i = 0
                (m, D_list, indices, _, last_assignment) = _assign_0(
                    m, D_list, indices, assignments, last_assignment, i)
            else:
                for h in hList:
                    #define variables:
                    d_ih = m[h, i]
                    c_i = m[i, i]
                    c_h = m[h, h]
                    if c_h + D_list[0, h] >= 0: #rule 1.0
                        #x_h = 1
                        m, D_list, indices, _, last_assignment, c_0 = _assign_1(
                            m, D_list, indices, assignments, last_assignment, c_0, h, c_h)
                        # drop node h
                        hList.remove(h)

                    elif c_h + D_list[1, h] <= 0: #rule 2.0
                        #x_h = 0
                        m, D_list, indices, _, last_assignment = _assign_0(
                            m, D_list, indices, assignments, last_assignment, h)
                        # drop node h
                        hList.remove(h)

                    #rule 3.1
                    elif d_ih >= 0 and c_i + c_h - d_ih + D_list[1, i] + D_list[1, h] <= 0:
                        #set x_i = x_h = 0
                        m, D_list, indices, _, last_assignment = _assign_0(
                            m, D_list, indices, assignments, last_assignment, i)
                        break
                    #rule 3.2 
                    elif d_ih < 0 and -c_i + c_h + d_ih - D_list[0, i] + D_list[1, h] <= 0:
                        #set x_i = 1, x_h = 0
                        m, D_list, indices, _, last_assignment, c_0 = _assign_1(
                            m, D_list, indices, assignments, last_assignment, c_0, i, c_i)
                        break
                    #rule 3.3
                    elif d_ih < 0 and -c_h + c_i + d_ih - D_list[0, h] + D_list[1, i] <= 0:
                        #set x_i = 0, x_h = 1
                        m, D_list, indices, _, last_assignment = _assign_0(
                            m, D_list, indices, assignments, last_assignment, i)
                        break
                    #rule 3.4
                    elif d_ih >= 0 and -c_i - c_h - d_ih - D_list[0, h] - D_list[0, i] <= 0:
                        #set x_i = 1, x_h = 1
                        m, D_list, indices, _, last_assignment, c_0 = _assign_1(
                            m, D_list, indices, assignments, last_assignment, c_0, i, c_i)
                        break
                    #rule 2.5
                    elif (d_ih < 0 and (c_i - d_ih + D_list[0][i] >= 0
                                        or c_h - d_ih + D_list[0][h] >= 0)
                                  and (c_i + d_ih + D_list[1][i] <= 0
                                       or c_h + d_ih + D_list[1][h]) <= 0):
                        # x_h = 1 - x_i
                        m, D_list, indices, _, last_assignment, c_0 = _apply_rule2_5(
                            m, D_list, indices, assignments, last_assignment, c_0, i, h, c_h)
                        # drop node h
                        hList.remove(h)

                    #rule 2.6
                    elif (d_ih > 0 and (c_i-d_ih+D_list[1][i] <= 0
                                        or c_h+d_ih+D_list[0][h] >= 0)
                                   and (c_i+d_ih+D_list[0][i] >= 0
                                        or c_h-d_ih+D_list[1][h]) <= 0):
                        # x_h = x_i
                        m, D_list, indices, _, last_assignment = _apply_rule2_6(
                            m, D_list, indices, assignments, last_assignment, i, h)
                        # drop node h
                        hList.remove(h)

            if last_assignment != i: # this means x_i could not be assigned
                hList.append(i)

    assignment_pattern = ''.join([str(assignments.get(f'x_{i}', '*')) for i in range(Q.n)])
    return partial_assignment.from_expression(assignment_pattern)
    # reduced qubo: qubo(-m[np.ix_(indices, indices)])
    # offset: -c_0