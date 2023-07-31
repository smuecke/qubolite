# coding: utf-8
import numpy as np
from qubolite import qubo
from tqdm import tqdm


def calculate_Dplus_and_Dminus(Q: np.array) -> list:
    """
    Calculates a bound for each variable of the possible impact of the variable
    :param Q: np.array that contains the QUBO
    :return: A 2d array, where the first column contains the positive bounds and the second colum the negative bounds
    """
    n = Q.shape[0]
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

def reduceQ(Q: np.array, assignment: tuple, D_list: np.array, indices: list) -> (np.array, np.array, list):
    """
    Given an assignment, updates the Q matrix so that the QUBO stays equivalent but the assigned variable can be removed
    :param Q: np.array containing the QUBO
    :param assignment: tuple, first element is the variabe index, second elment is the assignment, i.e. 0 or 1
    :param D_list: np.array containing bounds as calculated by calculate_Dplus_and_Dminus
    :return: updated Q, updated D_list bounds, updated indices
    """
    #does change input D_list
    #value is assumed to be either 0 or 1
    i = assignment[0]
    value = assignment[1]

    if value == 1:
        Q[indices, indices] += Q[indices, i] + Q[i, indices]

    D_list = D_list_remove(Q, D_list, i)

    #drop node i
    indices.remove(i)

    return Q, D_list, indices


def D_list_remove(Q: np.array, D_list: np.array, i: int) -> np.array:
    """
    Removes influence of variable i in D_list. Used in reduceQ2_5 and reduceQ2_6
    :param Q: np.array containing the QUBO
    :param D_list: np.array containing bounds as calculated by calculate_Dplus_and_Dminus
    :param i: variable whose values are to be removed
    :returns: updated D_list
    """
    d_ij = Q[:, i] + Q[i, :]
    d_plus = d_ij > 0
    d_minus = d_ij < 0
    D_list[1, d_plus] -= d_ij[d_plus]
    D_list[0, d_minus] -= d_ij[d_minus]

    return D_list


def D_list_correct_i(new_i_row_column: np.array, D_list: np.array, i: int, h:int, indices: list) -> np.array:
    """
    Correctes the entry of the i-th variable in D_list in reduceQ2_5 and reduceQ2_6
    :param new_i_row_column: np.array containing the updated row and column of variable i
    :param D_list: np.array containing bounds as calculated by calculate_Dplus_and_Dminus
    :param i: variable whose value is corrected
    :param h: variable that is removed by 2.5 or 2.6
    :param indices: list of variables that have not been assigned yet
    :returns: updated D_list
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


def reduceQ2_5(Q: np.array, assignment: tuple, D_list: np.array, indices: list)-> (np.array, np.array, list):
    """
    Implements updates according to rule 2.5, i.e. assumes x_h = 1 - x_i and updates QUOB accordingly
    :param Q: np.array containing the QUBO
    :param assignment: tuple, first element is the variabe h, second elment is the variable h
    :param D_list: np.array containing bounds as calculated by calculate_Dplus_and_Dminus
    :param indices: list of variables that have not been assigned yet
    :return: updated Q, updated D_list bounds, updated indices
    """
    #x_h = 1 - x_i
    i = assignment[1]
    h = assignment[0]

    c_i = Q[i,i]
    c_h = Q[h,h]
    #D_list update
    #remove h and i from all calculations
    D_list = D_list_remove(Q, D_list, i)
    D_list = D_list_remove(Q, D_list, h)

    new_i_row_column = (Q[:, i] + Q[i, :]) - (Q[:, h] + Q[h, :])
    Q[:i, i] = new_i_row_column[:i]
    Q[i, i+1:] = new_i_row_column[i+1:]
    Q[indices, indices] += (Q[indices, h] + Q[h, indices])
    Q[i,i] = c_i - c_h

    #add new elements d_hj and fix D_i^+ and D_i^-
    D_list = D_list_correct_i(new_i_row_column, D_list, i, h, indices)
    #remove variable x_h from matrix by deleting row and column h
    indices.remove(h)

    return Q, D_list, indices


def reduceQ2_6(Q:np.array, assignment: tuple, D_list: np.array, indices:list)-> (np.array, np.array, list):
    """
    Implements updates according to rule 2.6, i.e. assumes x_h = x_i and updates QUOB accordingly
    :param Q: np.array containing the QUBO
    :param assignment: tuple, first element is the variabe h, second elment is the variable h
    :param D_list: np.array containing bounds as calculated by calculate_Dplus_and_Dminus
    :param indices: list of variables that have not been assigned yet
    :return: updated Q, updated D_list bounds, updated indices
    """
    #x_h = x_i
    i = assignment[1]
    h = assignment[0]

    c_i = Q[i,i]
    c_h = Q[h,h]
    d_hi = Q[h, i]

    #D_list update
    #remove h and i from all calculations
    D_list = D_list_remove(Q, D_list, i)
    D_list = D_list_remove(Q, D_list, h)

    #calculate new x_i values as d_ij + d_hj
    new_i_row_column = (Q[:, i] + Q[i, :]) + (Q[:, h] + Q[h, :])
    Q[:i, i] = new_i_row_column[:i]
    Q[i, i+1:] = new_i_row_column[i+1:]
    Q[i,i] = c_i + c_h + d_hi

    #add new elements d_hj and fix D_i^+ and D_i^-
    D_list = D_list_correct_i(new_i_row_column, D_list, i, h, indices)

    #remove variable x_h from matrix by deleting row and column h
    indices.remove(h)


    return Q, D_list, indices


def assign_1(Qmatrix: np.array, D_list: np.array,
                indices: list, assignments: dict,
                last_assignment: int, c_0: int,
                i: int, c_i: int) ->(np.array, np.array, list, dict, int, int):
    assignments["x_" + str(i)] = 1 #store what assignment is beeing made
    last_assignment = i
    Qmatrix, D_list, indices = reduceQ(Qmatrix, (i, 1), D_list, indices)
    c_0 += c_i
    return Qmatrix, D_list, indices, assignments, last_assignment, c_0


def assign_0(Qmatrix: np.array, D_list: np.array,
                indices: list, assignments: dict,
                last_assignment: int, i: int) ->(np.array, np.array, list, dict, int, int):

    assignments["x_" + str(i)] = 0 #store what assignment is beeing made
    last_assignment = i
    Qmatrix, D_list, indices = reduceQ(Qmatrix, (i, 0), D_list, indices)

    return Qmatrix, D_list, indices, assignments, last_assignment


def apply_rule2_5(Qmatrix: np.array, D_list: np.array,
                indices: list, assignments: dict,
                last_assignment: int, c_0: int,
                i: int, h: int, c_h: int) ->(np.array, np.array, list, dict, int, int):
    assignments["x_" + str(i)] = "1 - x_" + str(h)  #store what assignment is beeing made
    assignments["x_" + str(h)] = "1 - x_" + str(i)  #store what assignment is beeing made
    last_assignment = i+1
    Qmatrix, D_list, indices = reduceQ2_5(Qmatrix, (h, i), D_list, indices)
    c_0 += c_h
    return Qmatrix, D_list, indices, assignments, last_assignment, c_0


def apply_rule2_6(Qmatrix: np.array, D_list: np.array,
                indices: list, assignments: dict,
                last_assignment: int,
                i: int, h: int) ->(np.array, np.array, list, dict, int, int):

    assignments["x_" + str(i)] = "x_" + str(h)  #store what assignment is beeing made
    assignments["x_" + str(h)] = "x_" + str(i)  #store what assignment is beeing made
    last_assignment = i+1
    Qmatrix, D_list, indices = reduceQ2_6(Qmatrix, (h, i), D_list, indices)
    return Qmatrix, D_list, indices, assignments, last_assignment


def reduce_QUBO(Q: qubo) -> (qubo, dict):
    """
    Implements the routine applying rules and reducing the QUBO as much as possible
    :param Q: np.array containing the QUBO
    :return: updated QUBO, a dictionary containing all assignments made, indices of variables that have not been assigned
    """
    #assumes QUBO to be upper triangluar
    #paper assumes maximazation hence we need to flip the sign for minimization
    Qmatrix = -Q.m
    #dict of assignments
    assignments = dict()
    hList = list()
    #needed to recalculate the value of the original objective function
    c_0 = 0

    #calculate D_i^+ and D_i^- for each variable
    D_list = calculate_Dplus_and_Dminus(Qmatrix)

    last_assignment = 0 #index of the most recent variable that has been assigned
    change = True

    #variables that have not been assigned yet
    indices = [i for i in range(Q.n)]

    while (last_assignment != -1):
        last_assignment = -1
        #apply rules
        hList.clear()
        for i in indices.copy():
            #apply rule 1.0 and 2.0
            c_i = Qmatrix[i,i]
            if c_i + D_list[0, i] >= 0: #rule 1.0
                #x_i = 1
                (Qmatrix, D_list, indices,
                 assigments, last_assignment, c_0) = assign_1(Qmatrix, D_list,
                                                                 indices, assignments,
                                                                 last_assignment, c_0,
                                                                 i, c_i)
            elif c_i + D_list[1, i] <= 0: #rule 2.0
                #x_i = 0
                (Qmatrix, D_list, indices,
                 assigments, last_assignment) = assign_0(Qmatrix, D_list,
                                                                 indices, assignments,
                                                                 last_assignment, i)
            else:
                for h in hList:
                    #define variables:
                    d_ih = Qmatrix[h, i]
                    c_i = Qmatrix[i, i]
                    c_h = Qmatrix[h, h]
                    if c_h + D_list[0, h] >= 0: #rule 1.0
                        #x_h = 1
                        (Qmatrix, D_list, indices,
                         assigments, last_assignment, c_0) = assign_1(Qmatrix, D_list,
                                                                 indices, assignments,
                                                                 last_assignment, c_0,
                                                                 h, c_h)
                        #drop node h:
                        hList.remove(h)

                    elif c_h + D_list[1, h] <= 0: #rule 2.0
                        #x_h = 0
                        (Qmatrix, D_list, indices,
                         assigments, last_assignment) = assign_0(Qmatrix, D_list,
                                                                 indices, assignments,
                                                                 last_assignment, h)
                        #drop node h:
                        hList.remove(h)

                    #rule 3.1
                    elif d_ih >= 0 and c_i + c_h - d_ih + D_list[1, i] + D_list[1, h] <= 0:
                        #set x_i = x_h = 0
                        (Qmatrix, D_list, indices,
                         assigments, last_assignment) = assign_0(Qmatrix, D_list,
                                                                 indices, assignments,
                                                                 last_assignment, i)
                        break
                    #rule 3.2 
                    elif d_ih < 0 and -c_i + c_h + d_ih - D_list[0, i] + D_list[1, h] <= 0:
                        #set x_i = 1, x_h = 0
                        (Qmatrix, D_list, indices,
                         assigments, last_assignment, c_0) = assign_1(Qmatrix, D_list,
                                                                 indices, assignments,
                                                                 last_assignment, c_0,
                                                                 i, c_i)
                        break
                    #rule 3.3
                    elif d_ih < 0 and -c_h + c_i + d_ih - D_list[0, h] + D_list[1, i] <= 0:
                        #set x_i = 0, x_h = 1
                        (Qmatrix, D_list, indices,
                         assigments, last_assignment) = assign_0(Qmatrix, D_list,
                                                                 indices, assignments,
                                                                 last_assignment, i)
                        break
                    #rule 3.4
                    elif d_ih >= 0 and -c_i - c_h - d_ih - D_list[0, h] - D_list[0, i] <= 0:
                        #set x_i = 1, x_h = 1
                        (Qmatrix, D_list, indices,
                         assigments, last_assignment, c_0) = assign_1(Qmatrix, D_list,
                                                                 indices, assignments,
                                                                 last_assignment, c_0,
                                                                 i, c_i)
                        break
                    #rule 2.5
                    elif (d_ih < 0 and (c_i - d_ih + D_list[0][i] >= 0
                                        or c_h - d_ih + D_list[0][h] >= 0)
                                  and (c_i + d_ih + D_list[1][i] <= 0
                                       or c_h + d_ih + D_list[1][h]) <= 0):
                        # x_h = 1 - x_i
                        (Qmatrix, D_list, indices,
                         assigments, last_assignment, c_0) = apply_rule2_5(Qmatrix, D_list,
                                                                 indices, assignments,
                                                                 last_assignment, c_0,
                                                                 i, h, c_h)
                        #drop node h:
                        hList.remove(h)

                    #rule 2.6
                    elif (d_ih > 0 and (c_i - d_ih + D_list[1][i] <= 0
                                        or c_h + d_ih + D_list[0][h] >= 0)
                                  and (c_i + d_ih + D_list[0][i] >= 0
                                       or c_h - d_ih + D_list[1][h]) <= 0):
                        # x_h = x_i
                        (Qmatrix, D_list, indices,
                         assigments, last_assignment) = apply_rule2_6(Qmatrix, D_list,
                                                                 indices, assignments,
                                                                 last_assignment, i, h)
                        #drop node h:
                        hList.remove(h)

            if last_assignment != i: #this means x_i could not be assigned
                hList.append(i)

    return qubo(-Qmatrix[np.ix_(indices, indices)]), assignments, -c_0, indices #because sign of Q is flipped at the beginning we have to reverse that her
