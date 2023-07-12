import numpy as np


class Greedy:
    @staticmethod
    def compute_change(matrix_order, i, j, increase=True):
        sorted_index = matrix_order.get_sorted_index(i, j)
        S = matrix_order.unique_elements
        sorted = matrix_order.unique
        second_min_distance = matrix_order.second_min_distance
        if sorted_index == 0:
            if matrix_order.min_index_lower == sorted_index and not np.isclose(second_min_distance,
                                                                               matrix_order.min_distance):
                change = sorted[S - 1] - 2 * sorted[0] + sorted[1] + matrix_order.extra_summand
            else:
                change = sorted[S - 1] - 2 * sorted[0] + sorted[1]
        elif sorted_index == S - 1:
            if matrix_order.min_index_upper == sorted_index and not np.isclose(second_min_distance,
                                                                               matrix_order.min_distance):
                change = sorted[0] - 2 * sorted[S - 1] + sorted[S - 2] - matrix_order.extra_summand
            else:
                change = sorted[0] - 2 * sorted[S - 1] + sorted[S - 2]
        else:
            if increase:
                if matrix_order.min_index_lower == sorted_index and not np.isclose(second_min_distance,
                                                                                   matrix_order.min_distance):
                    change = sorted[S - 1] - sorted[sorted_index] + matrix_order.extra_summand
                else:
                    change = sorted[S - 1] - sorted[sorted_index] - matrix_order.min_distance
            else:
                if matrix_order.min_index_upper == sorted_index and not np.isclose(second_min_distance,
                                                                                   matrix_order.min_distance):
                    change = sorted[1] - sorted[sorted_index] - matrix_order.extra_summand
                else:
                    change = sorted[1] - sorted[sorted_index] + matrix_order.min_distance
        return change

    @staticmethod
    def decide_increase(matrix_order, i, j):
        sorted_index = matrix_order.get_sorted_index(i, j)
        if matrix_order.unique[sorted_index] < 0:
            increase = True
        else:
            increase = False
        return increase

    @staticmethod
    def set_to_zero():
        return False


class GreedyZero:
    @staticmethod
    def compute_change(matrix_order, i, j, increase=True):
        return Greedy.compute_change(matrix_order, i, j, increase=increase)

    @staticmethod
    def decide_increase(matrix_order, i, j):
        return Greedy.decide_increase(matrix_order, i, j)

    @staticmethod
    def set_to_zero():
        return True


class MaintainOrder:
    @staticmethod
    def compute_change(matrix_order, i, j, increase=True):
        sorted_index = matrix_order.get_sorted_index(i, j)
        S = matrix_order.unique_elements
        sorted = matrix_order.unique
        if sorted_index == 0:
            if increase:
                change = sorted[1] - sorted[0] - matrix_order.min_distance
            else:
                change = - matrix_order.extra_summand
        elif sorted_index == S - 1:
            if increase:
                change = matrix_order.extra_summand
            else:
                change = sorted[S - 2] - sorted[S - 1] + matrix_order.min_distance
        else:
            current = sorted[sorted_index]
            if increase:
                lower, upper = matrix_order.obtain_increase_bounds(sorted_index)
                mid = (upper - lower) / 2.0
                min_Q = min(upper - current, current - lower)
                change = mid - min_Q
            else:
                lower, upper = matrix_order.obtain_decrease_bounds(sorted_index)
                mid = (lower - upper) / 2.0
                min_Q = min(upper - current, current - lower)
                change = mid + min_Q
        return change

    @staticmethod
    def decide_increase(matrix_order, i, j):
        sorted_index = matrix_order.get_sorted_index(i, j)
        sorted = matrix_order.unique
        if sorted_index == 0:
            second_min_distance = matrix_order.second_min_distance
            if matrix_order.min_index_lower == sorted_index and not np.isclose(second_min_distance,
                                                                               matrix_order.min_distance):
                increase = False
            else:
                increase = True
        elif sorted_index == sorted.shape[0] - 1:
            second_min_distance = matrix_order.second_min_distance
            if matrix_order.min_index_upper == sorted_index and not np.isclose(second_min_distance,
                                                                               matrix_order.min_distance):
                increase = True
            else:
                increase = False
        else:
            current = sorted[sorted_index]
            lower, _ = matrix_order.obtain_decrease_bounds(sorted_index)
            _, upper = matrix_order.obtain_increase_bounds(sorted_index)
            if current - lower < upper - current:
                increase = True
            else:
                increase = False
        return increase

    @staticmethod
    def set_to_zero():
        return False


HEURISTICS = {
    'greedy0': GreedyZero,
    'greedy': Greedy,
    'order':  MaintainOrder
}


class MatrixOrder:
    def __init__(self, matrix, precision=8):
        self.precision = precision
        self.matrix = matrix
        self.matrix = np.round(self.matrix, decimals=self.precision)
        self.sorted = np.sort(self.matrix, axis=None)
        self.unique, self.indices, self.sorted_indices, self.counts = np.unique(self.matrix,
                                                                                return_inverse=True,
                                                                                return_index=True,
                                                                                return_counts=True)
        self.distances = self.unique[1:] - self.unique[:-1]
        self.min_index_lower = np.argmin(self.distances)
        self.min_index_upper = self.min_index_lower + 1
        self.min_distance = self.distances[self.min_index_lower]
        self.exclusive_min_distance = np.min(self.distances[self.distances > self.min_distance])
        self.second_min_distance = np.min(np.r_[self.distances[:self.min_index_lower],
                                                self.distances[self.min_index_lower + 1:]])
        self.extra_summand = self.exclusive_min_distance - self.min_distance
        self.dynamic_range = np.log2((self.unique[-1] - self.unique[0]) / self.min_distance)
        self.unique_elements = len(self.unique)

    def update_entry(self, i, j, change, return_matrix=False):
        if not isinstance(change, str):
            new_value = self.matrix[i, j] + change
        else:
            new_value = self.unique[int(change)]
        new_value = np.round(new_value, decimals=self.precision)
        if return_matrix:
            matrix = self.matrix.copy()
            matrix[i, j] = new_value
            return matrix
        self.matrix[i, j] = new_value
        self.sorted = np.sort(self.matrix, axis=None)
        self.unique, self.indices, self.sorted_indices, self.counts = np.unique(self.matrix,
                                                                                return_inverse=True,
                                                                                return_index=True,
                                                                                return_counts=True)
        self.distances = self.unique[1:] - self.unique[:-1]
        self.min_index_lower = np.argmin(self.distances)
        self.min_index_upper = self.min_index_lower + 1
        self.min_distance = self.distances[self.min_index_lower]
        if (np.invert(self.distances > self.min_distance)).all():
            return True

        self.exclusive_min_distance = np.min(self.distances[self.distances > self.min_distance])
        self.second_min_distance = np.min(np.r_[self.distances[:self.min_index_lower],
                                                self.distances[self.min_index_lower + 1:]])
        self.extra_summand = self.exclusive_min_distance - self.min_distance
        self.dynamic_range = np.log2((self.unique[-1] - self.unique[0]) / self.min_distance)
        self.unique_elements = len(self.unique)
        return False

    def get_sorted_index(self, i, j):
        return self.sorted_indices[i * self.matrix.shape[0] + j]

    def obtain_increase_bounds(self, sorted_index):
        # sorted_index = self.get_sorted_index(i, j)
        upper_increase = self.unique[sorted_index + 1]
        if self.counts[sorted_index] > 1:
            lower_increase = self.unique[sorted_index]
        else:
            lower_increase = self.unique[sorted_index - 1]
        return lower_increase, upper_increase

    def obtain_decrease_bounds(self, sorted_index):
        # sorted_index = self.get_sorted_index(i, j)
        lower_decrease = self.unique[sorted_index - 1]
        if self.counts[sorted_index] > 1:
            upper_decrease = self.unique[sorted_index]
        else:
            upper_decrease = self.unique[sorted_index + 1]
        return lower_decrease, upper_decrease

    def dynamic_range_impact(self):
        elems = []
        # add border elements for max D
        if self.counts[0] == 1:
            elems.append(self.indices[0])
        if self.counts[-1] == 1:
            elems.append(self.indices[-1])
        # add elements for min D
        dists = self.distances
        min_dist_indices = np.where(np.isclose(dists, self.min_distance))[0]
        for dist_index in min_dist_indices:
            if self.unique[dist_index] != 0:
                if self.counts[dist_index] == 1:
                    if self.indices[dist_index] not in elems:
                        elems.append(self.indices[dist_index])
            if self.unique[dist_index + 1] != 0:
                if self.counts[dist_index + 1] == 1:
                    if self.indices[dist_index + 1] not in elems:
                        elems.append(self.indices[dist_index + 1])
        return elems

    @staticmethod
    def to_matrix_indices(indices, n):
        return [(index // n, index % n) for index in indices]
