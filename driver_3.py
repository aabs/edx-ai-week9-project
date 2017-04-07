import sys
import unittest

from copy import deepcopy


class Csp():
    def __init__(self):
        self.C = dict()  # tuple(x,y) -> func(x,y) -> bool

    def ac3(self, A, D) -> bool:
        queue = set([])
        # initialise the queue
        for k in self.C.keys():
            queue.add(k)

        while len(queue) > 0:
            Xi, Xj = queue.pop()
            if self.revise(D, Xi, Xj):
                if len(D[Xi]) == 0:
                    return False
                for Xk in self.neighbours(Xi, [Xj]):
                    queue.add((Xk, Xi))
        return True

    # given a choice of x for Xi, are there any values in Dj that satisfy the constraints between Xi and Xj?
    def can_be_satisfied(self, D, x, Xi, Xj):
        satisfactory = False
        for d in D[Xj]:
            for c in self.C[Xi, Xj]:
                if c(x, d):
                    satisfactory = True
        return satisfactory

    def revise(self, D, Xi, Xj) -> bool:
        result = False
        for x in D[Xi]:
            if not self.can_be_satisfied(D, x, Xi, Xj):
                D[Xi].remove(x)
                result = True
        return result

    # find the set of neighbours excepting all members of Ex
    def neighbours(self, X, Ex):
        # you are a neighbour if you share a binary constraint...
        result = [n for x, n in self.C.keys() if x == X and not n not in Ex]
        return result


def init_csp(grid):
    C = Csp()
    indexes = []

    A = dict()  # Assignments
    D = dict()  # Domain values
    tmp = [(i, int(d)) for i, d in enumerate(grid)]
    for i, x in tmp:
        A[i] = x

    # This sets up the initial domains of each cell, in preparation for the intial assignments to be inserted
    for x in range(0, 81):
        dx = A[x]
        if dx != 0:
            D[x] = [dx]
        else:
            D[x] = [v for v in range(1, 10)] # remember the upper bound is exclusive

    for row in range(0, 9):
        indexes = [x for x in range(row * 9, (row + 1) * 9)]
        setup_all_diff(C, indexes)

    for col in range(0, 9):
        indexes = []
        for row in range(0, 9):
            indexes.append((row * 9) + col)
        setup_all_diff(C, indexes)

    #############################
    indexes = []
    for row in range(0, 3):
        for col in range(0, 3):
            indexes.append((row * 9) + col)
    setup_all_diff(C, indexes)

    indexes = []
    for row in range(3, 6):
        for col in range(0, 3):
            indexes.append((row * 9) + col)
    setup_all_diff(C, indexes)

    indexes = []
    for row in range(6, 9):
        for col in range(0, 3):
            indexes.append((row * 9) + col)
    setup_all_diff(C, indexes)

    #############################
    indexes = []
    for row in range(0, 3):
        for col in range(3, 6):
            indexes.append((row * 9) + col)
    setup_all_diff(C, indexes)

    indexes = []
    for row in range(3, 6):
        for col in range(3, 6):
            indexes.append((row * 9) + col)
    setup_all_diff(C, indexes)

    indexes = []
    for row in range(6, 9):
        for col in range(3, 6):
            indexes.append((row * 9) + col)
    setup_all_diff(C, indexes)

    #############################
    indexes = []
    for row in range(0, 3):
        for col in range(6, 9):
            indexes.append((row * 9) + col)
    setup_all_diff(C, indexes)

    indexes = []
    for row in range(3, 6):
        for col in range(6, 9):
            indexes.append((row * 9) + col)
    setup_all_diff(C, indexes)

    indexes = []
    for row in range(6, 9):
        for col in range(6, 9):
            indexes.append((row * 9) + col)
    setup_all_diff(C, indexes)

    return C, A, D


def setup_all_diff(csp: Csp, xs) -> Csp:
    for i in range(0, len(xs)):
        for j in range(i + 1, len(xs)):
            pair = (xs[i], xs[j])
            pair2 = (xs[j], xs[i])
            if pair not in csp.C:
                csp.C[pair] = []
            csp.C[pair].append(lambda x, d: x != d)
            if pair2 not in csp.C:
                csp.C[pair2] = []
            csp.C[pair2].append(lambda x, d: x != d)


def backtracking_search(C, A, D):
    return backtrack(C, A, D)


# return [] to signify failure
def inference(csp, var, value):
    return []


def complete(A):
    for k in A.keys():
        if A[k] == 0:
            return False
    return True


def get_next_unassigned_position(A):
    for i in range(0, 81):
        if A[i] == 0:
            return i


def backtrack(C: Csp, A: dict, D: dict):
    if complete(A):
        return A
    i = get_next_unassigned_position(A)

    # for each possible assignment to that position
    for d in D[i]:
        # make a copy of the board and update it with the assignment
        A_ = deepcopy(A)
        D_ = deepcopy(D)
        A_[i] = d
        D_[i] = [d]

        # check if the board as assigned is consistent
        if C.ac3(A_, D_):
            # otherwise move on to make a new assignment in another position
            solution = backtrack(C, A_, D_)
            if solution is not None:
                return solution

    # if we got here (through every possible assignment of values) without finding an assignment that works, then this assignment must be rejected
    # we need to backtrack and try something else.
    return None


def assignment_to_string(A):
    result = ""
    for x in range(0, 81):
        result += str(A[x])
    return result


def main():
    solution = solve(sys.argv[1])
    with open("output.txt", "w") as of:
        of.write(assignment_to_string(solution))


def solve(problem):
    C, A, D = init_csp(problem)
    # parse problem string
    # build representation
    # for i, v in enumerate(grid):
    #     csp.D[i] = [v]
    # solve problem
    ok = C.ac3(A, D)
    if not ok:
        raise Exception("errm")
    solution = backtrack(C, A, D)
    return solution


if __name__ == "__main__":
    main()


class SolverTests(unittest.TestCase):
    def test_all_given_tests(self):
        with open("sodokus_start.txt") as fp:
            with open("sodokus_finish.txt") as fs:
                problems = fp.readlines()
                solutions = fs.readlines()
                test_cases = [(p, s) for (p, s) in zip(problems, solutions)]
                solved = 0
                for problem, expected in test_cases:
                    # set up the test
                    # run test
                    actual = assignment_to_string(solve(problem.strip()))
                    self.assertTrue(actual.count("0") == 0)
                    solved += 1
                    print(solved)
                    self.assertEqual(expected.strip(), actual.strip() )
                    # compare against provided solution
