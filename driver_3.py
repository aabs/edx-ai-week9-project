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


class SodokuBoard:
    def __init__(self, grid: list):
        self.original_assignment = grid
        self.A = dict()
        tmp = [(i, x) for i, x in enumerate(grid) if x != 0]
        for i, x in tmp:
            self.A[i] = x

    @property
    def assignments(self):
        return self.A

    def __getitem__(self, i):
        return self.A[i]

    def __setitem__(self, i, v):
        self.A[i] = v


def init_csp(grid):
    C = Csp()
    indexes = []

    A = dict()  # Assignments
    D = dict()  # Domain values
    tmp = [(i, int(d)) for i, d in enumerate(grid)]
    for i, x in tmp:
        A[i] = x

    for x in range(0, 81):
        dx = A[x]
        if dx != 0:
            D[x] = [dx]
        else:
            D[x] = [v for v in range(1, 10)]  # any number can be assigned (we'll clear out initial assignments later
    for row in range(0, 9):
        indexes = [x for x in range(row * 9, (row + 1) * 9)]
        setup_all_diff(C, indexes)
    for col in range(0, 9):
        indexes = []
        for row in range(0, 9):
            indexes.append((row * 9) + col)
        setup_all_diff(C, indexes)

    #############################
    for row in range(0, 3):
        indexes = []
        for col in range(0, 3):
            indexes.append((row * 9) + col)
    setup_all_diff(C, indexes)

    for row in range(3, 6):
        indexes = []
        for col in range(0, 3):
            indexes.append((row * 9) + col)
    setup_all_diff(C, indexes)

    for row in range(6, 9):
        indexes = []
        for col in range(0, 3):
            indexes.append((row * 9) + col)
    setup_all_diff(C, indexes)

    #############################
    for row in range(0, 3):
        indexes = []
        for col in range(3, 6):
            indexes.append((row * 9) + col)
    setup_all_diff(C, indexes)

    for row in range(3, 6):
        indexes = []
        for col in range(3, 6):
            indexes.append((row * 9) + col)
    setup_all_diff(C, indexes)

    for row in range(6, 9):
        indexes = []
        for col in range(3, 6):
            indexes.append((row * 9) + col)
    setup_all_diff(C, indexes)

    #############################
    for row in range(0, 3):
        indexes = []
        for col in range(6, 9):
            indexes.append((row * 9) + col)
    setup_all_diff(C, indexes)

    for row in range(3, 6):
        indexes = []
        for col in range(6, 9):
            indexes.append((row * 9) + col)
    setup_all_diff(C, indexes)

    for row in range(6, 9):
        indexes = []
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


def backtrack_(C, A, D):
    if complete(A):
        return A
    position = get_next_unassigned_position(A)
    for possible_assignment in D[position]:
        if C.ac3(A, position, possible_assignment):
            A[position] = possible_assignment
            inferences = inference(C, position, possible_assignment)
            if len(inferences) > 0:
                A.add_assignments(inferences)


def bt(C: Csp, A: dict, D: dict):
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
            bt1 = bt(C, A_, D_)
            if bt1 is not None:
                return bt1

    # if we got here (through every possible assignment of values without finding an assignment that works, then this assignment must be rejected
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
    solution = bt(C, A, D)
    return solution


if __name__ == "__main__":
    main()

def open_input(input_file):
    fo = open(input_file, "r")
    return fo


def open_output(output_file):
    fo = open(output_file, "w")
    return fo

class SolverTests(unittest.TestCase):
    def test_all_given_tests(self):
        with open("sodokus_start.txt") as fp:
            with open("sodokus_finish.txt") as fs:
                problems = fp.readlines()
                solutions = fs.readlines()
                test_cases = [(p,s) for (p,s) in zip(problems, solutions)]
                solved = 0
                for problem,expected in test_cases:
                    # set up the test
                    # run test
                    actual = assignment_to_string(solve(problem.strip()))
                    self.assertTrue(actual.count("0") == 0)
                    solved += 1
                    print(solved)
                    # self.assertEqual(expected, actual )
                    # compare against provided solution


