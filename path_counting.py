# CSIT332
# Abhyuday Singh

# Coded using basic memoization. Had to write something similar for a Project Euler problem last semester.
# Conditions check for whether boundary or diagonal is touched, or if vertical / horizontal movement is allowed.

# Formula used for calculating catalan number:
# C(n+1) = summation (i = 0) [ C(i) * C(n - i - 1) ] for n >= 0

def countpaths(n):
    silo = [[0] * (n+1) for a0 in range(n+1)]
    silo[0][0] = 1

    for i in range(n+1):
        for j in range(n+1):
            if i > 0 and i >= j:
                silo[i][j] = silo[i-1][j] + silo[i][j-1]
            elif 0 < i < j:
                silo[i][j] = silo[i][j-1]
            elif j > 0 and i == 0:
                silo[i][j] = silo[i][j-1]
    return silo[n][n]


def catalan(n):
    if n <= 1:
        return 1

    result = 0
    for i in range(n):
        result += catalan(i) * catalan(n - i - 1)
    return result


N = int(input("Enter size of nxn grid: "))
path_no = countpaths(N)
catalan_no = catalan(N+1)
print(f"The number of increasing paths in a {N}x{N} grid that don't cross x=y is {path_no}")
print(f"Number {N+1} in Catalan series is {catalan_no}")
