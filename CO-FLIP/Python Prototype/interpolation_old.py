import numpy as np
import matplotlib.pyplot as plt

# TODO: Optimize
# Page 17 Section 5.2 (64)
def evaluate_b_spline_piece(knot_set, p: int, i: int, t: float):
    if p == 0:
        return 1.0 if knot_set[i] <= t < knot_set[i + 1] else 0.0

    first_term = 0.0
    second_term = 0.0
    piece = None
    piece1 = None

    if knot_set[i + p] != knot_set[i]:
        piece = evaluate_b_spline_piece(knot_set, p - 1, i, t)
        first_term = (t - knot_set[i]) / (
            knot_set[i + p] - knot_set[i]) * piece
    if knot_set[i + p + 1] != knot_set[i + 1]:
        piece1 = evaluate_b_spline_piece(knot_set, p - 1, i + 1, t)
        second_term = (knot_set[i + p + 1] - t) / (
            knot_set[i + p + 1] - knot_set[i + 1]) * piece1

    print(f"Degree {p} i {i}: \nt: {t} \nknotset: {knot_set[i]} \nknotsetp: {knot_set[i + p]} \nknotsetp1: {knot_set[i + p + 1]} \nknotset1: {knot_set[i + 1]} \npiece: {piece} \npiece1: {piece1}\n")

    return first_term + second_term

# Page 17 Section 5.2 (66)
def evaluate_b_spline(knot_set, p: int, t: float, f):
    N = knot_set.size - p - 1

    total = 0.0

    for i in range(N):
        total += f[i] * evaluate_b_spline_piece(knot_set, p, i, t)

    return total

# Page 17 Section 5.2 (67)
def evaluate_b_spline_derivative_piece(knot_set, p: int, i: int, t: float):
    return evaluate_m_spline_piece(knot_set, p, i, t) - evaluate_m_spline_piece(knot_set, p, i + 1, t)

def evaluate_b_spline_derivative(knot_set, p: int, t: float, f):
    N = knot_set.size - p - 1

    total = 0.0

    for i in range(N):
        total += f[i] * evaluate_b_spline_derivative_piece(knot_set, p, i, t)

    return total

# Page 17 Section 5.2 (68)
def evaluate_m_spline_piece(knot_set, p: int, i: int, t: float):
    """Evaluates the ith degree-p M-Spline function at t."""
    if knot_set[i + p] > knot_set[i]:
        return p / (knot_set[i + p] - knot_set[i]) * evaluate_b_spline_piece(knot_set, p - 1, i, t)

    return 0

def evaluate_m_spline(knot_set, p: int, t: float, f):
    N = knot_set.size - p - 1

    total = 0.0

    for i in range(N):
        total += f[i] * evaluate_m_spline_piece(knot_set, p, i, t)

    return total

def evaluate_m_spline_derivative_piece(knot_set, p: int, i: int, t: float):
    if knot_set[i + p] > knot_set[i]:
        return p / (knot_set[i + p] - knot_set[i]) * (evaluate_m_spline_piece(knot_set, p - 1, i, t) - evaluate_m_spline_piece(knot_set, p - 1, i + 1, t))

    return 0

def evaluate_m_spline_derivative(knot_set, p: int, t: float, f):
    N = knot_set.size - p - 1

    total = 0.0

    for i in range(N):
        total += f[i] * evaluate_m_spline_derivative_piece(knot_set, p, i, t)

    return total

# Page 18 section 5.2 (75)
def evaluate_2nd_degree_interpolation(
        knot_set_x, knot_set_y, knot_set_z, 
        N_x: int, N_y: int, N_z: int, 
        f_x, f_y, f_z,
        t_1: float, t_2: float, t_3: float, 
        p: int
    ):
    """Evaluates the 2nd degree interpolation operator at point (t_1, t_2, t_3)"""
    # TODO: Optimize with locality
    total_x = 0
    total_y = 0
    total_z = 0

    for i in range(N_x):
        for j in range(N_y - 1):
            for k in range(N_z - 1):
                x_value = evaluate_b_spline_piece(knot_set_x, p, i, t_1)
                y_value = evaluate_m_spline_piece(knot_set_y, p, j + 1, t_2)
                z_value = evaluate_m_spline_piece(knot_set_z, p, k + 1, t_3)

                total_x += x_value * y_value * z_value * f_x[i][j][k]

    for i in range(N_x - 1):
        for j in range(N_y):
            for k in range(N_z - 1):
                x_value = evaluate_m_spline_piece(knot_set_x, p, i + 1, t_1)
                y_value = evaluate_b_spline_piece(knot_set_y, p, j, t_2)
                z_value = evaluate_m_spline_piece(knot_set_z, p, k + 1, t_3)

                total_y += x_value * y_value * z_value * f_y[i][j][k]

    for i in range(N_x - 1):
        for j in range(N_y - 1):
            for k in range(N_z):
                x_value = evaluate_m_spline_piece(knot_set_x, p, i + 1, t_1)
                y_value = evaluate_m_spline_piece(knot_set_y, p, j + 1, t_2) 
                z_value = evaluate_b_spline_piece(knot_set_z, p, k, t_3) 

                total_z += x_value * y_value * z_value * f_z[i][j][k]

    return total_x, total_y, total_z

# Page 17 Section 5.2 (65)
def generate_knot_set(N: int, p: int):
    """Generates a knot set of size N + p + 1 in a compact parameter domain [0, 1]."""
    # Note that the number N + 2p + 1 is different from the paper's N + p + 1 knots, 
    # which I suspect is incorrect.
    return np.concatenate((np.zeros(p), np.linspace(0, 1, N - p + 1), np.ones(p)))

def main():
    p = 3
    N = 10
    f = np.array([
        1,1,1,1,1,1,1,1,1,1,
    ])

    knot_set = generate_knot_set(N, p)

    print(evaluate_b_spline(knot_set, p, 0.5, f))

if __name__ == "__main__":
    main()