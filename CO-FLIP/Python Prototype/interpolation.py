import numpy as np
import matplotlib.pyplot as plt

# TODO: Optimize
# Page 17 Section 5.2 (64)
def evaluate_b_spline_piece(knot_set, p: int, t):
    if p == 0:
        # TODO: Remove constant matrices and add them to inputs for optimization
        N = knot_set.size - p - 1
        knotrows = knot_set[:-1, np.newaxis] * np.ones(t.size)
        knotrows_one = knot_set[1:, np.newaxis] * np.ones(t.size)
        timerows = np.ones(knot_set.size - 1)[:, np.newaxis] * t

        return np.equal(
            np.greater_equal(timerows, knotrows),
            np.less(timerows, knotrows_one)
        ).astype(np.float64)

    b_spline_lower = evaluate_b_spline_piece(knot_set, p - 1, t)

    first_term = np.zeros((np.size(b_spline_lower, 0), np.size(b_spline_lower, 1) - 1))
    second_term = np.zeros((np.size(b_spline_lower, 0), np.size(b_spline_lower, 1) - 1))

    # Hope the range is correct :/
    first_term[1:, :] = 

    # if knot_set[i + p] != knot_set[i]:
    #     first_term = (t - knot_set[i]) / (
    #         knot_set[i + p] - knot_set[i]) * evaluate_b_spline_piece(knot_set, p - 1, t)
    # if knot_set[i + p + 1] != knot_set[i + 1]:
    #     second_term = (knot_set[i + p + 1] - t) / (
    #         knot_set[i + p + 1] - knot_set[i + 1]) * evaluate_b_spline_piece(knot_set, p - 1, i + 1, t)

    # return first_term + second_term

# Page 17 Section 5.2 (66)
def evaluate_b_spline(knot_set, p: int, t: float, f):
    # Potential implementation:
    # 1. Create matrix of time values
    # 2. Create 2 arrays of knot set values (to deal with generation of p = 0 matrix)
    # 3. Somehow run compare operations between them
    return np.sum(evaluate_b_spline_piece(knot_set, p, t), 0)

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
    graph_resolution = 10000
    p = 0
    N = 10
    f = np.array([
        # 0.1392126793708931,
        # -0.593890798991142,
        # 0.4321920959540454,
        # 0.1894696162675895,
        # -0.4428144308972264,
        # 0.850453270385332,
        # 0.37212327998772565,
        # -0.45597708821691263,
        # 0.8202207309351939,
        # 0.5901881972887161,
        1,1,1,1,1,1,1,1,1,1
    ])

    knot_set = generate_knot_set(N, p)
    values = np.linspace(0, 1, graph_resolution)
    # magnitudes = []
    # m_magnitudes = []
    # b_d_magnitudes = []
    # m_d_magnitudes = []

    magnitudes = evaluate_b_spline(knot_set, p, values, f)
    # m_magnitudes.append(evaluate_m_spline(knot_set, p, t, f))
    # b_d_magnitudes.append(evaluate_b_spline_derivative(knot_set, p, t, f))
    # m_d_magnitudes.append(evaluate_m_spline_derivative(knot_set, p, t, f))

    plt.plot(values, np.array(magnitudes), '-', label="B-Spline")
    # plt.plot(values, np.array(m_magnitudes), '-', label="M-Spline")
    # plt.plot(values, np.array(b_d_magnitudes), '-', label="B-Spline Derivative")
    # plt.plot(values, np.array(m_d_magnitudes), '-', label="M-Spline Derivative")
    plt.plot(np.linspace(0, 1, N), f, '.')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()