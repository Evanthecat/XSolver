import numpy as np
import matplotlib.pyplot as plt

# TODO: Optimize
# Page 17 Section 5.2 (64)
def evaluate_b_spline_piece(knot_set, p: int, N: int, t):
    p_full = knot_set.size - N - 1
    timerows = t[:, np.newaxis] * np.ones(N + p_full - p)

    if p == 0:
        # TODO: Remove constant matrices and add them to inputs for optimization
        knotrows = np.ones(t.size)[:, np.newaxis] * knot_set[:-1]
        knotrows_one = np.ones(t.size)[:, np.newaxis] * knot_set[1:]

        return np.equal(
            np.greater_equal(timerows, knotrows),
            np.less(timerows, knotrows_one)
        ).astype(np.float64)

    b_spline_lower = evaluate_b_spline_piece(knot_set, p - 1, N, t)

    first_term = np.zeros((np.size(b_spline_lower, 0), np.size(b_spline_lower, 1) - 1))
    second_term = np.zeros((np.size(b_spline_lower, 0), np.size(b_spline_lower, 1) - 1))

    knotrows = np.ones(t.size)[:, np.newaxis] * knot_set[:N + p_full - p]
    knotrows_one = np.ones(t.size)[:, np.newaxis] * knot_set[1:N + p_full - p + 1]
    knotrows_p = np.ones(t.size)[:, np.newaxis] * knot_set[p:N + p_full]
    knotrows_p_one = np.ones(t.size)[:, np.newaxis] * knot_set[p + 1:N + p_full + 1]

    subtracted_amount = p_full - p + 1

    if p_full > p:
        first_term[:, subtracted_amount:-subtracted_amount + 1] = (
            timerows[:, subtracted_amount:-subtracted_amount + 1] - knotrows[:, subtracted_amount:-subtracted_amount + 1]
        ) / (
            knotrows_p[:, subtracted_amount:-subtracted_amount + 1] - knotrows[:, subtracted_amount:-subtracted_amount + 1]
        ) * b_spline_lower[:, subtracted_amount:-subtracted_amount]

        second_term[:, subtracted_amount - 1:-subtracted_amount] = (
            1 - (timerows[:, subtracted_amount - 1:-subtracted_amount] - knotrows_one[:, subtracted_amount - 1:-subtracted_amount]
        ) / (
            knotrows_p_one[:, subtracted_amount - 1:-subtracted_amount] - knotrows_one[:, subtracted_amount - 1:-subtracted_amount]
        )) * b_spline_lower[:, subtracted_amount:-subtracted_amount]
    else:
        first_term[:, 1:] = (timerows[:, 1:] - knotrows[:, 1:]) / (knotrows_p[:, 1:] - knotrows[:, 1:]) * b_spline_lower[:, 1:-1]
        second_term[:, :-1] = (1 - (timerows[:, :-1] - knotrows_one[:, :-1]) / (knotrows_p_one[:, :-1] - knotrows_one[:, :-1])
            ) * b_spline_lower[:, 1:-1]

    return first_term + second_term

# Page 17 Section 5.2 (66)
def evaluate_b_spline(knot_set, p: int, N: int, t: float, f):
    return evaluate_b_spline_piece(knot_set, p, N, t) @ f

# Page 17 Section 5.2 (67)
def evaluate_b_spline_derivative(knot_set, p: int, N: int, t: float, f):
    values_array = np.zeros((t.size, N))
    piece = evaluate_m_spline_piece(knot_set, p, N, t)
    values_array[:, :-1] = piece[:, :-1] - piece[:, 1:]
    
    return values_array @ f

# Page 17 Section 5.2 (68)
def evaluate_m_spline_piece(knot_set, p: int, N: int, t: float):
    """Evaluates the ith degree-p M-Spline function at t."""
    # if knot_set[i + p] > knot_set[i]:
    knotrows = np.ones(t.size)[:, np.newaxis] * knot_set[:N]
    knotrows_p = np.ones(t.size)[:, np.newaxis] * knot_set[p:N + p]
    
    values_array = np.zeros_like(knotrows)
    values_array[:, 1:] = p / (knotrows_p[:, 1:] - knotrows[:, 1:]) * evaluate_b_spline_piece(knot_set, p - 1, N, t)[:, 1:-1]

    return values_array

def evaluate_m_spline(knot_set, p: int, N: int, t: float, f):
    return evaluate_m_spline_piece(knot_set, p, N, t) @ f

def evaluate_m_spline_derivative_piece(knot_set, p: int, i: int, t: float):
    if knot_set[i + p] > knot_set[i]:
        return p / (knot_set[i + p] - knot_set[i]) * (evaluate_m_spline_piece(knot_set, p - 1, i, t) - evaluate_m_spline_piece(knot_set, p - 1, i + 1, t))

    return 0

def evaluate_m_spline_derivative(knot_set, p: int, N: int, t: float, f):
    values_array = np.zeros((t.size, N))
    piece = evaluate_m_spline_piece(knot_set, p, N, t)

    knotrows = np.ones(t.size)[:, np.newaxis] * knot_set[:N]
    knotrows_p = np.ones(t.size)[:, np.newaxis] * knot_set[p:N + p]
    
    values_array[:, :-1] = p / (knotrows_p[:, 1:] - knotrows[:, 1:]) * piece[:, :-1] - piece[:, 1:]
    
    return values_array @ f

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
    graph_resolution = 1000
    p = 3
    N = 10
    f = np.array([
        0.1392126793708931,
        -0.593890798991142,
        0.4321920959540454,
        0.1894696162675895,
        -0.4428144308972264,
        0.850453270385332,
        0.37212327998772565,
        -0.45597708821691263,
        0.8202207309351939,
        0.5901881972887161,
    ])

    knot_set = generate_knot_set(N, p)
    values = np.linspace(0, 1, graph_resolution)
    # magnitudes = []
    # m_magnitudes = []
    # b_d_magnitudes = []
    # m_d_magnitudes = []

    magnitudes = evaluate_b_spline(knot_set, p, N, values, f)
    m_magnitudes = evaluate_m_spline(knot_set, p, N, values, f)
    b_d_magnitudes = evaluate_b_spline_derivative(knot_set, p, N, values, f)
    m_d_magnitudes = evaluate_m_spline_derivative(knot_set, p, N, values, f)

    plt.plot(values, np.array(magnitudes), '-', label="B-Spline")
    plt.plot(values, np.array(m_magnitudes), '-', label="M-Spline")
    plt.plot(values, np.array(b_d_magnitudes), '-', label="B-Spline Derivative")
    plt.plot(values, np.array(m_d_magnitudes), '-', label="M-Spline Derivative")
    plt.plot(np.linspace(0, 1, N), f, '.')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()