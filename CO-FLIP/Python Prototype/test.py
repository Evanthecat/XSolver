import numpy as np
import random
import math
from interpolation import evaluate_m_spline_piece, evaluate_b_spline_piece, generate_knot_set, evaluate_b_spline, evaluate_b_spline_derivative
from interpolation_pseudoinverse import get_packed_index, get_unpacked_index
import matplotlib.pylab as plt
from scipy.interpolate import BSpline

def cg():
    A = np.array([
        [1, 2],
        [3, 4]
    ])

    b = np.array(
        [5, 11]
    )

    x_i = 0
    r_i = b
    d_i = r_i

    for i in range(100):
        a_i = (r_i.T @ r_i) / (d_i.T @ A @ d_i)
        x_k = x_i + a_i * d_i
        r_k = r_i - a_i * A @ d_i
        b_k = (r_k.T @ r_k) / (r_i.T @ r_i)
        d_k = r_k + b_k * d_i

        print(x_k)

        x_i = x_k
        r_i = r_k
        d_i = d_k

def monte_carlo():
    samples = 1000
    p = 3

    N_x = 5
    N_y = 5
    N_z = 5

    knot_set_x = generate_knot_set(N_x, p)
    knot_set_y = generate_knot_set(N_y, p)
    knot_set_z = generate_knot_set(N_z, p)

    i = 30
    j = i

    total = 0

    index_i_x = math.floor(i / ((N_y - 1) * (N_z - 1)))
    index_i_y = math.floor((i - index_i_x * (N_y - 1) * (N_z - 1)) / (N_z - 1))
    index_i_z = i - index_i_x * (N_y - 1) * (N_z - 1) - index_i_y * (N_z - 1)
    index_j_x = math.floor(j / ((N_y - 1) * (N_z - 1)))
    index_j_y = math.floor((j - index_j_x * (N_y - 1) * (N_z - 1)) / (N_z - 1))
    index_j_z = j - index_j_x * (N_y - 1) * (N_z - 1) - index_j_y * (N_z - 1)

    t = np.linspace(0, 1, 100)
    values_x = []
    values_y = []
    values_z = []

    for i in t:
        values_x.append(evaluate_b_spline_piece(knot_set_x, p, index_i_x, i))
        values_y.append(evaluate_m_spline_piece(knot_set_y, p, index_i_y + 1, i))
        values_z.append(evaluate_m_spline_piece(knot_set_z, p, index_i_z + 1, i))

    plt.plot(t, values_x, '-')
    plt.plot(t, values_y, '-')
    plt.plot(t, values_z, '-')

    print()
    for i in range(samples):
        print(f"\r[Monte Carlo Integration]: Evaluating Sample {i + 1}/{samples}", end='', flush=True)
        random_point = (
            random.uniform(knot_set_x[index_i_x], knot_set_x[index_i_x + p + 1]),
            random.uniform(knot_set_y[index_i_y + 1], knot_set_y[index_i_y + p + 1]),
            random.uniform(knot_set_z[index_i_x + 1], knot_set_z[index_i_z + p + 1])
        )

        x_value = evaluate_b_spline_piece(knot_set_x, p, index_i_x, random_point[0])
        y_value = evaluate_m_spline_piece(knot_set_y, p, index_i_y + 1, random_point[1])
        z_value = evaluate_m_spline_piece(knot_set_z, p, index_i_z + 1, random_point[2])
        plt.plot(random_point[0], x_value, '.')
        plt.plot(random_point[1], y_value, '.')
        plt.plot(random_point[2], z_value, '.')

        i_value = x_value * y_value * z_value

        x_value = evaluate_b_spline_piece(knot_set_x, p, index_j_x, random_point[0])
        y_value = evaluate_m_spline_piece(knot_set_y, p, index_j_y + 1, random_point[1])
        z_value = evaluate_m_spline_piece(knot_set_z, p, index_j_z + 1, random_point[2])

        j_value = x_value * y_value * z_value

        total += i_value * j_value

    total *= (
        knot_set_x[index_i_x + p + 1] - knot_set_x[index_i_x]) * (
            knot_set_y[index_i_y + p + 1] - knot_set_y[index_i_y]) * (
                knot_set_z[index_i_z + p + 1] - knot_set_z[index_i_x]
    )

    print()
    print(total / samples)
    plt.show()

def matrix_multiplication():
    mat1 = np.zeros((2, 2))
    mat2 = np.zeros((2, 2))

    mat1[0][0] = 1
    mat1[1][0] = 2
    mat1[0][1] = 3
    mat1[1][1] = 4

    mat2[0][0] = 1
    mat2[1][0] = 2
    mat2[0][1] = 3
    mat2[1][1] = 4

    print(mat1)
    print()
    print(mat2)
    print()

    print(mat1 @ mat2)

def one_d_gradient():
    p = 3
    N = 10
    x = 0.1
    resolution = 1000

    knot_set = generate_knot_set(N, p)
    f = [
        0.10263132422410925,
        0.07655999005659819,
        0.2474696492890609,
        0.25168695784501527,
        0.13444919577783188,
        0.04278688528822938,
        0.5155511789885362,
        0.8121176896221981,
        0.19256669786736946,
        0.48497165566548606
    ]

    magnitudes = []
    t = np.linspace(0, 1, resolution)
    for value in t:
        magnitudes.append(evaluate_b_spline(knot_set, p, value, f))

    point_value = evaluate_b_spline(knot_set, p, x, f)

    plt.plot(t, magnitudes, '-')
    plt.plot(x, point_value, '.')
    plt.plot([x, x + evaluate_b_spline_derivative(knot_set, p, x, f)], [point_value, point_value])
    plt.show()

def gaussian_quadrature():
    p = 3
    N = 10
    graph_resolution = 1000
    knot_set = generate_knot_set(N, p)
    f = [
        0.49677733847697214,
        0.6675347747172513,
        0.6290865627599491,
        0.2677584481145915,
        0.3467756517996905,
        0.8869194327534721,
        0.023960887847491397,
        0.9161300595673169,
        0.8031725339004729,
        0.7866013523386274
    ]

    magnitudes = []
    t_values = np.linspace(0, 1, graph_resolution)

    for t in t_values:
        magnitudes.append(evaluate_b_spline(knot_set, p, t, f))

    total = 0.0

    for index in range(N - 1):
        a = index * 1 / (N - 1)
        b = (index + 1) * 1 / (N - 1)
        nodes = [-1 / math.sqrt(3), 1 / math.sqrt(3)]
        weights = [1.0, 1.0]
        n = 2

        sum = 0.0

        for i in range(n):
            sum += weights[i] * evaluate_b_spline(knot_set, p, (b - a) / 2 * nodes[i] + (a + b) / 2, f)

        print(f"Integral of Section {index + 1}: {(b - a) / 2 * sum}")
        section_integral = (b - a) / 2 * sum
        total += section_integral

        section_values = []
        basis_t_values = np.linspace(a, b, graph_resolution)

        for t in basis_t_values:
            section_values.append(evaluate_b_spline(knot_set, p, t, f))

        plt.plot(basis_t_values, section_values)
        plt.plot((a, b), (section_integral / (b - a), section_integral / (b - a)))
        plt.show()

    print()
    print(f"Total Integral of B-Spline: {total}")

    plt.plot(t_values, magnitudes)
    plt.plot((0, 1), (total, total), label="Calculated Integral")

    plt.legend()

    plt.show()

def inner_product_b():
    N_x = 10
    N_y = 10
    N_z = 10

    p = 3

    knot_set_x = generate_knot_set(N_x, p)

    nodes = [
        -0.3399810435848563,
        0.3399810435848563,
        -0.8611363115940526,
        0.8611363115940526
    ]
    weights = [
        0.6521451548625461,
        0.6521451548625461,
        0.3478548451374538,
        0.3478548451374538
    ]
    n = 4

    for index_pair in [(1, 2), (2, 1)]:
        i, j = index_pair
        index_i_x, index_i_y, index_i_z = get_packed_index(i, N_x, N_y - 1, N_z - 1)
        index_j_x, index_j_y, index_j_z = get_packed_index(j, N_x, N_y - 1, N_z - 1)
        print(index_i_x)

        total_x = 0.0

        # TODO: Optimize with range knot_set_x[index_i_x] to knot_set_x[index_i_x + p + 1]
        for index in range(N_x):
            print(index)
            
            a = index * 1 / (N_x - 1)
            b = (index + 1) * 1 / (N_x - 1)
            
            sum_x = 0.0

            for node_index in range(n):
                sum_x += weights[node_index] * evaluate_b_spline_piece(knot_set_x, p, index_i_x, (b - a) / 2 * nodes[node_index] + (a + b) / 2
                    ) * evaluate_b_spline_piece(knot_set_x, p, index_j_x, (b - a) / 2 * nodes[node_index] + (a + b) / 2)
                plt.plot((b - a) / 2 * nodes[node_index] + (a + b) / 2, evaluate_b_spline_piece(knot_set_x, p, index_i_x, (b - a) / 2 * nodes[node_index] + (a + b) / 2
                    ) * evaluate_b_spline_piece(knot_set_x, p, index_j_x, (b - a) / 2 * nodes[node_index] + (a + b) / 2), '.')

            total_x += (b - a) / 2 * sum_x

            # t_values = np.linspace(a, b, 1000)
            # magnitudes = []
            # b_magnitudes = []

            # for t in t_values:
            #     magnitudes.append(evaluate_b_spline_piece(knot_set_x, p, index_i_x, t
            #         ) * evaluate_b_spline_piece(knot_set_x, p, index_j_x, t))
            #     b_magnitudes.append(evaluate_b_spline_piece(knot_set_x, p, index_i_x, t))

            # plt.plot(t_values, magnitudes, label="Product Function")
            # plt.plot(t_values, b_magnitudes, label="ith Basis Function")
            # calculated_integral = (b - a) / 2 * sum_x
            # plt.plot((a, b), (calculated_integral / (b - a), calculated_integral / (b - a)), label="Integral")

            # plt.legend()

            # plt.show()

        t_values = np.linspace(0, 1, 1000)
        i_values = []
        j_values = []
        product_values = []

        for t in t_values:
            i_values.append(evaluate_b_spline_piece(knot_set_x, p, index_i_x, t))
            j_values.append(evaluate_b_spline_piece(knot_set_x, p, index_j_x, t))
            product_values.append(evaluate_b_spline_piece(knot_set_x, p, index_i_x, t) * evaluate_b_spline_piece(knot_set_x, p, index_j_x, t))

        approx_total = 0

        for value in product_values:
            approx_total += value / 1000

        print(knot_set_x)
        print(approx_total)
        print(total_x)


        # plt.plot(t_values, i_values, label="ith Basis Function")
        # plt.plot(t_values, j_values, label="jth Basis Function")
        plt.plot(t_values, product_values, label="Product Function")
        plt.plot((0, 1), (approx_total, approx_total), label="Approximated Integral")
        plt.plot((0, 1), (total_x, total_x), label="Quadrature Integral")

        plt.legend()

        plt.show()

def inner_product_m():
    i = 2
    j = 1

    N_x = 10
    N_y = 10
    N_z = 10

    p = 3

    knot_set_x = generate_knot_set(N_x, p)

    nodes = [
        -0.3399810435848563,
        0.3399810435848563,
        -0.8611363115940526,
        0.8611363115940526
    ]
    weights = [
        0.6521451548625461,
        0.6521451548625461,
        0.3478548451374538,
        0.3478548451374538
    ]
    n = 4

    index_i_x, index_i_y, index_i_z = get_packed_index(i, N_x, N_y - 1, N_z - 1)
    index_j_x, index_j_y, index_j_z = get_packed_index(j, N_x, N_y - 1, N_z - 1)
    print(index_i_x)

    total_x = 0.0

    # TODO: Optimize with range knot_set_x[index_i_x] to knot_set_x[index_i_x + p + 1]
    for index in range(N_x):
        print(index)
        
        a = index * 1 / (N_x - 1)
        b = (index + 1) * 1 / (N_x - 1)
        
        sum_x = 0.0

        for node_index in range(n):
            sum_x += weights[node_index] * evaluate_m_spline_piece(knot_set_x, p, index_i_x + 1, (b - a) / 2 * nodes[node_index] + (a + b) / 2
                ) * evaluate_m_spline_piece(knot_set_x, p, index_j_x + 1, (b - a) / 2 * nodes[node_index] + (a + b) / 2)

        total_x += (b - a) / 2 * sum_x

    t_values = np.linspace(0, 1, 1000)
    i_values = []
    j_values = []
    product_values = []

    for t in t_values:
        i_values.append(evaluate_m_spline_piece(knot_set_x, p, index_i_x, t))
        j_values.append(evaluate_m_spline_piece(knot_set_x, p, index_j_x, t))
        product_values.append(evaluate_m_spline_piece(knot_set_x, p, index_i_x + 1, t) * evaluate_m_spline_piece(knot_set_x, p, index_j_x + 1, t))

    approx_total = 0

    for value in product_values:
        approx_total += value / 1000

    print(knot_set_x)
    print(approx_total)
    print(total_x)


    # plt.plot(t_values, i_values, label="ith Basis Function")
    # plt.plot(t_values, j_values, label="jth Basis Function")
    plt.plot(t_values, product_values, label="Product Function")
    plt.plot((0, 1), (approx_total, approx_total), label="Approximated Integral")
    plt.plot((0, 1), (total_x, total_x), label="Quadrature Integral")

    plt.legend()

    plt.show()

inner_product_b()