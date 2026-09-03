import numpy as np
import matplotlib.pyplot as plt
import math
from interpolation_pseudoinverse import get_packed_index, get_unpacked_index, unpack_grid, pack_grid
from interpolation import generate_knot_set, evaluate_m_spline_piece, evaluate_b_spline_piece

# Page 18 Section 5.2 (72)
# TODO: Optimize with SIMD or similar
def generate_d_0(N_x: int, N_y: int, N_z: int):
    x_block_size = (N_x - 1) * N_y * N_z
    y_block_size = N_x * (N_y - 1) * N_z
    z_block_size = N_x * N_y * (N_z - 1)
    grid_size = N_x * N_y * N_z

    matrix_size = x_block_size + y_block_size + z_block_size

    d_0 = np.zeros((matrix_size, grid_size))

    for i in range(x_block_size):
        index_x, index_y, index_z = get_packed_index(i, N_x - 1, N_y, N_z)

        index = get_unpacked_index(index_x + 1, index_y, index_z, N_x, N_y, N_z)
        d_0[i][index] = 1

        subtracted_index = get_unpacked_index(index_x, index_y, index_z, N_x, N_y, N_z)
        d_0[i][subtracted_index] = -1

    for i in range(y_block_size):
        index_x, index_y, index_z = get_packed_index(i, N_x, N_y - 1, N_z)

        index = get_unpacked_index(index_x, index_y + 1, index_z, N_x, N_y, N_z)
        d_0[i + x_block_size][index] = 1

        subtracted_index = get_unpacked_index(index_x, index_y, index_z, N_x, N_y, N_z)
        d_0[i + x_block_size][subtracted_index] = -1

    for i in range(z_block_size):
        index_x, index_y, index_z = get_packed_index(i, N_x, N_y, N_z - 1)

        index = get_unpacked_index(index_x, index_y, index_z + 1, N_x, N_y, N_z)
        d_0[i + x_block_size + y_block_size][index] = 1

        subtracted_index = get_unpacked_index(index_x, index_y, index_z, N_x, N_y, N_z)
        d_0[i + x_block_size + y_block_size][subtracted_index] = -1

    return d_0

def generate_d_1(N_x: int, N_y: int, N_z: int):
    x_block_size = N_x * (N_y - 1) * (N_z - 1)
    y_block_size = (N_x - 1) * N_y * (N_z - 1)
    z_block_size = (N_x - 1) * (N_y - 1) * N_z

    matrix_size = x_block_size + y_block_size + z_block_size

    h_x_size = (N_x - 1) * N_y * N_z
    h_y_size = N_z * (N_y - 1) * N_z
    h_z_size = N_x * N_y * (N_z - 1)

    d_0_size = h_x_size + h_y_size + h_z_size
    d_1 = np.zeros((matrix_size, d_0_size))

    for i in range(x_block_size):
        index_x, index_y, index_z = get_packed_index(i, N_x, N_y - 1, N_z - 1)

        index = get_unpacked_index(index_x, index_y + 1, index_z, N_x, N_y, N_z - 1)
        d_1[i][index + h_x_size + h_y_size] = 1

        subtracted_index = get_unpacked_index(index_x, index_y, index_z, N_x, N_y, N_z - 1)
        d_1[i][subtracted_index + h_x_size + h_y_size] = -1

        index = get_unpacked_index(index_x, index_y, index_z + 1, N_x, N_y - 1, N_z)
        d_1[i][index + h_x_size] = -1

        subtracted_index = get_unpacked_index(index_x, index_y, index_z, N_x, N_y - 1, N_z)
        d_1[i][subtracted_index + h_x_size] = 1

    for i in range(y_block_size):
        index_x, index_y, index_z = get_packed_index(i, N_x - 1, N_y, N_z - 1)

        index = get_unpacked_index(index_x, index_y, index_z + 1, N_x - 1, N_y, N_z)
        d_1[i + x_block_size][index] = 1

        subtracted_index = get_unpacked_index(index_x, index_y, index_z, N_x - 1, N_y, N_z)
        d_1[i + x_block_size][subtracted_index] = -1

        index = get_unpacked_index(index_x + 1, index_y, index_z, N_x, N_y, N_z - 1)
        d_1[i + x_block_size][index + h_x_size + h_y_size] = -1

        subtracted_index = get_unpacked_index(index_x, index_y, index_z, N_x, N_y, N_z - 1)
        d_1[i + x_block_size][subtracted_index + h_x_size + h_y_size] = 1

    for i in range(z_block_size):
        index_x, index_y, index_z = get_packed_index(i, N_x - 1, N_y - 1, N_z)

        index = get_unpacked_index(index_x + 1, index_y, index_z, N_x, N_y - 1, N_z)
        d_1[i + x_block_size + y_block_size][index + h_x_size] = 1

        subtracted_index = get_unpacked_index(index_x, index_y, index_z, N_x, N_y - 1, N_z)
        d_1[i + x_block_size + y_block_size][subtracted_index + h_x_size] = -1

        index = get_unpacked_index(index_x, index_y + 1, index_z, N_x - 1, N_y, N_z)
        d_1[i + x_block_size + y_block_size][index] = -1

        subtracted_index = get_unpacked_index(index_x, index_y, index_z, N_x - 1, N_y, N_z)
        d_1[i + x_block_size + y_block_size][subtracted_index] = 1

    return d_1

def generate_galerkin_hodge_star(N_x: int, N_y: int, N_z: int, p: int):
    x_block_size = N_x * (N_y - 1) * (N_z - 1)
    y_block_size = (N_x - 1) * N_y * (N_z - 1)
    z_block_size = (N_x - 1) * (N_y - 1) * N_z

    matrix_size = x_block_size + y_block_size + z_block_size

    galerkin_hodge_star = np.zeros((matrix_size, matrix_size))

    knot_set_x = generate_knot_set(N_x, p)
    knot_set_y = generate_knot_set(N_y, p)
    knot_set_z = generate_knot_set(N_z, p)

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

    print()
    print("Calculating Galerkin Hodge Star")

    for i in range(x_block_size):
        for j in range(x_block_size):
            print(f"\r[Calculating Inner Product] X Coefficient: {i + 1}, {j + 1}/{x_block_size}, {x_block_size}", end='', flush=True)

            index_i_x, index_i_y, index_i_z = get_packed_index(i, N_x, N_y - 1, N_z - 1)
            index_j_x, index_j_y, index_j_z = get_packed_index(j, N_x, N_y - 1, N_z - 1)

            total_x = 0.0
            total_y = 0.0
            total_z = 0.0

            # Maybe needs to be N_x - 1 when m-splines?
            for index in range(N_x):
                a = index * 1 / (N_x - 1)
                b = (index + 1) * 1 / (N_x - 1)
                
                sum_x = 0.0

                for node_index in range(n):
                    sum_x += weights[node_index] * evaluate_b_spline_piece(knot_set_x, p, index_i_x, (b - a) / 2 * nodes[node_index] + (a + b) / 2
                        ) * evaluate_b_spline_piece(knot_set_x, p, index_j_x, (b - a) / 2 * nodes[node_index] + (a + b) / 2)

                total_x += (b - a) / 2 * sum_x

            for index in range(N_y):
                a = index * 1 / (N_y - 1)
                b = (index + 1) * 1 / (N_y - 1)
                
                sum_y = 0.0

                for node_index in range(n):
                    sum_y += weights[node_index] * evaluate_m_spline_piece(knot_set_y, p, index_i_y + 1, (b - a) / 2 * nodes[node_index] + (a + b) / 2
                        ) * evaluate_m_spline_piece(knot_set_y, p, index_j_y + 1, (b - a) / 2 * nodes[node_index] + (a + b) / 2)

                total_y += (b - a) / 2 * sum_y

            for index in range(N_z):
                a = index * 1 / (N_z - 1)
                b = (index + 1) * 1 / (N_z - 1)
                
                sum_z = 0.0

                for node_index in range(n):
                    sum_z += weights[node_index] * evaluate_m_spline_piece(knot_set_z, p, index_i_z + 1, (b - a) / 2 * nodes[node_index] + (a + b) / 2
                        ) * evaluate_m_spline_piece(knot_set_z, p, index_j_z + 1, (b - a) / 2 * nodes[node_index] + (a + b) / 2)

                total_z += (b - a) / 2 * sum_z

            galerkin_hodge_star[i][j] = total_x * total_y * total_z

    for i in range(y_block_size):
        for j in range(y_block_size):
            print(f"\r[Calculating Inner Product] Y Coefficient: {i + 1}, {j + 1}/{y_block_size}, {y_block_size}", end='', flush=True)

            index_i_x, index_i_y, index_i_z = get_packed_index(i, N_x - 1, N_y, N_z - 1)
            index_j_x, index_j_y, index_j_z = get_packed_index(j, N_x - 1, N_y, N_z - 1)

            total_x = 0.0
            total_y = 0.0
            total_z = 0.0
            
            for index in range(N_x):
                a = index * 1 / (N_x - 1)
                b = (index + 1) * 1 / (N_x - 1)
                
                sum_x = 0.0

                for node_index in range(n):
                    sum_x += weights[node_index] * evaluate_m_spline_piece(knot_set_x, p, index_i_x + 1, (b - a) / 2 * nodes[node_index] + (a + b) / 2
                        ) * evaluate_m_spline_piece(knot_set_x, p, index_j_x + 1, (b - a) / 2 * nodes[node_index] + (a + b) / 2)

                total_x += (b - a) / 2 * sum_x

            for index in range(N_y):
                a = index * 1 / (N_y - 1)
                b = (index + 1) * 1 / (N_y - 1)
                
                sum_y = 0.0

                for node_index in range(n):
                    sum_y += weights[node_index] * evaluate_b_spline_piece(knot_set_y, p, index_i_y, (b - a) / 2 * nodes[node_index] + (a + b) / 2
                        ) * evaluate_b_spline_piece(knot_set_y, p, index_j_y, (b - a) / 2 * nodes[node_index] + (a + b) / 2)

                total_y += (b - a) / 2 * sum_y

            for index in range(N_z):
                a = index * 1 / (N_z - 1)
                b = (index + 1) * 1 / (N_z - 1)
                
                sum_z = 0.0

                for node_index in range(n):
                    sum_z += weights[node_index] * evaluate_m_spline_piece(knot_set_z, p, index_i_z + 1, (b - a) / 2 * nodes[node_index] + (a + b) / 2
                        ) * evaluate_m_spline_piece(knot_set_z, p, index_j_z + 1, (b - a) / 2 * nodes[node_index] + (a + b) / 2)

                total_z += (b - a) / 2 * sum_z

            galerkin_hodge_star[i + x_block_size][j + x_block_size] = total_x * total_y * total_z

    for i in range(z_block_size):
        for j in range(z_block_size):
            print(f"\r[Calculating Inner Product] Z Coefficient: {i + 1}, {j + 1}/{z_block_size}, {z_block_size}", end='', flush=True)

            index_i_x, index_i_y, index_i_z = get_packed_index(i, N_x - 1, N_y - 1, N_z)
            index_j_x, index_j_y, index_j_z = get_packed_index(j, N_x - 1, N_y - 1, N_z)

            total_x = 0.0
            total_y = 0.0
            total_z = 0.0
            
            for index in range(N_x):
                a = index * 1 / (N_x - 1)
                b = (index + 1) * 1 / (N_x - 1)
                
                sum_x = 0.0

                for node_index in range(n):
                    sum_x += weights[node_index] * evaluate_m_spline_piece(knot_set_x, p, index_i_x + 1, (b - a) / 2 * nodes[node_index] + (a + b) / 2
                        ) * evaluate_m_spline_piece(knot_set_x, p, index_j_x + 1, (b - a) / 2 * nodes[node_index] + (a + b) / 2)

                total_x += (b - a) / 2 * sum_x

            for index in range(N_y):
                a = index * 1 / (N_y - 1)
                b = (index + 1) * 1 / (N_y - 1)
                
                sum_y = 0.0

                for node_index in range(n):
                    sum_y += weights[node_index] * evaluate_m_spline_piece(knot_set_y, p, index_i_y + 1, (b - a) / 2 * nodes[node_index] + (a + b) / 2
                        ) * evaluate_m_spline_piece(knot_set_y, p, index_j_y + 1, (b - a) / 2 * nodes[node_index] + (a + b) / 2)

                total_y += (b - a) / 2 * sum_y

            for index in range(N_z):
                a = index * 1 / (N_z - 1)
                b = (index + 1) * 1 / (N_z - 1)
                
                sum_z = 0.0

                for node_index in range(n):
                    sum_z += weights[node_index] * evaluate_b_spline_piece(knot_set_z, p, index_i_z, (b - a) / 2 * nodes[node_index] + (a + b) / 2
                        ) * evaluate_b_spline_piece(knot_set_z, p, index_j_z, (b - a) / 2 * nodes[node_index] + (a + b) / 2)

                total_z += (b - a) / 2 * sum_z

            galerkin_hodge_star[i + x_block_size + y_block_size][j + x_block_size + y_block_size] = total_x * total_y * total_z

    print()

    return galerkin_hodge_star

def main():
    # d_0
    N = 3
    d_0 = generate_d_0(N, N, N)

    plt.figure(figsize=(5, 5))
    plt.pcolormesh(d_0, cmap="RdYlGn")
    plt.ylim(54, 0)
    plt.show()

    # d_1
    d_1 = generate_d_1(N, N, N)

    plt.figure(figsize=(5, 5))
    plt.pcolormesh(d_1, cmap="RdYlGn")
    plt.ylim(36, 0)
    plt.show()

    # Galerkin-Hodge Star
    galerkin_hodge_star = generate_galerkin_hodge_star(N, N, N, 3)

    plt.pcolormesh(galerkin_hodge_star, cmap="RdYlGn")
    plt.ylim(36, 0)
    plt.colorbar()
    plt.show()

    # Tests
    N = 2

    d0 = generate_d_0(N, N, N)
    grid = unpack_grid(np.linspace(1, N, N)[:, None, None] * np.linspace(1, N, N)[None, :, None] * np.linspace(1, N, N)[None, None, :])
    d1_grid = d0 @ grid
    d1_grid_x = pack_grid(d1_grid[:(N - 1) * N ** 2], N - 1, N, N)

    print("Simple Grid and Edges:")
    print(grid)
    print(d1_grid_x)

    zero_matrix = d_1 @ d_0

    # plt.figure(figsize=(5, 5))
    # plt.pcolormesh(zero_matrix, cmap="RdYlGn")
    # plt.ylim(36, 0)
    # plt.show()

    assert np.linalg.norm(zero_matrix - np.zeros_like(zero_matrix)) == 0
    assert np.linalg.norm(galerkin_hodge_star - galerkin_hodge_star.T) == 0

if __name__ == "__main__":
    main()