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