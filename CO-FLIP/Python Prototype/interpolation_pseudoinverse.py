import numpy as np
import matplotlib.pyplot as plt
import random
import math
from classes import Particles

from interpolation import evaluate_b_spline_piece, evaluate_b_spline, evaluate_m_spline_piece, evaluate_2nd_degree_interpolation, generate_knot_set

def pack_grid(grid, N_x: int, N_y: int, N_z: int):
    packed_grid = np.zeros((N_x, N_y, N_z))

    for index in range(grid.size):
        k = math.floor(index / (N_x * N_y))
        j = math.floor((index - k * N_x * N_y
            ) / N_x)
        i = index - k * N_x * N_y - j * N_x

        packed_grid[i][j][k] = grid[index]

    return packed_grid

def unpack_grid(grid):
    flattened_grid = np.zeros(grid.size)

    N_x = grid[:, 0, 0].size
    N_y = grid[0, :, 0].size
    N_z = grid[0, 0, :].size

    for i in range(N_x):
        for j in range(N_y):
            for k in range(N_z):
                flattened_grid[
                    i + k * N_x * N_y + j * N_x
                ] = grid[i][j][k]

    return flattened_grid

def get_packed_index(index: int, N_x: int, N_y: int, N_z: int):
    k = math.floor(index / (N_x * N_y))
    j = math.floor((index - k * N_x * N_y
        ) / N_x)
    i = index - k * N_x * N_y - j * N_x

    return i, j, k

def get_unpacked_index(i: int, j: int, k: int, N_x: int, N_y: int, N_z: int):
    return i + k * N_x * N_y + j * N_x

def generate_interpolation_matrix(particles, N_x: int, N_y: int, N_z: int, p: int):
    x_block_size = N_x * (N_y - 1) * (N_z - 1)
    y_block_size = (N_x - 1) * N_y * (N_z - 1)
    z_block_size = (N_x - 1) * (N_y - 1) * N_z

    matrix_size = x_block_size + y_block_size + z_block_size
    
    interpolation_matrix = np.zeros((len(particles) * 3, matrix_size))

    knot_set_x = generate_knot_set(N_x, p)
    knot_set_y = generate_knot_set(N_y, p)
    knot_set_z = generate_knot_set(N_z, p)

    print()

    for particle_index in range(len(particles)):
        print(f"\r[Evaluating Particle Basis Functions]: {particle_index + 1}/{len(particles)}", end='', flush=True)

        x_basis_functions = np.zeros((N_x, N_y - 1, N_z - 1))
        for i in range(N_x):
            for j in range(1, N_y):
                for k in range(1, N_z):
                    x_value = evaluate_b_spline_piece(knot_set_x, p, i, particles[particle_index].position[0])
                    y_value = evaluate_m_spline_piece(knot_set_y, p, j, particles[particle_index].position[1])
                    z_value = evaluate_m_spline_piece(knot_set_z, p, k, particles[particle_index].position[2])

                    x_basis_functions[i][j - 1][k - 1] = x_value * y_value * z_value

        interpolation_matrix[particle_index][:x_block_size] = unpack_grid(x_basis_functions)

        y_basis_functions = np.zeros((N_x - 1, N_y, N_z - 1))
        for i in range(1, N_x):
            for j in range(N_y):
                for k in range(1, N_z):
                    x_value = evaluate_m_spline_piece(knot_set_x, p, i, particles[particle_index].position[0])
                    y_value = evaluate_b_spline_piece(knot_set_y, p, j, particles[particle_index].position[1])
                    z_value = evaluate_m_spline_piece(knot_set_z, p, k, particles[particle_index].position[2])

                    y_basis_functions[i - 1][j][k - 1] = x_value * y_value * z_value

        interpolation_matrix[particle_index + len(particles)][x_block_size:x_block_size + y_block_size] = unpack_grid(y_basis_functions)

        z_basis_functions = np.zeros((N_x - 1, N_y - 1, N_z))
        for i in range(1, N_x):
            for j in range(1, N_y):
                for k in range(N_z):
                    x_value = evaluate_m_spline_piece(knot_set_x, p, i, particles[particle_index].position[0])
                    y_value = evaluate_m_spline_piece(knot_set_y, p, j, particles[particle_index].position[1])
                    z_value = evaluate_b_spline_piece(knot_set_z, p, k, particles[particle_index].position[2])

                    z_basis_functions[i - 1][j - 1][k] = x_value * y_value * z_value

        interpolation_matrix[particle_index + 2 * len(particles)][x_block_size + y_block_size:] = unpack_grid(z_basis_functions)

    return interpolation_matrix

# Page 8 Section 3.3 (15)
def calculate_pseudoinverse(particles, weights, N_x: int, N_y: int, N_z: int, p: int):
    x_block_size = N_x * (N_y - 1) * (N_z - 1)
    y_block_size = (N_x - 1) * N_y * (N_z - 1)

    interpolation_matrix = generate_interpolation_matrix(particles, N_x, N_y, N_z, p)

    A = interpolation_matrix.T @ weights @ interpolation_matrix
    b = interpolation_matrix.T @ weights @ np.concatenate((
        np.array([(particle.jacobian.T @ particle.rest_impulse)[0] for particle in particles]),
        np.array([(particle.jacobian.T @ particle.rest_impulse)[1] for particle in particles]),
        np.array([(particle.jacobian.T @ particle.rest_impulse)[2] for particle in particles])
    ))

    x_i = 0
    r_i = b
    d_i = r_i

    print()

    while True:
        a_i = (r_i.T @ r_i) / (d_i.T @ A @ d_i)
        x_k = x_i + a_i * d_i
        r_k = r_i - a_i * A @ d_i
        b_k = (r_k.T @ r_k) / (r_i.T @ r_i)
        d_k = r_k + b_k * d_i

        print(f"\r[CG Solver Progress]: Error: {np.linalg.norm(x_i - x_k)}", end='', flush=True)

        if np.linalg.norm(x_i - x_k) < 1e-3:
            break

        x_i = x_k
        r_i = r_k
        d_i = d_k

    f_x = pack_grid(x_k[:x_block_size], N_x, N_y - 1, N_z - 1)
    f_y = pack_grid(x_k[x_block_size:x_block_size + y_block_size], N_x - 1, N_y, N_z -1)
    f_z = pack_grid(x_k[x_block_size + y_block_size:], N_x - 1, N_y - 1, N_z)

    return f_x, f_y, f_z

def main():
    # Packing test
    array = np.linspace(1, 18, 18)
    assert np.linalg.norm(array - unpack_grid(pack_grid(array, 2, 3, 3))) == 0

    # Interpolation test
    N_x = 3
    N_y = 3
    N_z = 3
    p = 3
    particle_resolution = 3

    X_x, Y_x, Z_x = np.meshgrid(np.linspace(0, 1, N_y - 1), np.linspace(0, 1, N_x), np.linspace(0, 1, N_z - 1))
    f_x = X_x * Y_x * Z_x

    X_y, Y_y, Z_y = np.meshgrid(np.linspace(0, 1, N_y), np.linspace(0, 1, N_x - 1), np.linspace(0, 1, N_z - 1))
    f_y = X_y * Y_y * Z_y

    X_z, Y_z, Z_z = np.meshgrid(np.linspace(0, 1, N_y - 1), np.linspace(0, 1, N_x - 1), np.linspace(0, 1, N_z))
    f_z = X_z * Y_z * Z_z

    f = np.concatenate((unpack_grid(f_x), unpack_grid(f_y), unpack_grid(f_z)))

    knot_set_x = generate_knot_set(N_x, p)
    knot_set_y = generate_knot_set(N_y, p)
    knot_set_z = generate_knot_set(N_z, p)

    particles = []

    for i in range(particle_resolution):
        for j in range(particle_resolution):
            for k in range(particle_resolution):
                particles.append(Particles(np.array(((i + 0.5) / particle_resolution, (j + 0.5) / particle_resolution, (k + 0.5) / particle_resolution)), 
                    np.array((i + 1, j + 1, k + 1)), 0.0))

    interpolation_matrix = generate_interpolation_matrix(particles, N_x, N_y, N_z, p)
    matrix_values = interpolation_matrix @ f
    interpolation_values_x = [evaluate_2nd_degree_interpolation(knot_set_x, knot_set_y, knot_set_z, N_x, N_y, 
        N_z, f_x, f_y, f_z, particle.position[0], particle.position[1], particle.position[2], p)[0] for particle in particles]
    interpolation_values_y = [evaluate_2nd_degree_interpolation(knot_set_x, knot_set_y, knot_set_z, N_x, N_y, 
            N_z, f_x, f_y, f_z, particle.position[0], particle.position[1], particle.position[2], p)[1] for particle in particles]
    interpolation_values_z = [evaluate_2nd_degree_interpolation(knot_set_x, knot_set_y, knot_set_z, N_x, N_y, 
            N_z, f_x, f_y, f_z, particle.position[0], particle.position[1], particle.position[2], p)[2] for particle in particles]

    assert np.linalg.norm(matrix_values - np.concatenate((
        interpolation_values_x, interpolation_values_y, interpolation_values_z))) == 0
    
    # 2D test
    N = 10
    p = 3
    particle_count = 20
    weights = np.diag([N / particle_count for _ in range(particle_count)]) # ???
    interpolation_matrix = np.zeros((particle_count, N))
    particle_positions = [random.random() for _ in range(particle_count)]
    particle_impulses = np.array([random.random() for _ in range(particle_count)])
    knot_set = generate_knot_set(N, p)
    graph_resolution = 1000

    for particle in range(particle_count):
        for i in range(N):
            interpolation_matrix[particle][i] = evaluate_b_spline_piece(knot_set, p, i, particle_positions[particle])

    A = interpolation_matrix.T @ weights @ interpolation_matrix
    b = interpolation_matrix.T @ weights @ particle_impulses

    x_i = 0
    r_i = b
    d_i = r_i

    while True:
        a_i = (r_i.T @ r_i) / (d_i.T @ A @ d_i)
        x_k = x_i + a_i * d_i
        r_k = r_i - a_i * A @ d_i
        b_k = (r_k.T @ r_k) / (r_i.T @ r_i)
        d_k = r_k + b_k * d_i

        if np.linalg.norm(x_i - x_k) < 1e-3:
            break

        x_i = x_k
        r_i = r_k
        d_i = d_k

    interpolated_t = np.linspace(0, 1, graph_resolution)
    interpolated_values = []
    for t in interpolated_t:
        interpolated_values.append(evaluate_b_spline(knot_set, p, t, x_k))

    plt.plot(particle_positions, particle_impulses, '.')
    plt.plot(interpolated_t, interpolated_values)
    plt.show()

    # 3D test
    particle_grid_resolution = 10
    particles = []

    torus_radius = 0.35
    tube_radius = 0.15
    torus_x = 0.5
    torus_y = 0.5
    torus_z = 0.5

    N_x = 10
    N_y = 10
    N_z = 10

    weights = np.diag(np.ones(particle_grid_resolution ** 3 * 3) * N_x * N_y * N_z / (particle_grid_resolution ** 3))

    particle_x = np.linspace(0, 1, particle_grid_resolution)
    particle_y = np.linspace(0, 1, particle_grid_resolution)
    particle_z = np.linspace(0, 1, particle_grid_resolution)

    particle_values_x = np.zeros((particle_grid_resolution, particle_grid_resolution, particle_grid_resolution))
    particle_values_y = np.zeros((particle_grid_resolution, particle_grid_resolution, particle_grid_resolution))
    particle_values_z = np.zeros((particle_grid_resolution, particle_grid_resolution, particle_grid_resolution))

    for i in range(particle_grid_resolution):
        for j in range(particle_grid_resolution):
            for k in range(particle_grid_resolution):
                in_range = (torus_radius - math.sqrt((particle_x[i] - torus_x) ** 2 + (
                    particle_y[j] - torus_y) ** 2)) ** 2 + (particle_z[k] - torus_z) ** 2 <= tube_radius ** 2
                impulse = np.array((-1.0 if in_range else 0.0, -1.0 if in_range else 0.0, -1.0 if in_range else 0.0))
                
                particles.append(Particles(np.array((particle_x[i], particle_y[j], particle_z[k])), impulse, 0.0))

                particle_values_x[i][j][k] =  impulse[0]
                particle_values_y[i][j][k] = impulse[1]
                particle_values_z[i][j][k] = impulse[2]

    plt.style.use('_mpl-gallery')

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.quiver(*np.meshgrid(particle_x, particle_y, particle_z), particle_values_x, particle_values_y, particle_values_z, length=0.05)
    plt.show()

    f_x, f_y, f_z = calculate_pseudoinverse(particles, weights, N_x, N_y, N_z, p)

    particle_values_final_x = np.zeros((particle_grid_resolution, particle_grid_resolution, particle_grid_resolution))
    particle_values_final_y = np.zeros((particle_grid_resolution, particle_grid_resolution, particle_grid_resolution))
    particle_values_final_z = np.zeros((particle_grid_resolution, particle_grid_resolution, particle_grid_resolution))

    knot_set_x = generate_knot_set(N_x, p)
    knot_set_y = generate_knot_set(N_y, p)
    knot_set_z = generate_knot_set(N_z, p)

    print()

    for i in range(len(particle_x)):
        for j in range(len(particle_y)):
            for k in range(len(particle_z)):
                x, y, z = evaluate_2nd_degree_interpolation(
                    knot_set_x, knot_set_y, knot_set_z, N_x, N_y, N_z, f_x, f_y, f_z, particle_x[
                        i], particle_y[j], particle_z[k], p
                )

                print(f"\r[Interpolation Progress]: Particle: X: {i + 1}/{len(particle_x)} Y: {j + 1}/{len(particle_y)} Z: {k + 1}/{len(particle_z)}", end='', flush=True)

                particle_values_final_x[i][j][k] = x
                particle_values_final_y[i][j][k] = y
                particle_values_final_z[i][j][k] = z

    print()
    print("Finished!")

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.quiver(*np.meshgrid(particle_x, particle_y, particle_z), particle_values_final_x, particle_values_final_y, particle_values_final_z, length=0.05)
    plt.show()

if __name__ == "__main__":
    main()