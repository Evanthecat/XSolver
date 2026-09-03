from interpolation import evaluate_2nd_degree_interpolation, generate_knot_set, evaluate_b_spline_piece, evaluate_m_spline_piece, evaluate_b_spline_derivative_piece, evaluate_m_spline_derivative_piece
from interpolation_pseudoinverse import pack_grid
from classes import Particles
import numpy as np
import matplotlib.pyplot as plt

def calculate_gradient(f_x, f_y, f_z, x_p, p):
    gradient = np.zeros((3, 3))

    N_x = f_x[:, 0, 0].size
    N_y = f_y[0, :, 0].size
    N_z = f_z[0, 0, :].size

    knot_set_x = generate_knot_set(N_x, p)
    knot_set_y = generate_knot_set(N_y, p)
    knot_set_z = generate_knot_set(N_z, p)

    for i in range(N_x):
        for j in range(N_y - 1):
            for k in range(N_z - 1):
                x_value = evaluate_b_spline_derivative_piece(knot_set_x, p, i, x_p[0])
                y_value = evaluate_m_spline_piece(knot_set_y, p, j + 1, x_p[1])
                z_value = evaluate_m_spline_piece(knot_set_z, p, k + 1, x_p[2])

                gradient[0][0] += x_value * y_value * z_value * f_x[i][j][k]

                x_value = evaluate_b_spline_piece(knot_set_x, p, i, x_p[0])
                y_value = evaluate_m_spline_derivative_piece(knot_set_y, p, j + 1, x_p[1])
                z_value = evaluate_m_spline_piece(knot_set_z, p, k + 1, x_p[2])

                gradient[0][1] += x_value * y_value * z_value * f_x[i][j][k]

                x_value = evaluate_b_spline_piece(knot_set_x, p, i, x_p[0])
                y_value = evaluate_m_spline_piece(knot_set_y, p, j + 1, x_p[1])
                z_value = evaluate_m_spline_derivative_piece(knot_set_z, p, k + 1, x_p[2])

                gradient[0][2] += x_value * y_value * z_value * f_x[i][j][k]

    for i in range(N_x - 1):
        for j in range(N_y):
            for k in range(N_z - 1):
                x_value = evaluate_m_spline_derivative_piece(knot_set_x, p, i + 1, x_p[0])
                y_value = evaluate_b_spline_piece(knot_set_y, p, j, x_p[1])
                z_value = evaluate_m_spline_piece(knot_set_z, p, k + 1, x_p[2])

                gradient[1][0] += x_value * y_value * z_value * f_y[i][j][k]

                x_value = evaluate_m_spline_piece(knot_set_x, p, i + 1, x_p[0])
                y_value = evaluate_b_spline_derivative_piece(knot_set_y, p, j, x_p[1])
                z_value = evaluate_m_spline_piece(knot_set_z, p, k + 1, x_p[2])

                gradient[1][1] += x_value * y_value * z_value * f_y[i][j][k]

                x_value = evaluate_m_spline_piece(knot_set_x, p, i + 1, x_p[0])
                y_value = evaluate_b_spline_piece(knot_set_y, p, j, x_p[1])
                z_value = evaluate_m_spline_derivative_piece(knot_set_z, p, k + 1, x_p[2])

                gradient[1][2] += x_value * y_value * z_value * f_y[i][j][k]

    for i in range(N_x - 1):
        for j in range(N_y - 1):
            for k in range(N_z):
                x_value = evaluate_m_spline_derivative_piece(knot_set_x, p, i + 1, x_p[0])
                y_value = evaluate_m_spline_piece(knot_set_y, p, j + 1, x_p[1])
                z_value = evaluate_b_spline_piece(knot_set_z, p, k, x_p[2])

                gradient[2][0] += x_value * y_value * z_value * f_z[i][j][k]

                x_value = evaluate_m_spline_piece(knot_set_x, p, i + 1, x_p[0])
                y_value = evaluate_m_spline_derivative_piece(knot_set_y, p, j + 1, x_p[1])
                z_value = evaluate_b_spline_piece(knot_set_z, p, k, x_p[2])

                gradient[2][1] += x_value * y_value * z_value * f_z[i][j][k]

                x_value = evaluate_m_spline_piece(knot_set_x, p, i + 1, x_p[0])
                y_value = evaluate_m_spline_piece(knot_set_y, p, j + 1, x_p[1])
                z_value = evaluate_b_spline_derivative_piece(knot_set_z, p, k, x_p[2])

                gradient[2][2] += x_value * y_value * z_value * f_z[i][j][k]

    return gradient

def calculate_derivative(state, f_x, f_y, f_z, p: int):
    x_p, psi_p = state

    N_x = f_x[:, 0, 0].size
    N_y = f_y[0, :, 0].size
    N_z = f_z[0, 0, :].size

    knot_set_x = generate_knot_set(N_x, p)
    knot_set_y = generate_knot_set(N_y, p)
    knot_set_z = generate_knot_set(N_z, p)

    dt_x_p = np.array(evaluate_2nd_degree_interpolation(knot_set_x, knot_set_y, knot_set_z, N_x, N_y, N_z, f_x, f_y, f_z, x_p[0], x_p[1], x_p[2], p))
    velocity_gradient = calculate_gradient(f_x, f_y, f_z, x_p, p)
    dt_psi_p = -psi_p @ velocity_gradient

    return dt_x_p, dt_psi_p

# Page 22 Section 6.5.3 (87)
def apply_advection(particles, timestep: float, f_x, f_y, f_z, p):
    new_particles = []
    for i, particle in enumerate(particles):
        print(f"\r[Advection Progress]: Particle {i + 1}/{len(particles)}", end='', flush=True)
        # print()

        k_1 = calculate_derivative((particle.position, particle.jacobian), f_x, f_y, f_z, p)
        k_1 = (timestep * k_1[0], timestep * k_1[1])
        # print()
        # print("k_1:")
        # print(f"Position: {k_1[0]}")
        # print(f"Jacobian:")
        # print(k_1[1])
        # print()

        input_state = (particle.position + k_1[0] / 2, particle.jacobian + k_1[1] / 2)
        k_2 = calculate_derivative(input_state, f_x, f_y, f_z, p)
        k_2 = (timestep * k_2[0], timestep * k_2[1])
        # print()
        # print("k_2:")
        # print(f"Position: {k_2[0]}")
        # print(f"Jacobian:")
        # print(k_2[1])
        # print()

        input_state = (particle.position + k_2[0] / 2, particle.jacobian + k_2[1] / 2)
        k_3 = calculate_derivative(input_state, f_x, f_y, f_z, p)
        k_3 = (timestep * k_3[0], timestep * k_3[1])
        # print()
        # print("k_3:")
        # print(f"Position: {k_3[0]}")
        # print(f"Jacobian:")
        # print(k_3[1])
        # print()

        input_state = (particle.position + k_3[0], particle.jacobian + k_3[1])
        k_4 = calculate_derivative(input_state, f_x, f_y, f_z, p)
        k_4 = (timestep * k_4[0], timestep * k_4[1])
        # print()
        # print("k_4:")
        # print(f"Position: {k_4[0]}")
        # print(f"Jacobian:")
        # print(k_4[1])
        # print()

        new_x_p = (k_1[0] + 2 * k_2[0] + 2 * k_3[0] + k_4[0]) / 6
        new_psi_p = (k_1[1] + 2 * k_2[1] + 2 * k_3[1] + k_4[1]) / 6
        # print("Added Values:")
        # print(f"Position: {new_x_p}")
        # print("Jacobian:")
        # print(new_psi_p)

        # print()
        # print(particle.position)
        # print(particle.jacobian)
        new_particles.append(Particles(
            particle.rest_position, 
            particle.rest_impulse, 
            particle.rest_time, 
            particle.position + new_x_p,
            particle.jacobian + new_psi_p
        ))

        # print(particle.position)
        # print(particle.jacobian)
        # print()

    return new_particles

def main():
    # p = 3
    # particles = []
    # timestep = 1

    # N_x = 5
    # N_y = 5
    # N_z = 5

    # particles.append(Particle((0.5, 0.5, 0.5), (0.1, 0.1, 0.1), 0.0))

    # f_x = np.linspace(0, 1, N_y - 1)[:, None, None] * np.linspace(0, 1, N_x)[None, :, None] * np.linspace(0, 1, N_z - 1)[None, None, :]
    # f_y = np.linspace(0, 1, N_y)[:, None, None] * np.linspace(0, 1, N_x - 1)[None, :, None] * np.linspace(0, 1, N_z - 1)[None, None, :]
    # f_z = np.linspace(0, 1, N_y - 1)[:, None, None] * np.linspace(0, 1, N_x - 1)[None, :, None] * np.linspace(0, 1, N_z)[None, None, :]

    # print(f"Timestep: {timestep}")
    # print()

    # print("Original Values:")
    # print(f"Position: {particles[0].position}")
    # print("Jacobian:")
    # print(particles[0].jacobian)
    # print(f"Impulse: {particles[0].jacobian.T @ particles[0].rest_impulse}")
    # print()

    # print("Particle Gradient:")
    # print(calculate_gradient(f_x, f_y, f_z, particles[0].position, p))
    # print()

    # derivative = calculate_derivative((particles[0].position, particles[0].jacobian), f_x, f_y, f_z, p)
    # position = derivative[0]
    # jacobian = derivative[1]

    # print("Particle Derivative:")
    # print(f"Position Change: {position[0]}, {position[1]}, {position[2]}")
    # print("Jacobian Change:")
    # print(jacobian)
    # print(f"Impulse Change: {jacobian.T @ particles[0].rest_impulse}")

    # print()
    # apply_advection(particles, timestep, f_x, f_y, f_z, p)
    # print()
    # print()

    # print("Final Values:")
    # print(f"Position: {particles[0].position}")
    # print("Jacobian:")
    # print(particles[0].jacobian)
    # print(f"Impulse: {particles[0].jacobian.T @ particles[0].rest_impulse}")

    p = 3
    particle_grid_resolution = 10
    particles = []

    with open("div_free_velocity_field.txt") as file:
        data_blocks = file.read().split("\n\n")

        N_x = int(data_blocks[0])
        N_y = int(data_blocks[1])
        N_z = int(data_blocks[2])

        f_x = pack_grid(np.array(data_blocks[3].split('\n')), N_x, N_y - 1, N_z - 1)
        f_y = pack_grid(np.array(data_blocks[4].split('\n')), N_x - 1, N_y, N_z - 1)
        f_z = pack_grid(np.array(data_blocks[5].strip('\n').split('\n')), N_x - 1, N_y - 1, N_z)

    knot_set_x = generate_knot_set(N_x, p)
    knot_set_y = generate_knot_set(N_y, p)
    knot_set_z = generate_knot_set(N_z, p)

    particle_x = np.linspace(0, 1, particle_grid_resolution)
    particle_y = np.linspace(0, 1, particle_grid_resolution)
    particle_z = np.linspace(0, 1, particle_grid_resolution)

    particle_values_x = np.zeros((particle_grid_resolution, particle_grid_resolution, particle_grid_resolution))
    particle_values_y = np.zeros((particle_grid_resolution, particle_grid_resolution, particle_grid_resolution))
    particle_values_z = np.zeros((particle_grid_resolution, particle_grid_resolution, particle_grid_resolution))

    print()

    for i in range(particle_grid_resolution):
        for j in range(particle_grid_resolution):
            for k in range(particle_grid_resolution):
                x, y, z = evaluate_2nd_degree_interpolation(
                    knot_set_x, knot_set_y, knot_set_z, N_x, N_y, N_z, f_x, f_y, f_z, particle_x[
                        i], particle_y[j], particle_z[k], p
                )

                print(f"\r[Interpolation Progress]: Particle: X: {i + 1}/{particle_grid_resolution} Y: {j + 1}/{particle_grid_resolution} Z: {k + 1}/{particle_grid_resolution}", end='', flush=True)

                particles.append(Particles((particle_x[i], particle_y[j], particle_z[k]), (x, y, z), 0.0))

                particle_values_x[i][j][k] = x
                particle_values_y[i][j][k] = y
                particle_values_z[i][j][k] = z

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.quiver(*np.meshgrid(particle_x, particle_y, particle_z), particle_values_x, particle_values_y, particle_values_z, length=0.05)
    plt.show()

    print()

    particles = apply_advection(particles, 2, f_x, f_y, f_z, p)

    particle_advection_values_x = []
    particle_advection_values_y = []
    particle_advection_values_z = []

    particle_advection_positions_x = []
    particle_advection_positions_y = []
    particle_advection_positions_z = []

    for particle in particles:
        impulse = particle.jacobian.T @ particle.rest_impulse
        particle_advection_values_x.append(impulse[0])
        particle_advection_values_y.append(impulse[1])
        particle_advection_values_z.append(impulse[2])
    
        particle_advection_positions_x.append(particle.position[0])
        particle_advection_positions_y.append(particle.position[1])
        particle_advection_positions_z.append(particle.position[2])

        with open("log.txt", 'a') as file:
            file.write(f"{impulse[0]}\n")
            file.write(f"{impulse[1]}\n")
            file.write(f"{impulse[2]}\n")

            file.write(f"{particle.position[0]}\n")
            file.write(f"{particle.position[1]}\n")
            file.write(f"{particle.position[2]}\n")
            file.write(f"\n")

    # particle_advection_values_x = pack_grid(np.array(particle_advection_values_x), particle_grid_resolution, particle_grid_resolution, particle_grid_resolution)
    # particle_advection_values_y = pack_grid(np.array(particle_advection_values_y), particle_grid_resolution, particle_grid_resolution, particle_grid_resolution)
    # particle_advection_values_z = pack_grid(np.array(particle_advection_values_z), particle_grid_resolution, particle_grid_resolution, particle_grid_resolution)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.quiver(particle_advection_positions_x, particle_advection_positions_y, particle_advection_positions_z, 
        particle_advection_values_x,
        particle_advection_values_y,
        particle_advection_values_z,
        length=0.05)
    plt.show()

    print()
    print("Finished!")

if __name__ == "__main__":
    main()