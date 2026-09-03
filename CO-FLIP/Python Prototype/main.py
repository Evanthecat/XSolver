from interpolation_pseudoinverse import calculate_pseudoinverse, pack_grid, unpack_grid
from interpolation import evaluate_2nd_degree_interpolation, generate_knot_set
from pressure_projection import evaluate_pressure_projection
from operators import generate_galerkin_hodge_star, generate_d_0, generate_d_1
from advection import apply_advection
from classes import Particles
import numpy as np
import matplotlib.pyplot as plt

def integrator(particles, grid, weights, N_x: int, N_y: int, N_z: int, p: int, timestep: float, tolerence, galerkin_hodge_star, d_0, d_1):
    particle_to_grid = calculate_pseudoinverse(particles, weights, N_x, N_y, N_z, p)
    tau = evaluate_pressure_projection(*particle_to_grid, galerkin_hodge_star, d_0, d_1)
    f_star_x = unpack_grid(particle_to_grid[0]) + tau[0]
    f_star_y = unpack_grid(particle_to_grid[1]) + tau[1]
    f_star_z = unpack_grid(particle_to_grid[2]) + tau[2]
    f_star = np.array((f_star_x, f_star_y, f_star_z))
    initial_grid_velocity = f_star

    y_0 = apply_advection(particles, timestep, 
        pack_grid(f_star[0], N_x, N_y - 1, N_z - 1), 
        pack_grid(f_star[1], N_x - 1, N_y, N_z - 1), 
        pack_grid(f_star[2], N_x - 1, N_y - 1, N_z), 
        p
    ) # Hopefully this is right
    particle_to_grid = calculate_pseudoinverse(y_0, weights, N_x, N_y, N_z, p)
    tau = evaluate_pressure_projection(*particle_to_grid, galerkin_hodge_star, d_0, d_1)
    y_0_grid_x = unpack_grid(particle_to_grid[0]) + tau[0]
    y_0_grid_y = unpack_grid(particle_to_grid[1]) + tau[1]
    y_0_grid_z = unpack_grid(particle_to_grid[2]) + tau[2]
    y_0_grid = np.array((y_0_grid_x, y_0_grid_y, y_0_grid_z))

    while True:
        y_new = apply_advection(particles, timestep, 
            pack_grid(f_star[0], N_x, N_y - 1, N_z - 1), 
            pack_grid(f_star[1], N_x - 1, N_y, N_z - 1), 
            pack_grid(f_star[2], N_x - 1, N_y - 1, N_z), 
        p)

        particle_to_grid = calculate_pseudoinverse(y_new, weights, N_x, N_y, N_z, p)
        tau = evaluate_pressure_projection(*particle_to_grid, galerkin_hodge_star, d_0, d_1)
        f_new_x = unpack_grid(particle_to_grid[0]) + tau[0]
        f_new_y = unpack_grid(particle_to_grid[1]) + tau[1]
        f_new_z = unpack_grid(particle_to_grid[2]) + tau[2]
        f_new = np.array((f_new_x, f_new_y, f_new_z))

        f_star = 1 / 2 * (np.array((grid[0], grid[1], grid[2])) + f_new)
        # f_difference = f_new - grid
        # f_new = grid + evaluate_pressure_projection(...)

        if np.linalg.norm(f_new - initial_grid_velocity) / np.linalg.norm(y_0_grid) < tolerence:
            break

    f_new_x = f_new[:grid[0].size]
    f_new_y = f_new[grid[0].size:grid[0].size + grid[1].size]
    f_new_z = f_new[grid[0].size + grid[1].size:]

    return y_new, (f_new_x, f_new_y, f_new_z)

def main():
    iterations = 10
    timestep = 0.1
    particle_grid_resolution = 10
    particles = []
    p = 3

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

    galerkin_hodge_star = generate_galerkin_hodge_star(N_x, N_y, N_z, p)
    d_0 = generate_d_0(N_x, N_y, N_z)
    d_1 = generate_d_1(N_x, N_y, N_z)

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

    f = (unpack_grid(f_x), unpack_grid(f_y), unpack_grid(f_z))
    y = particles

    for _ in range(iterations):
        y_new, f_new = integrator(
            y, 
            f, 
            np.identity(particle_grid_resolution ** 3 * 3) / (particle_grid_resolution ** 3), 
            N_x, N_y, N_z, 
            p, 
            timestep, 
            10 ** -7,
            galerkin_hodge_star, d_0, d_1
        )

        f = f_new
        y = y_new

        particle_advection_values_x = []
        particle_advection_values_y = []
        particle_advection_values_z = []
    
        particle_advection_positions_x = []
        particle_advection_positions_y = []
        particle_advection_positions_z = []
    
        for particle in y:
            impulse = particle.jacobian.T @ particle.rest_impulse
            particle_advection_values_x.append(impulse[0])
            particle_advection_values_y.append(impulse[1])
            particle_advection_values_z.append(impulse[2])
        
            particle_advection_positions_x.append(particle.position[0])
            particle_advection_positions_y.append(particle.position[1])
            particle_advection_positions_z.append(particle.position[2])

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.quiver(particle_advection_positions_x, particle_advection_positions_y, particle_advection_positions_z, 
            particle_advection_values_x,
            particle_advection_values_y,
            particle_advection_values_z,
            length=0.05)
        plt.show()

if __name__ == "__main__":
    main()