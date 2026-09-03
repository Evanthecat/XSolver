import numpy as np
import matplotlib.pyplot as plt
from interpolation import evaluate_2nd_degree_interpolation, generate_knot_set
from interpolation_pseudoinverse import Particles, calculate_pseudoinverse, unpack_grid, pack_grid
from operators import generate_galerkin_hodge_star, generate_d_0, generate_d_1
import math

# Page 20 Section 6.1.3 (83)
def evaluate_pressure_projection(f_1, f_2, f_3, galerkin_hodge_star, d_0, d_1):
    A = d_1.T @ galerkin_hodge_star @ d_1 + d_0 @ d_0.T
    b = d_1.T @ galerkin_hodge_star @ np.concatenate([f_1.flatten(), f_2.flatten(), f_3.flatten()])

    x_i = 0
    r_i = b
    d_i = r_i

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

    f_new = d_1 @ x_k
    tau = f_new - np.concatenate([unpack_grid(f_1), unpack_grid(f_2), unpack_grid(f_3)])

    return tau[:f_1.size], tau[f_1.size:f_1.size + f_2.size], tau[f_1.size + f_2.size:]

def main():
    p = 3
    particle_grid_resolution = 10
    particles = []

    torus_radius = 0.35
    tube_radius = 0.15
    torus_x = 0.5
    torus_y = 0.5
    torus_z = 0.5

    N_x = 5
    N_y = 5
    N_z = 5

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
    
    galerkin_hodge_star = generate_galerkin_hodge_star(N_x, N_y, N_z, p)
    d_0 = generate_d_0(N_x, N_y, N_z)
    d_1 = generate_d_1(N_x, N_y, N_z)
    
    tau_x, tau_y, tau_z = evaluate_pressure_projection(f_x, f_y, f_z, galerkin_hodge_star, d_0, d_1)

    f_x += pack_grid(tau_x, N_x, N_y - 1, N_z - 1)
    f_y += pack_grid(tau_y, N_x - 1, N_y, N_z - 1)
    f_z += pack_grid(tau_z, N_x - 1, N_y - 1, N_z)

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

    with open("div_free_velocity_field.txt", 'w') as file:
        file.write(f"{N_x}\n\n")
        file.write(f"{N_y}\n\n")
        file.write(f"{N_z}\n\n")

        for coefficient in unpack_grid(f_x):
            file.write(f"{coefficient}\n")

        file.write(f"\n")

        for coefficient in unpack_grid(f_y):
            file.write(f"{coefficient}\n")

        file.write(f"\n")

        for coefficient in unpack_grid(f_z):
            file.write(f"{coefficient}\n")

if __name__ == "__main__":
    main()