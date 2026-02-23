import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def ocean_simulation():

    S = 100  # Ocean grid size (m)
    N = 50 # Num coefficients
    V = 10 # Wind speed (m/s)
    g = 9.8 # Gravitational acceleration (m/s)
    A = .001 # Tuning constant for Phillips spectrum
    L = V**2/g # Used for computing Phillips spectrum
    w = np.array([1, 0]) # wind direction vector
    w_hat = w / np.linalg.norm(w)

    # k_x, k_y
    k = [(2*np.pi*n)/S for n in np.arange(-N//2, N//2)]
    K_x, K_y = np.meshgrid(k, k) # frequencies in x, y

    # Magnitude of k
    K_m = np.sqrt(K_x**2 + K_y**2)
    K_m[K_m == 0] = 1e-10  # avoid division by zero at origin

    # K unit vector at each point
    k_hat_x = K_x / K_m
    k_hat_y = K_y / K_m

    # k_hat dot w_hat
    # w_hat[0] = x component
    # w_hat[1] = y_component
    k_dot_w = k_hat_x * w_hat[0] + k_hat_y * w_hat[1]

    # Phillips spectrum
    P_h_k = A * np.exp(-1/(K_m*L)**2) / K_m**4 * np.abs(k_dot_w)

    # Random gaussian real part
    e_r = np.random.randn(N, N)
    # Random gaussian imaginary part
    e_i = np.random.randn(N, N)

    # Initial Coefficients
    h_0_k = (1/np.sqrt(2))*(e_r + e_i*1j)*np.sqrt(P_h_k)

    # Dispersion relation
    o = np.sqrt(g*K_m)

    # Spatial grid
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)

    # Plot / window configuration
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    manager = plt.get_current_fig_manager()
    manager.window.wm_geometry("+300+100")

    dt = 0.1

    # Compute initial frame to determine z range
    h_k_0 = h_0_k + np.conj(h_0_k)
    heights_0 = np.fft.ifft2(np.fft.ifftshift(h_k_0)).real
    z_max = np.max(np.abs(heights_0)) * 1.5
    z_lim = (-z_max, z_max)

    def update(frame):
        ax.clear()
        t = frame * dt

        # Time evolution of coefficients
        h_k_t = h_0_k * np.exp(1j*o*t) + np.conj(h_0_k) * np.exp(-1j*o*t)

        # Map back to spatial domain
        heights = np.fft.ifft2(np.fft.ifftshift(h_k_t)).real

        ax.plot_surface(X, Y, heights, cmap="ocean", edgecolor="none")
        ax.set_title("Ocean Wave")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_zlim(z_lim)

    ani = FuncAnimation(fig, update, frames=200, interval=50)
    plt.show()


if __name__ == "__main__":
    ocean_simulation()
