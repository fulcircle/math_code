import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

GRID_SIZE = 100                 # Ocean patch size in meters
NUM_COEFFICIENTS = 100          # Frequency grid resolution (NxN)
WIND_SPEED = 10                 # Wind speed in m/s
WATER_DEPTH = 10                # Water depth in meters
PHILLIPS_CONSTANT = 1           # Amplitude tuning for Phillips spectrum
WIND_DIRECTION = (1, 1)         # Wind direction vector (unnormalized)
WIND_ALIGNMENT_EXPONENT = 15    # How tightly waves align with wind direction
GRAVITY = 9.8                   # Gravitational acceleration in m/s^2
DT = 0.1                        # Time step between frames in seconds
NUM_FRAMES = 200                # Total animation frames


def compute_ocean(
    S=GRID_SIZE,
    N=NUM_COEFFICIENTS,
    V=WIND_SPEED,
    D=WATER_DEPTH,
    A=PHILLIPS_CONSTANT,
    wind=WIND_DIRECTION,
    wind_exp=WIND_ALIGNMENT_EXPONENT,
):
    g = GRAVITY
    L = V**2/g
    w = np.array(wind)
    w_hat = w / np.linalg.norm(w)

    # k_x, k_y
    k = [(2*np.pi*n)/S for n in np.arange(-N//2, N//2)]
    K_x, K_y = np.meshgrid(k, k)

    # Magnitude of k
    K_m = np.sqrt(K_x**2 + K_y**2)
    K_m[K_m == 0] = 1e-10  # avoid division by zero at origin

    # K unit vector at each point
    k_hat_x = K_x / K_m
    k_hat_y = K_y / K_m

    # k_hat dot w_hat
    k_dot_w = k_hat_x * w_hat[0] + k_hat_y * w_hat[1]

    # Phillips spectrum
    P_h_k = A * np.exp(-1/(K_m*L)**2) / K_m**4 * np.abs(k_dot_w)**wind_exp

    # Random gaussian
    e_r = np.random.randn(N, N) # real part
    e_i = np.random.randn(N, N) # imaginary part

    # Initial Coefficients
    h_0_k = (1/np.sqrt(2))*(e_r + e_i*1j)*np.sqrt(P_h_k)

    # Dispersion relation
    o = np.sqrt(g*K_m*np.tanh(K_m*D))

    # Spatial grid
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)

    # Z limits from initial frame
    h_k_0 = h_0_k + np.conj(h_0_k)
    heights_0 = np.fft.ifft2(np.fft.ifftshift(h_k_0)).real
    z_max = np.max(np.abs(heights_0)) * 10

    return X, Y, h_0_k, o, z_max


def compute_heights(h_0_k, o, t):
    h_k_t = h_0_k * np.exp(1j*o*t) + np.conj(h_0_k) * np.exp(-1j*o*t)
    return np.fft.ifft2(np.fft.ifftshift(h_k_t)).real


def ocean_simulation_plotly():
    import plotly.graph_objects as go

    X, Y, h_0_k, o, z_max = compute_ocean()

    # Precompute frames
    frames = []
    for i in range(NUM_FRAMES):
        heights = compute_heights(h_0_k, o, i * DT)
        frames.append(go.Frame(data=[go.Surface(z=heights, x=X, y=Y, colorscale="Blues", showscale=False)], name=str(i)))

    fig = go.Figure(
        data=[frames[0].data[0]],
        frames=frames,
        layout=go.Layout(
            title="Ocean Wave",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                zaxis=dict(range=[-z_max, z_max]),
            ),
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(label="Play", method="animate", args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True)]),
                    dict(label="Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
                ],
            )],
        ),
    )

    fig.show()

def ocean_simulation():
    X, Y, h_0_k, o, z_max = compute_ocean()

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    manager = plt.get_current_fig_manager()
    manager.window.wm_geometry("+300+100")

    def update(frame):
        ax.clear()
        heights = compute_heights(h_0_k, o, frame * DT)
        ax.plot_surface(X, Y, heights, cmap="Blues", edgecolor="none")
        ax.set_title("Ocean Wave")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_zlim(-z_max, z_max)

    ani = FuncAnimation(fig, update, frames=NUM_FRAMES, interval=50)
    plt.show()


if __name__ == "__main__":
    ocean_simulation_plotly()
