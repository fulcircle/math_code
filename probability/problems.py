import math
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import expon, pareto

# ----------------- FUNCTIONS -----------------

def exp_inv_cdf(u, lam):
    """
    Inverse CDF for Exponential Distribution.
    x = -ln(1 - u) / lambda
    """
    return -math.log(1.0 - u) / lam

def power_law_inv_cdf(u, alpha):
    """
    Inverse CDF for Power Law / Pareto Distribution.
    y = (1 - u) ** (1 / (1 - alpha))
    """
    return (1.0 - u) ** (1.0 / (1.0 - alpha))

def sample_household_wealth(p, alpha):
    """
    Samples the wealth of an American household using the mixture model:
    - Proportion p have 0 wealth.
    - Proportion (1 - p) follow a power law distribution starting at x >= 1.
    """
    u = random.random()
    if u < p:
        return 0.0
    else:
        return ((1.0 - u) / (1.0 - p)) ** (1.0 / (1.0 - alpha))


# ----------------- PARAMETERS & SIMULATION -----------------

n_trials = 1000

# Problem 1 Parameters (Exponential)
lam = 1.5
scale = 1.0 / lam

# Problem 2 Parameters (Pareto/Power Law)
alpha = 2.5
b = alpha - 1.0

# Problem 3 Parameters (Household Wealth Mixture)
p_wealth = 0.25

# Generate samples for Problem 1
u_samples_1 = [random.random() for _ in range(n_trials)]
our_exp_samples = [exp_inv_cdf(u, lam) for u in u_samples_1]
scipy_exp_samples = expon.rvs(scale=scale, size=n_trials)

# Generate samples for Problem 2
u_samples_2 = [random.random() for _ in range(n_trials)]
our_pareto_samples = [power_law_inv_cdf(u, alpha) for u in u_samples_2]
pareto_dist = pareto(b=b)
scipy_pareto_samples = pareto_dist.rvs(size=n_trials)

# Generate samples for Household Wealth (Problem 3)
our_household_samples = [sample_household_wealth(p_wealth, alpha) for _ in range(n_trials)]
scipy_household_samples = [0.0 if random.random() < p_wealth else pareto_dist.rvs() for _ in range(n_trials)]


# ----------------- MULTI-PLOT VISUALIZATION -----------------

fig, axs = plt.subplots(2, 2, figsize=(16, 12))
((ax1, ax2), (ax3, ax4)) = axs

# --- Plot 1: Exponential Distribution ---
ax1.hist(our_exp_samples, bins=50, density=True, alpha=0.5, color='skyblue', edgecolor='blue', label='Custom Inverse CDF')
ax1.hist(scipy_exp_samples, bins=50, density=True, alpha=0.5, color='orange', edgecolor='red', label='scipy.stats.expon.rvs')

# Theoretical curve
x_vals_exp = np.linspace(0, max(max(our_exp_samples), max(scipy_exp_samples)), 500)
y_theoretical_exp = expon.pdf(x_vals_exp, scale=scale)
ax1.plot(x_vals_exp, y_theoretical_exp, 'g-', lw=2, label=f'Theoretical PDF (λ={lam})')

ax1.set_title('Problem 1: Exponential Distribution')
ax1.set_xlabel('Value')
ax1.set_ylabel('Density')
ax1.legend()
ax1.grid(True, alpha=0.3)


# --- Plot 2: Pareto/Power Law Distribution ---
max_val_pareto = min(max(max(our_pareto_samples), max(scipy_pareto_samples)), 15.0)

ax2.hist(our_pareto_samples, bins=np.linspace(1, max_val_pareto, 50), density=True, alpha=0.5, color='skyblue', edgecolor='blue', label='Custom Inverse CDF')
ax2.hist(scipy_pareto_samples, bins=np.linspace(1, max_val_pareto, 50), density=True, alpha=0.5, color='orange', edgecolor='red', label='scipy.stats.pareto.rvs')

# Theoretical curve
x_vals_pareto = np.linspace(1, max_val_pareto, 500)
y_theoretical_pareto = pareto_dist.pdf(x_vals_pareto)
ax2.plot(x_vals_pareto, y_theoretical_pareto, 'g-', lw=2, label=f'Theoretical PDF (b=α-1={alpha-1:.1f})')

ax2.set_title('Problem 2: Pareto / Power Law Distribution')
ax2.set_xlabel('Value')
ax2.set_ylabel('Density')
ax2.set_xlim(1, max_val_pareto)
ax2.legend()
ax2.grid(True, alpha=0.3)


# --- Plot 3: Household Zero-Wealth Proportions ---
our_zeros = sum(1 for w in our_household_samples if w == 0.0) / n_trials
scipy_zeros = sum(1 for w in scipy_household_samples if w == 0.0) / n_trials
categories = ['Target Zero', 'Custom Zero', 'SciPy Zero']
proportions = [p_wealth, our_zeros, scipy_zeros]
colors = ['lightgray', 'skyblue', 'orange']

ax3.bar(categories, proportions, color=colors, edgecolor='black', alpha=0.8, width=0.5)
ax3.set_title(f'Household Zero-Wealth Proportion (Target={p_wealth:.2%})')
ax3.set_ylabel('Proportion of Population')
ax3.set_ylim(0, 1.0)
for i, val in enumerate(proportions):
    ax3.text(i, val + 0.02, f'{val:.2%}', ha='center', fontweight='bold')
ax3.grid(True, axis='y', alpha=0.3)


# --- Plot 4: Non-Zero Household Wealth Distribution ---
our_nonzero = [w for w in our_household_samples if w > 0.0]
scipy_nonzero = [w for w in scipy_household_samples if w > 0.0]
max_val_wealth = min(max(max(our_nonzero), max(scipy_nonzero)), 15.0)

ax4.hist(our_nonzero, bins=np.linspace(1, max_val_wealth, 50), density=True, alpha=0.5, color='skyblue', edgecolor='blue', label='Custom Inverse CDF')
ax4.hist(scipy_nonzero, bins=np.linspace(1, max_val_wealth, 50), density=True, alpha=0.5, color='orange', edgecolor='red', label='SciPy Mixed rvs')

# Theoretical curve for the Pareto portion of the mixture model
x_vals_wealth = np.linspace(1, max_val_wealth, 500)
# Scaled by (1 - p) as non-zeros make up (1 - p) of the population
y_theoretical_wealth = (1 - p_wealth) * pareto_dist.pdf(x_vals_wealth)
ax4.plot(x_vals_wealth, y_theoretical_wealth, 'g-', lw=2.5, label=f'Theoretical PDF: (1-p)*Pareto(b={b:.1f})')

ax4.set_title('Household Wealth Distribution (Non-Zero Tail)')
ax4.set_xlabel('Value')
ax4.set_ylabel('Density')
ax4.set_xlim(1, max_val_wealth)
ax4.legend()
ax4.grid(True, alpha=0.3)


plt.tight_layout()
plt.show()
