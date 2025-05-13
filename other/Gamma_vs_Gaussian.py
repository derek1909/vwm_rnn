import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, poisson

# Parameters
r = 20                 # your rate parameter
dt = 0.01          # Δt

k = 0.01                 # choose k
M = 1.0 / k**2          # M = 1/k^2
lam = dt/k**2

# X-axis values for discrete and continuous distributions
x_values = torch.arange(0, r*4, 1)
x_values_M = torch.arange(0, r*4*M, M)
x_dense = torch.linspace(0, r*4, 500)

# Poisson(λ) PMF via SciPy for comparison
pmf_poisson = poisson.pmf(x_values.numpy(), mu=r)

# (1/M) * Poisson(Mλ) via torch
pmf_poisson_M = poisson.pmf(x_values_M.numpy()*dt, mu=M*r*dt)

# Normal approximation (using SciPy)
sigma = np.sqrt(r/(dt*M))
mass_at_zero = norm.cdf(0, loc=r, scale=sigma)
normal_pdf = norm.pdf(x_dense.numpy(), loc=r, scale=sigma)
normal_pdf[0] = mass_at_zero

# Gamma distribution via torch
gamma_dist = torch.distributions.Gamma(concentration=r*lam, rate=lam)
gamma_pdf = torch.exp(gamma_dist.log_prob(x_dense))

# Plot
plt.figure(figsize=(5, 4))

# Poisson(λ)
# plt.stem(x_values.numpy(), pmf_poisson*dt, linefmt='b-', markerfmt='bo',
#          basefmt=' ', label='Poisson($\\lambda$)')

# Poisson(Mλ)/M
plt.stem(x_values.numpy(), pmf_poisson_M*(M*dt),
         basefmt=' ', label='Poisson($\\lambda M$)/M')

# Normal approximation
plt.plot(x_dense.numpy(), normal_pdf, 'r--', linewidth=2,
         label='Gaussian')

# Gamma approximation
plt.plot(x_dense.numpy(), gamma_pdf.numpy(), 'g-.', linewidth=2,
         label=f'Gamma')


# Formatting
# plt.title(f'r={r} Hz, k={k}')
plt.xlabel('corrupted rate (Hz)')
plt.ylabel('Probability / Density')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"/homes/jd976/working/vwm_rnn/other/gamma_vs_gauss/gammavsgauss_r={r}_k={k}.png")
plt.close()
