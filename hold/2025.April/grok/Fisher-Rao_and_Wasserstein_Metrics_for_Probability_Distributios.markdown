# Fisher-Rao and Wasserstein Metrics for Probability Distributions

Information geometry provides a framework to study statistical models and probability distributions using differential geometry. Two key metrics, the **Fisher-Rao metric** and the **Wasserstein metric**, quantify distances between probability distributions. Below, we explain their definitions, calculations, and applications.

## 1. Fisher-Rao Metric

### Overview
The Fisher-Rao metric is a Riemannian metric on the manifold of parametric probability distributions, derived from the Fisher information matrix. It measures the intrinsic geometric distance between distributions in the same parametric family (e.g., two Gaussians).

### Definition
For a parametric family \( p(x|\theta) \), where \( \theta = (\theta_1, \dots, \theta_n) \), the Fisher-Rao metric tensor \( g_{ij}(\theta) \) is the Fisher information matrix:

$$ g_{ij}(\theta) = \mathbb{E}_{p(x|\theta)} \left[ \frac{\partial \log p(x|\theta)}{\partial \theta_i} \frac{\partial \log p(x|\theta)}{\partial \theta_j} \right] $$

or equivalently:

$$ g_{ij}(\theta) = \int p(x|\theta) \frac{\partial \log p(x|\theta)}{\partial \theta_i} \frac{\partial \log p(x|\theta)}{\partial \theta_j} \, dx $$

The geodesic distance between \( p(x|\theta_1) \) and \( p(x|\theta_2) \) is:

$$ d_{FR}(\theta_1, \theta_2) = \inf_{\gamma} \int_0^1 \sqrt{ \sum_{i,j} g_{ij}(\gamma(t)) \dot{\gamma}_i(t) \dot{\gamma}_j(t) } \, dt $$

where \( \gamma(t) \) is a curve connecting \( \theta_1 \) to \( \theta_2 \).

### Calculation
1. Specify the distribution family (e.g., \( \mathcal{N}(\mu, \sigma^2) \)).
2. Compute \( \log p(x|\theta) \).
3. Calculate score functions \( \frac{\partial \log p(x|\theta)}{\partial \theta_i} \).
4. Compute the Fisher information matrix:

$$ g_{ij}(\theta) = \mathbb{E} \left[ \frac{\partial \log p(x|\theta)}{\partial \theta_i} \frac{\partial \log p(x|\theta)}{\partial \theta_j} \right] $$

5. Solve for the geodesic distance (analytically or numerically).

**Example: Univariate Gaussian**
For \( \mathcal{N}(\mu, \sigma^2) \), the density is:

$$ p(x|\mu, \sigma) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{(x-\mu)^2}{2\sigma^2} \right) $$

Log-likelihood:

$$ \log p(x|\mu, \sigma) = -\frac{1}{2} \log (2\pi) - \log \sigma - \frac{(x-\mu)^2}{2\sigma^2} $$

Partial derivatives:

$$ \frac{\partial \log p}{\partial \mu} = \frac{x-\mu}{\sigma^2}, \quad \frac{\partial \log p}{\partial \sigma} = -\frac{1}{\sigma} + \frac{(x-\mu)^2}{\sigma^3} $$

Fisher information matrix:

$$ G(\mu, \sigma) = \begin{pmatrix} \frac{1}{\sigma^2} & 0 \\ 0 & \frac{2}{\sigma^2} \end{pmatrix} $$

For fixed \( \sigma \), the distance between \( \mathcal{N}(\mu_1, \sigma^2) \) and \( \mathcal{N}(\mu_2, \sigma^2) \):

$$ d_{FR}(\mu_1, \mu_2) = \frac{|\mu_1 - \mu_2|}{\sigma} $$

### Applications
- Statistical inference (e.g., maximum likelihood).
- Machine learning (e.g., natural gradient descent).
- Information theory and neuroscience.

### Limitations
- Restricted to parametric families.
- Computationally intensive for complex manifolds.

## 2. Wasserstein Metric

### Overview
The Wasserstein metric (Earth Mover’s Distance) measures the cost of transforming one probability distribution into another via optimal transport. It applies to arbitrary probability measures.

### Definition
For probability measures \( \mu \) and \( \nu \) on a metric space \( (X, d) \), the \( p \)-Wasserstein distance is:

$$ W_p(\mu, \nu) = \left( \inf_{\gamma \in \Gamma(\mu, \nu)} \int_{X \times X} d(x, y)^p \, d\gamma(x, y) \right)^{1/p} $$

where \( \Gamma(\mu, \nu) \) is the set of couplings with marginals \( \mu \) and \( \nu \).

For \( p=2 \):

$$ W_2(\mu, \nu) = \left( \inf_{\gamma \in \Gamma(\mu, \nu)} \int_{X \times X} d(x, y)^2 \, d\gamma(x, y) \right)^{1/2} $$

### Calculation
1. **Discrete Case**:
   For discrete distributions \( \mu = \sum_{i=1}^n a_i \delta_{x_i} \), \( \nu = \sum_{j=1}^m b_j \delta_{y_j} \):

$$ W_p(\mu, \nu) = \left( \min_{T} \sum_{i,j} T_{ij} d(x_i, y_j)^p \right)^{1/p} $$

subject to:

$$ \sum_j T_{ij} = a_i, \quad \sum_i T_{ij} = b_j, \quad T_{ij} \geq 0 $$

2. **1D Continuous Case**:
   For CDFs \( F_\mu \), \( F_\nu \):

$$ W_2(\mu, \nu) = \left( \int_0^1 |F_\mu^{-1}(t) - F_\nu^{-1}(t)|^2 \, dt \right)^{1/2} $$

3. **General Case**: Use numerical methods (e.g., Sinkhorn’s algorithm, sliced Wasserstein).

**Example: 1D Gaussians**
For \( \mu = \mathcal{N}(m_1, \sigma_1^2) \), \( \nu = \mathcal{N}(m_2, \sigma_2^2) \):

$$ W_2(\mu, \nu) = \sqrt{(m_1 - m_2)^2 + (\sigma_1 - \sigma_2)^2} $$

### Applications
- Machine learning (e.g., Wasserstein GANs, domain adaptation).
- Computer vision (e.g., histogram comparison).
- Optimal transport problems.

### Limitations
- Computationally expensive without approximations.
- Sensitive to ground metric choice.

## Comparison

| **Aspect**                | **Fisher-Rao**                         | **Wasserstein**                        |
|---------------------------|---------------------------------------|---------------------------------------|
| **Domain**                | Parametric manifold                   | General probability measures          |
| **Nature**                | Riemannian (intrinsic)                | Metric (optimal transport)            |
| **Applications**          | Inference, optimization               | Generative models, vision             |
| **Computational Cost**    | High for complex cases               | High, mitigated by approximations     |

Use Fisher-Rao for parametric settings and Wasserstein for arbitrary distributions or transport-based problems.