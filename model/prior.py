import haiku as hk
from jax._src.random import gamma
import jax.numpy as jnp
import jax.scipy as jsp

class Prior(hk.Module):
    """
    Container for Laplace, Gaussian, Cauchy,
    Student-t or Mixture of Gaussians isotropic prior distributions.
    """
    def __init__(self, dim, distribution, params):
        """
        Build isotropic prior distribution.
        :params dim: dimension of the distribution.
        :params distribution: name of the prior distribution.
        :params params: parameters of the prior distribution.
        """
        if distribution not in ["gaussian", "laplace", "cauchy", "t", "mixture_gaussian", "logistic"]:
            raise Exception("Prior should be chosen among gaussian, laplace, cauchy, t or mixture_gaussian")
        super().__init__(distribution+"_prior")
        self.dim = dim
        self.distribution = distribution
        self.params = params

    @hk.experimental.name_like("__call__")
    def log_pdf(self, x):
        """
        Compute log-pdf of the prior given input x.
        :params x: input x.
        """
        x = x.flatten()
        
        if self.distribution == "gaussian":
            log_prior = jsp.stats.norm.logpdf(
                x, loc=self.params["loc"], scale=self.params["scale"]
            ).sum()
        elif self.distribution == "laplace":
            log_prior = jsp.stats.laplace.logpdf(
                x, loc=self.params["loc"], scale=self.params["scale"]
            ).sum()
        elif self.distribution == "cauchy":
            log_prior = jsp.stats.cauchy.logpdf(
                x, loc=self.params["loc"], scale=self.params["scale"]
            ).sum()
        elif self.distribution == "t":
            log_prior = jsp.stats.t.logpdf(
                x, df=self.params["df"], loc=self.params["loc"], scale=self.params["scale"]
            ).sum()
        elif self.distribution == "logistic":
            log_prior = jnp.sum(
                jsp.stats.logistic.logpdf((x-self.params["loc"])/self.params["scale"]) \
                    - jnp.log(self.params["scale"])
            )
        elif self.distribution == "mixture_gaussian":
            prior_1 = jsp.stats.norm.pdf(x, loc=self.params["loc"], scale=self.params["scale_1"])
            prior_2 = jsp.stats.norm.pdf(x, loc=self.params["loc"], scale=self.params["scale_2"])
            log_prior = jnp.log(self.params["pi"] * prior_1 + (1-self.params["pi"]) * prior_2).sum()
        
        return log_prior
