import jax
import haiku as hk
import numpy as np
import jax.numpy as jnp
import jax.scipy as jsp

from model.prior import Prior

class LinearLayer(hk.Module):
    """
    General Linear Layer : choose between VI linear or normal linear.
    """
    def __init__(self, input_size, output_size, name, bayesian=False, 
                prior_distribution=None, prior_params=None, with_bias=True):
        """
        Build General Linear Layer.
        :params input_size: Input size of the linear layer.
        :params output_size: Output size of the linear layer.
        :params bayesian: Set to True in order to perform variational inference on the layer weights else False.
        :params prior_distribution: Prior to be used if Bayesian is True.
        :params prior_params: Parameters of the prior.
        :params with_bias: True to include bias in linear layer.
        """
        super().__init__(name=name)
        self.bayesian = bayesian
        if self.bayesian:
            if not prior_distribution or not prior_params:
                raise ValueError("Must specify prior when using VI.")
            self.linear = BayesianLinear(
                input_size=input_size, 
                output_size=output_size, 
                prior_distribution=prior_distribution, 
                prior_params=prior_params, 
                with_bias=with_bias,
                name="BayesianLinear",
            )
        else:
            self.linear = hk.Linear(
                output_size=output_size,
                w_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform"),
                b_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform"), 
                with_bias=with_bias,
                name="StandardLinear"
            )

    def __call__(self, x, key, kl_mc_samples):
        """
        Forward pass on the General Linear Layer.
        :params x: Inputs of shape (batch, n_seq)
        :params key: Random key for forward pass. 
        :params kl_mc_samples: Number of Monte Carlo posterior samples to compute the KL divergence.
        """
        if self.bayesian:
            return self.linear(x, key, kl_mc_samples) 
        else:
            return self.linear(x), 0.


class BayesianLinear(hk.Module):
    """
    Linear layer with Mean-Field Gaussian Variational Inference.
    """
    def __init__(self, input_size, output_size, prior_distribution, prior_params, with_bias=True, name="BayesianLinear"):
        """
        Build Linear layer.
        :params input_size: Input size of the linear layer.
        :params output_size: Output size of the linear layer.
        :params prior_distribution: Prior to be used if Bayesian is True.
        :params prior_params: Parameters of the prior.
        :params with_bias: True to include bias in linear layer.
        """
        super().__init__(name=name)
        # Weight dimension
        self.input_size = input_size
        self.output_size = output_size
        # Posterior weight distribution 
        self.posterior_w_mean = hk.get_parameter(
            name="posterior_w_mean",
            shape=[self.input_size, self.output_size],
            init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform"),
        )
        self.posterior_w_rho = hk.get_parameter(
            name="posterior_w_rho",
            shape=[self.input_size, self.output_size],
            init=hk.initializers.RandomNormal(stddev=1e-2, mean=-7) 
        )
        self.posterior_w_sig = jax.nn.softplus(self.posterior_w_rho)
        # Prior distribution on weights
        self.prior_w = Prior(input_size*output_size, prior_distribution, prior_params)
        # Bias
        self.bias = with_bias
        if self.bias:
            # Posterior bias distribution 
            self.posterior_b_mean = hk.get_parameter(
                name="posterior_b_mean",
                shape=[self.output_size],
                init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform"),
            )
            self.posterior_b_rho = hk.get_parameter(
                name="posterior_b_rho",
                shape=[self.output_size],
                init=hk.initializers.RandomNormal(stddev=1e-2, mean=-7)
            )
            self.posterior_b_sig = jax.nn.softplus(self.posterior_b_rho)
            # Prior bias distribution
            self.prior_b = Prior(output_size, prior_distribution, prior_params)

    def __call__(self, x, key, kl_mc_samples):
        """
        Forward pass on the linear layer.
        :params x: inputs of shape (batch, n_seq, n_hidden).
        :params key: random key.
        :params kl_mc_samples: Number of Monte Carlo posterior samples to compute the KL divergence.
        """
        # Split keys
        key1, key2 = jax.random.split(key, 2)

        # Local reparameterization trick
        logits_mean = jnp.einsum("bni,io->bno", x, self.posterior_w_mean)
        logits_var = jnp.einsum("bni,io->bno", x**2, self.posterior_w_sig**2)    
        
        if self.bias:
            # Add mean of bias
            logits_mean += self.posterior_b_mean
            # Add variance of bias
            logits_var += self.posterior_b_sig**2
            
        # Sample from standard normal 
        eps = jax.random.normal(key1, shape=logits_var.shape)

        # Sample from activations
        logits_sig = jnp.sqrt(logits_var)    
        logits_sample = logits_mean + jnp.multiply(logits_sig, eps)

        # Compute KL divergence
        if self.prior_w.distribution == "gaussian":
            kl_div = self.KL_divergence() 
        else:
            kl_div = self.MC_KL_divergence(key2, kl_mc_samples)

        return logits_sample, kl_div

    def _sample_gaussian_posterior(self, key):
        """
        Sample from Gaussian posterior distribution.
        :params key: random key.
        """
        # Split keys
        key1, key2 = jax.random.split(key, 2)

        # Sample from standard normal 
        eps_w = jax.random.normal(key1, shape=self.posterior_w_sig.shape)

        # Sample from posterior weight
        posterior_w_sample = self.posterior_w_mean + self.posterior_w_sig * eps_w

        if self.bias:
            # Sample from standard normal 
            eps_b = jax.random.normal(key2, shape=self.posterior_b_sig.shape)

            # Sample from posterior bias
            posterior_b_sample = self.posterior_b_mean + self.posterior_b_sig*eps_b

            return posterior_w_sample.flatten(), posterior_b_sample.flatten()
        
        return posterior_w_sample.flatten(), None

    def MC_KL_divergence(self, key, mc_samples):
        """
        Compute KL divergence by Monte Carlo sampling.
        This function vectorizes the computation.
        :params key: random key.
        :params mc_samples: number of posterior samples.
        """
        # Split keys
        keys = jnp.array(jax.random.split(key, mc_samples))
        
        kl_div = jax.vmap(self._MC_KL_divergence)(keys).mean()
        
        return kl_div

    def _MC_KL_divergence(self, key):
        """
        Compute KL divergence by Monte Carlo sampling.
        :params key: random key.
        """
        # Sample from posterior weight and bias
        posterior_w_sample, posterior_b_sample = self._sample_gaussian_posterior(key)

        # Compute log posterior weight probability
        log_posterior_w = jsp.stats.norm.logpdf(
            posterior_w_sample, loc=self.posterior_w_mean.flatten(), scale=self.posterior_w_sig.flatten()
        ).sum()
        
        # Compute log prior weight probability
        log_prior_w = self.prior_w.log_pdf(posterior_w_sample).sum()

        log_posterior_b, log_prior_b = 0, 0
        if self.bias:
            # Compute log posterior bias probability
            log_posterior_b = jsp.stats.norm.logpdf(
                posterior_b_sample, loc=self.posterior_b_mean.flatten(), scale=self.posterior_b_sig.flatten()
            ).sum()
           
            # Compute log prior bias probability
            log_prior_b = self.prior_b.log_pdf(posterior_b_sample).sum()

        # Compute KL estimate
        kl_div = log_posterior_w + log_posterior_b - log_prior_w - log_prior_b

        return kl_div.sum()

    def KL_divergence(self):
        """
        KL divergence between a fully factorized posterior and isotropic prior.
        """
        # compute KL divergence between weight posterior and prior
        kl_div = self._KL_divergence_Gaussians(
            self.posterior_w_mean.flatten(), 
            jnp.square(self.posterior_w_sig).flatten(), 
            self.prior_w.params["loc"], 
            self.prior_w.params["scale"]**2
        )

        if self.bias:
            # Compute KL divergence between bias posterior and prior
            kl_div += self._KL_divergence_Gaussians(
                self.posterior_b_mean.flatten(), 
                jnp.square(self.posterior_b_sig).flatten(),
                self.prior_b.params["loc"], 
                self.prior_b.params["scale"]**2
            )

        return kl_div

    def _KL_divergence_Gaussians(self, posterior_mean, posterior_cov, prior_mean, prior_cov):
        """
        Compute KL divergence between two Gaussians using the analytical formulation.
        :params posterior_mean: mean vector of the posterior
        :params posterior_cov: vector with the diagonal elements of the covariance matrix
        :params prior_mean: scalar or mean vector of the prior
        :params prior_cov: scalar with the value of the variation of the isotropic Gaussian prior
        """
        eps = 1e-9
        d = np.prod(posterior_mean.shape)

        kl = 0.5*(jnp.sum((posterior_cov+(posterior_mean-prior_mean)**2)/prior_cov - jnp.log(posterior_cov+eps)) - d + d*jnp.log(prior_cov))

        return kl
