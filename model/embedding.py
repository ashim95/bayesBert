import jax
import haiku as hk
import numpy as np
import jax.numpy as jnp

from model.prior import Prior

class Embedding(hk.Module):
    """
    Embeds tokens and positions into an array of shape [n_batch, n_seq, n_hidden].
    Choose between learnable and VI embeddings.
    """
    def __init__(self, config, vocab_size, hidden_size, name):
        """
        Build Embedding from configuration.
        :params config: configuration dictionary.
        """
        super().__init__(name=name)
        self.config = config
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        if self.config["bayesian_emb"]:
            self.embedding = BayesianEmbedding(config, vocab_size=vocab_size, hidden_size=hidden_size)
        else:
            self.embedding = LearnableEmbedding(config, vocab_size=vocab_size, hidden_size=hidden_size)

    def __call__(self, ids, key, training=True):
        """
        Forward pass on the embedding layer.
        :params ids: input of shape (batch, n_seq).
        :params key: random key.
        :params training: if True, apply dropout.
        """
        if self.config["bayesian_emb"]:
            return self.embedding(ids, key, training)
        else:
            return self.embedding(ids, key, training),  0.

class LearnableEmbedding(hk.Module):
    """
    Embeds tokens and positions into an array of shape [n_batch, n_seq, n_hidden].
    """
    def __init__(self, config, vocab_size, hidden_size):
        """
        Build Embedding from configuration.
        :params config: configuration dictionary.
        """
        super().__init__(name="LearnableEmbedding")
        self.config = config
        self.vocab_size=vocab_size
        self.hidden_size = hidden_size
 
    def __call__(self, token_ids, key, training=True):
        """
        Forward pass on the embedding layer.
        :params token_ids: input of shape (batch, n_seq).
        :params key: random key.
        :params training: if True, apply dropout.
        """
        # Reshape ids
        flat_token_ids = jnp.reshape(token_ids, [token_ids.shape[0]*token_ids.shape[1]])
        
        # Embed tokens
        flat_token_embeddings = hk.Embed(
            vocab_size=self.vocab_size,
            embed_dim=self.hidden_size
        )(flat_token_ids)
        
        # After we've embedded our token IDs, we reshape to recover our batch dimension
        embeddings = jnp.reshape(
            flat_token_embeddings, 
            [token_ids.shape[0], token_ids.shape[1], -1]
        )
        
        # Combine our token embeddings with a set of learned positional embeddings
        ### embeddings = token_embeddings + PositionEncoding(self.config)() # Will do outside this

        # Apply layer norm
        ###
        ### embeddings = hk.LayerNorm(
        ###    axis=-1, create_scale=True, create_offset=True
        ### )(embeddings) # Will do outside this function
        
        # # Apply dropout layer during training
        # if training:
        #     embeddings = hk.dropout(
        #         key, 
        #         rate=self.config['embed_dropout_rate'], 
        #         x=embeddings
        #     ) # Will do outside this
        
        return embeddings

class BayesianEmbedding(hk.Module):
    """
    Embeds tokens and positions into an array of shape [n_batch, n_seq, n_hidden].
    Apply Variational Inference to the embedding vectors.
    """
    def __init__(self, config, vocab_size, hidden_size):
        """
        Build Embedding from configuration.
        :params config: configuration dictionary.
        """
        super().__init__(name="BayesianEmbedding")
        self.config = config
        self.vocab_size=vocab_size
        self.hidden_size = hidden_size
        # Posterior weight distribution 
        self.w_mean = hk.get_parameter(
            name="w_mean",
            shape=[self.vocab_size, self.hidden_size],
            init=hk.initializers.TruncatedNormal(),
        )
        self.w_rho = hk.get_parameter(
            name="w_rho",
            shape=[self.vocab_size, self.hidden_size],
            init=hk.initializers.RandomNormal(stddev=1e-2, mean=-7) 
        )
        self.w_sig = jax.nn.softplus(self.w_rho)
        # Prior weight distribution
        prior_distribution = self.config['emb_prior_distribution']
        prior_params = self.config['emb_prior_params']
        self.prior_w = Prior(self.vocab_size*self.hidden_size, prior_distribution, prior_params)
        
    def __call__(self, ids, key, training=True):
        """
        Forward pass on the embeddings.
        :params ids: input of shape (batch, n_seq).
        :params key: random key.
        :params training: if True, apply dropout.
        """
        # Split keys
        keys = jax.random.split(key, num=2)
        
        # Sample embedding matrix from embedding distribution
        eps = jax.random.normal(keys[0], shape=self.w_sig.shape)
        sample_embeddings = self.w_mean + jnp.multiply(self.w_sig, eps)

        # Select embeddings corresponding to input ids, add positional embedding
        # embeddings = jnp.asarray(sample_embeddings)[(ids,)] + PositionEncoding(self.config)()
        # Will do position embeddings outside this
        embeddings = jnp.asarray(sample_embeddings)[(ids,)]

        # Apply layer norm                                          
        # embeddings = hk.LayerNorm(
        #     axis=-1, create_scale=True, create_offset=True,
        # )(embeddings)
        
        # When training apply dropout
        # if training:
        #     embeddings = hk.dropout(
        #         rng=keys[1],
        #         rate=self.config['embed_dropout_rate'],
        #         x=embeddings
        #     )

        kl_div = self.KL_divergence(
            self.w_mean, self.w_sig**2, self.prior_w.params["loc"], self.prior_w.params["scale"]**2
        )

        return embeddings, kl_div
    
    def KL_divergence(self, posterior_mean, posterior_cov, prior_mean, prior_cov):
        """
        Compute KL divergence between the Gaussian posterior and prior distributions.
        :params posterior_mean: mean vector of the posterior.
        :params posterior_cov: vector with the diagonal elements of the covariance matrix.
        :params prior_mean: scalar or mean vector of the prior.
        :params prior_cov: scalar with the value of the variation of the isotropic Gaussian prior.
        """
        eps = 1e-9
        d = np.prod(posterior_mean.shape)
        kl = 0.5*(jnp.sum((posterior_cov+(posterior_mean-prior_mean)**2)/prior_cov - jnp.log(posterior_cov+eps)) - d + d*jnp.log(prior_cov))

        return kl

class PositionEncoding(hk.Module):
    """
    A position encoding of shape [n_seq, n_hidden].
    """
    def __init__(self, config):
        """
        Build positional encoding.
        :params config: configuration dictionary.
        """
        super().__init__(name="PositionEncoding")
        self.config = config

    def __call__(self):
        """
        Returns sinusoidal position encoding.
        """
        positional_encoding = np.zeros((self.config["max_length"], self.config["hidden_size"]), dtype=np.float32)
        position = np.arange(0, self.config["max_length"])[:, np.newaxis]
        scale_factor = -np.log(self.config["max_scale"] / self.config["min_scale"]) / (self.config["hidden_size"] // 2 - 1)
        div_term = self.config["min_scale"] * np.exp(np.arange(0, self.config["hidden_size"] // 2) * scale_factor)
        positional_encoding[:, :self.config["hidden_size"] // 2] = np.sin(position * div_term)
        positional_encoding[:, self.config["hidden_size"] // 2: 2 * (self.config["hidden_size"] // 2)] = np.cos(position * div_term)
        
        return positional_encoding