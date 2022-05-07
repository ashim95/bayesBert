# coding=utf-8
# Copyright 2021 The Google Flax Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import (Any, Callable, Iterable, List, Optional, Sequence, Tuple,
                    Union)
import abc
import dataclasses

import numpy as np

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from flax.linen.attention import dot_product_attention_weights
from jax import lax

from jax._src.random import gamma
import jax.scipy as jsp
from flax.linen.initializers import lecun_normal
from flax.linen.initializers import variance_scaling
from flax.linen.initializers import normal
from flax.linen.initializers import zeros

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Array = Any
PrecisionLike = Union[None, str, lax.Precision, Tuple[str, str],
                      Tuple[lax.Precision, lax.Precision]]
default_kernel_init = variance_scaling(1.0, 'fan_avg', 'uniform')

# from ...modeling_flax_outputs import (
#     FlaxBaseModelOutput,
#     FlaxBaseModelOutputWithPooling,
#     FlaxMaskedLMOutput,
#     FlaxMultipleChoiceModelOutput,
#     FlaxNextSentencePredictorOutput,
#     FlaxQuestionAnsweringModelOutput,
#     FlaxSequenceClassifierOutput,
#     FlaxTokenClassifierOutput,
# )
# from ...modeling_flax_utils import (
#     ACT2FN,
#     FlaxPreTrainedModel,
#     append_call_sample_docstring,
#     append_replace_return_docstrings,
#     overwrite_call_docstring,
# )
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_bert import BertConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "bert-base-uncased"
_CONFIG_FOR_DOC = "BertConfig"
_TOKENIZER_FOR_DOC = "BertTokenizer"


BERT_START_DOCSTRING = r"""

    This model inherits from [`FlaxPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading, saving and converting weights from PyTorch models)

    This model is also a Flax Linen [flax.linen.Module](https://flax.readthedocs.io/en/latest/flax.linen.html#module)
    subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to
    general usage and behavior.

    Finally, this model supports inherent JAX features such as:

    - [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
    - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
    - [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
    - [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

    Parameters:
        config ([`BertConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~FlaxPreTrainedModel.from_pretrained`] method to load the model weights.
        dtype (`jax.numpy.dtype`, *optional*, defaults to `jax.numpy.float32`):
            The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and
            `jax.numpy.bfloat16` (on TPUs).

            This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If
            specified all the computation will be performed with the given `dtype`.

            **Note that this only specifies the dtype of the computation and does not influence the dtype of model
            parameters.**

            If you wish to change the dtype of the model parameters, see [`~FlaxPreTrainedModel.to_fp16`] and
            [`~FlaxPreTrainedModel.to_bf16`].
        dtype (`jax.numpy.dtype`, *optional*, defaults to `jax.numpy.float32`):
            The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and
            `jax.numpy.bfloat16` (on TPUs).

            This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If
            specified all the computation will be performed with the given `dtype`.

            **Note that this only specifies the dtype of the computation and does not influence the dtype of model
            parameters.**

            If you wish to change the dtype of the model parameters, see [`~FlaxPreTrainedModel.to_fp16`] and
            [`~FlaxPreTrainedModel.to_bf16`].

"""

BERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`numpy.ndarray` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`BertTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`numpy.ndarray` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`numpy.ndarray` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`numpy.ndarray` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.
        head_mask (`numpy.ndarray` of shape `({0})`, `optional):
            Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

"""

class BayesBertConfig(BertConfig):

    model_type = "bert"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        bayesian_classifier=False,
        classifer_prior="gaussian",
        classifier_prior_params = {"loc":0, "scale":1e-2},
        **kwargs
    ):
        super().__init__(vocab_size=30522,
                        hidden_size=768,
                        num_hidden_layers=12,
                        num_attention_heads=12,
                        intermediate_size=3072,
                        hidden_act="gelu",
                        hidden_dropout_prob=0.1,
                        attention_probs_dropout_prob=0.1,
                        max_position_embeddings=512,
                        type_vocab_size=2,
                        initializer_range=0.02,
                        layer_norm_eps=1e-12,
                        pad_token_id=0,
                        position_embedding_type="absolute",
                        use_cache=True,
                        classifier_dropout=None, 
                        **kwargs)

        self.bayesian_classifer = bayesian_classifer
        self.classifer_prior = classifer_prior
        self.classifier_prior_params = classifier_prior_params

class Prior(nn.Module):
    """
    Container for Laplace, Gaussian, Cauchy,
    Student-t or Mixture of Gaussians isotropic prior distributions.
    """
    config: BertConfig
    distribution: str
    params: dict

    @nn.compact
    def __call__(self, x):
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


class FlaxBayesLinear(nn.Module):
    """Linear layer with Mean-Field Gaussian Variational Inference."""

    # config: BertConfig
    features: int
    prior_distribution: str
    prior_params: dict
    use_bias: bool = True
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32
    precision: PrecisionLike = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    rho_init:Callable[[PRNGKey, Shape, Dtype], Array] = normal(stddev=1e-2)
    # bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = normal(stddev=1e-2)


    @nn.compact
    def __call__(self, inputs, key, kl_mc_samples):
    
        # kernel = self.param('kernel',
        #                 self.kernel_init,
        #                 (inputs.shape[-1], self.features),
        #                 self.param_dtype)
        # kernel = jnp.asarray(kernel, self.dtype)
        # y = lax.dot_general(inputs, kernel,
        #                     (((inputs.ndim - 1,), (0,)), ((), ())),
        #                     precision=self.precision)
        # if self.use_bias:
        # bias = self.param('bias', self.bias_init, (self.features,),
        #                     self.param_dtype)
        # bias = jnp.asarray(bias, self.dtype)
        # y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
        
        # Posterior weight distribution
        posterior_w_mean = self.param(
                                "posterior_w_mean",
                                self.kernel_init,
                                (inputs.shape[-1], self.features,),
                                self.param_dtype
                                )
        
        posterior_w_rho = self.param(
                                "posterior_w_rho",
                                self.rho_init,
                                (inputs.shape[-1], self.features,),
                                self.param_dtype
                                )


        posterior_w_sig = jax.nn.softplus(self.posterior_w_rho)
        # Prior distribution on weights
        prior_w = Prior(self.prior_distribution, self.prior_params)
        # Bias
        if self.use_bias:
            # Posterior bias distribution 
            posterior_b_mean = self.param(
                "posterior_b_mean",
                self.kernel_init,
                self.features,
                self.param_dtype
            )
            posterior_b_rho = self.param(
                "posterior_b_rho",
                self.rho_init,
                self.output_size,
                self.param_dtype
            )
            posterior_b_sig = jax.nn.softplus(self.posterior_b_rho)
            # Prior bias distribution
            prior_b = Prior(self.prior_distribution, self.prior_params)  
        else:
            posterior_b_mean = None
            posterior_b_sig = None
            prior_b = None   
        
        # Split keys
        key1, key2 = jax.random.split(key, 2)

        # Local reparameterization trick
        # logits_mean = jnp.einsum("bni,io->bno", x, self.posterior_w_mean)
        # logits_var = jnp.einsum("bni,io->bno", x**2, self.posterior_w_sig**2)    
        logits_mean = lax.dot_general(inputs, posterior_w_mean,
                            (((inputs.ndim - 1,), (0,)), ((), ())),
                            precision=self.precision)
        logits_var = lax.dot_general(inputs**2, posterior_w_sig**2,
                            (((inputs.ndim - 1,), (0,)), ((), ())),
                            precision=self.precision)
   
        if self.use_bias:
            # Add mean of bias
            logits_mean += posterior_b_mean
            # Add variance of bias
            logits_var += posterior_b_sig**2
            
        # Sample from standard normal 
        eps = jax.random.normal(key1, shape=logits_var.shape)

        # Sample from activations
        logits_sig = jnp.sqrt(logits_var)    
        logits_sample = logits_mean + jnp.multiply(logits_sig, eps)

        # Compute KL divergence
        if prior_w.distribution == "gaussian":
            kl_div = self.KL_divergence(posterior_w_mean, posterior_w_sig, prior_w, 
                                        posterior_b_mean, posterior_b_sig, prior_b) 
        else:
            kl_div = self.MC_KL_divergence(key2, kl_mc_samples, 
                                            posterior_w_mean, posterior_w_sig, prior_w, 
                                            posterior_b_mean, posterior_b_sig, prior_b)

        return logits_sample, kl_div

    def _sample_gaussian_posterior(self, key, posterior_w_mean, 
                                    posterior_w_sig, posterior_b_mean,
                                    posterior_b_sig, 
                                    ):
        """
        Sample from Gaussian posterior distribution.
        :params key: random key.
        """
        # Split keys
        key1, key2 = jax.random.split(key, 2)

        # Sample from standard normal 
        eps_w = jax.random.normal(key1, shape=posterior_w_sig.shape)

        # Sample from posterior weight
        posterior_w_sample = posterior_w_mean + posterior_w_sig * eps_w

        if self.use_bias:
            # Sample from standard normal 
            eps_b = jax.random.normal(key2, shape=posterior_b_sig.shape)

            # Sample from posterior bias
            posterior_b_sample = posterior_b_mean + posterior_b_sig*eps_b

            return posterior_w_sample.flatten(), posterior_b_sample.flatten()
        
        return posterior_w_sample.flatten(), None

    def MC_KL_divergence(self, key, mc_samples, 
                        posterior_w_mean, posterior_w_sig, prior_w, 
                        posterior_b_mean, posterior_b_sig, prior_b):
        """
        Compute KL divergence by Monte Carlo sampling.
        This function vectorizes the computation.
        :params key: random key.
        :params mc_samples: number of posterior samples.
        """
        # Split keys
        keys = jnp.array(jax.random.split(key, mc_samples))
        
        kl_div = jax.vmap(self._MC_KL_divergence)(keys, 
                                                posterior_w_mean, posterior_w_sig, prior_w, 
                                                posterior_b_mean, posterior_b_sig, prior_b).mean()
        
        return kl_div

    def _MC_KL_divergence(self, key, 
                            posterior_w_mean, posterior_w_sig, prior_w, 
                            posterior_b_mean, posterior_b_sig, prior_b
                                ):
        """
        Compute KL divergence by Monte Carlo sampling.
        :params key: random key.
        """
        # Sample from posterior weight and bias
        posterior_w_sample, posterior_b_sample = self._sample_gaussian_posterior(key, posterior_w_mean, 
                                                                            posterior_w_sig, posterior_b_mean,
                                                                            posterior_b_sig)

        # Compute log posterior weight probability
        log_posterior_w = jsp.stats.norm.logpdf(
            posterior_w_sample, loc=posterior_w_mean.flatten(), scale=posterior_w_sig.flatten()
        ).sum()
        
        # Compute log prior weight probability
        log_prior_w = prior_w(posterior_w_sample).sum()

        log_posterior_b, log_prior_b = 0, 0
        if self.use_bias:
            # Compute log posterior bias probability
            log_posterior_b = jsp.stats.norm.logpdf(
                posterior_b_sample, loc=posterior_b_mean.flatten(), scale=posterior_b_sig.flatten()
            ).sum()
           
            # Compute log prior bias probability
            log_prior_b = prior_b(posterior_b_sample).sum()

        # Compute KL estimate
        kl_div = log_posterior_w + log_posterior_b - log_prior_w - log_prior_b

        return kl_div.sum()

    def KL_divergence(self, posterior_w_mean, posterior_w_sig, prior_w, 
                        posterior_b_mean, posterior_b_sig, prior_b):
        """
        KL divergence between a fully factorized posterior and isotropic prior.
        """
        # compute KL divergence between weight posterior and prior
        kl_div = self._KL_divergence_Gaussians(
            posterior_w_mean.flatten(), 
            jnp.square(posterior_w_sig).flatten(), 
            prior_w.params["loc"], 
            prior_w.params["scale"]**2
        )

        if self.bias:
            # Compute KL divergence between bias posterior and prior
            kl_div += self._KL_divergence_Gaussians(
                posterior_b_mean.flatten(), 
                jnp.square(posterior_b_sig).flatten(),
                prior_b.params["loc"], 
                prior_b.params["scale"]**2
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
