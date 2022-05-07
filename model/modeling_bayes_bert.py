import jax
import haiku as hk
import jax.numpy as jnp
import numpy as np

from model.embedding import Embedding 
from model.linear import LinearLayer

def gelu(x):
    """
    We use this in place of jax.nn.relu because the approximation used 
    produces a non-trivial difference in the output state
    """
    return x * 0.5 * (1.0 + jax.scipy.special.erf(x / jnp.sqrt(2.0)))

class BertEmbeddings(hk.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = Embedding(self.config, 
                                self.config["vocab_size"],
                                self.config["hidden_size"],
                                "word_embeddings",
                                )

        self.position_embeddings = Embedding(self.config, 
                                self.config["max_position_embeddings"],
                                self.config["hidden_size"],
                                "position_embeddings",
                                )          

        self.token_type_embeddings = Embedding(self.config, 
                                self.config["type_vocab_size"],
                                self.config["hidden_size"],
                                "token_type_embeddings",
                                )

    def __call__(self, 
                input_ids, 
                token_type_ids,  
                key,
                training=True):
        
        # Split keys
        keys = jax.random.split(key, num=4)

        word_embeddings, kl_div_word_embedding = self.word_embeddings(input_ids, keys[0], training)
        # if position_ids is None:
        # print('input_ids ', input_ids)
        # print(input_ids.shape)
        seq_length = input_ids.shape[1]
        bsz = input_ids.shape[0]
        position_ids = jnp.repeat(jnp.expand_dims(jnp.arange(seq_length, dtype=jnp.int32), axis=0), bsz, axis=0)
        # print('position_ids ', position_ids)
        # print('position_ids.shape ', position_ids.shape)
        position_embeddings, kl_div_position_embedding = self.position_embeddings(position_ids, keys[1], training)

        token_type_embeddings, kl_div_token_type_embedding = self.token_type_embeddings(token_type_ids, keys[2], training)


        embeddings = word_embeddings + position_embeddings + token_type_embeddings

        embeddings = hk.LayerNorm(
            axis=-1, create_scale=True, create_offset=True,
            name="layernorm"
        )(embeddings)
        
        # When training apply dropout
        if training:
            embeddings = hk.dropout(
                rng=keys[-1],
                rate=self.config['embed_dropout_rate'],
                x=embeddings
            )

        kl_div = kl_div_word_embedding + kl_div_position_embedding + kl_div_token_type_embedding
        return embeddings, kl_div
                   
class MultiHeadAttention(hk.Module):
    """
    Multi-head self-attention block.
    """
    def __init__(self, config, n):
        """
        Build multi-head self-attention block.
        :params config: configuration dictionary.
        :params n: int block identifier.
        """
        super().__init__(name=f"MultiHeadAttention_{n}")
        self.config = config
        if 'bayesian_mhsa_{n}' not in self.config:
            config[f'bayesian_mhsa_{n}'] = config[f'bayesian_mhsa_0']
            config[f'mhsa_prior_distribution_{n}'] = config[f'mhsa_prior_distribution_0']
            config[f'mhsa_prior_params_{n}'] = config[f'mhsa_prior_params_0']
        self.w_queries = LinearLayer(
            input_size=self.config['hidden_size'], 
            output_size=self.config['hidden_size'], 
            bayesian=config[f'bayesian_mhsa_{n}'],
            prior_distribution=config[f'mhsa_prior_distribution_{n}'], 
            prior_params=config[f'mhsa_prior_params_{n}'],
            with_bias=True,
            name="query",
        )
        self.w_keys = LinearLayer(
            input_size=self.config['hidden_size'], 
            output_size=self.config['hidden_size'], 
            bayesian=config[f'bayesian_mhsa_{n}'], 
            prior_distribution=config[f'mhsa_prior_distribution_{n}'], 
            prior_params=config[f'mhsa_prior_params_{n}'],  
            with_bias=True,
            name="key",
        )
        self.w_values = LinearLayer(
            input_size=self.config['hidden_size'], 
            output_size=self.config['hidden_size'], 
            bayesian=config[f'bayesian_mhsa_{n}'], 
            prior_distribution=config[f'mhsa_prior_distribution_{n}'], 
            prior_params=config[f'mhsa_prior_params_{n}'],  
            with_bias=True,
            name="value",
        )
        self.feedforward = LinearLayer(
            input_size=self.config['hidden_size'], 
            output_size=self.config['hidden_size'], 
            bayesian=config[f'bayesian_mhsa_{n}'], 
            prior_distribution=config[f'mhsa_prior_distribution_{n}'], 
            prior_params=config[f'mhsa_prior_params_{n}'],  
            with_bias=True,
            name="output",
        )

    def _split_into_heads(self, x):
        return jnp.reshape(
            x, 
            [
                x.shape[0],
                x.shape[1],
                self.config['n_heads'],
                x.shape[2] // self.config['n_heads']
            ]
        )

    def __call__(self, x, key, mask, kl_mc_samples=3, training=True):
        """
        Forward pass on the multi-head self-attention layer.
        :params x: input x of shape (batch, seq, n_hidden).
        :params key: random key.
        :params mask: self_attention mask of shape (batch, seq)
        :parmas kl_mc_samples: number of Monte Carlo samples to compute the KL divergence.
        :params training: if True apply dropout.
        """        
        # split keys
        rngs = jax.random.split(key, num=5)

        # Project to queries, keys, and values
        # Shapes are all [batch, sequence_length, hidden_size]
        queries, kl_div_queries = self.w_queries(x, rngs[0], kl_mc_samples)
        keys, kl_div_keys = self.w_keys(x, rngs[1], kl_mc_samples)
        values, kl_div_values = self.w_values(x, rngs[2], kl_mc_samples)
        
        # Reshape our hidden state to group into heads
        # New shape are [batch, sequence_length, n_heads, size_per_head]
        queries = self._split_into_heads(queries)
        keys = self._split_into_heads(keys)
        values = self._split_into_heads(values)
        
        # Compute per head attention weights 
        # b: batch
        # s: source sequence
        # t: target sequence
        # n: number of heads
        # h: per-head hidden state
        attention_logits = jnp.einsum('bsnh,btnh->bnst', queries, keys) / np.sqrt(queries.shape[-1])
        # attention_logits = jnp.matmul(queries, keys) / np.sqrt(queries.shape[-1]) # queries: bsnh (need: bnsh), keys: btnh (bnht, using jnp.transpose(keys, (0, 2, 3, 1)).shape)
        # attention_logits = jnp.matmul(jnp.transpose(queries, (0,2,1,3)), jnp.transpose(keys, (0,2,3,1))) / np.sqrt(queries.shape[-1])
        # Add logits of mask tokens with a large negative number to prevent attending to those terms.
        attention_logits += mask * -2**32 
        attention_weights = jax.nn.softmax(attention_logits, axis=-1)
        per_head_attention_output = jnp.einsum('btnh,bnst->bsnh', values, attention_weights)
        attention_output = jnp.reshape(
            per_head_attention_output, 
            [
                per_head_attention_output.shape[0],
                per_head_attention_output.shape[1],
                per_head_attention_output.shape[2] * per_head_attention_output.shape[3]
            ]
        )

        # Apply dense layer to output of attention operation
        attention_output, kl_div_feedforward = self.feedforward(attention_output, rngs[3], kl_mc_samples)

        # Apply dropout at training time
        if training:
            attention_output = hk.dropout(rng=rngs[4], rate=self.config['attention_drop_rate'], x=attention_output)

        # Compute kl divergence 
        kl_div = kl_div_queries + kl_div_keys + kl_div_values + kl_div_feedforward

        return attention_output, kl_div


class BertMLP(hk.Module):
    """
    BertMLP MLP layer. 
    """
    def __init__(self, config, n):
        """
        Build BertMLP MLP layer.
        :params config: configuration dictionary.
        :params n: int block identifier.
        """
        super().__init__(name=f"BertMLPMLP_{n}")
        self.config = config
        if 'bayesian_mlp_{n}' not in self.config:
            config[f'bayesian_mlp_{n}'] = config[f'bayesian_mlp_0']
            config[f'mlp_prior_distribution_{n}'] = config[f'mlp_prior_distribution_0']
            config[f'mlp_prior_params_{n}'] = config[f'mlp_prior_params_0']
        self.linear1 = LinearLayer(
            input_size=self.config['hidden_size'], 
            output_size=self.config['intermediate_size'], 
            bayesian=config[f'bayesian_mlp_{n}'], 
            prior_distribution=config[f'mlp_prior_distribution_{n}'], 
            prior_params=config[f'mlp_prior_params_{n}'],  
            with_bias=True,
            name="intermediate_dense",
        )
        self.linear2 = LinearLayer(
            input_size=self.config['intermediate_size'], 
            output_size=self.config['hidden_size'], 
            bayesian=config[f'bayesian_mlp_{n}'], 
            prior_distribution=config[f'mlp_prior_distribution_{n}'], 
            prior_params=config[f'mlp_prior_params_{n}'],  
            with_bias=True,
            name="output_dense"
        )

    def __call__(self, x, key, kl_mc_samples, training=True):
        """
        Forward pass on the BertMLP MLP layer.
        :params x: input x.
        :params key: random key.
        :parmas kl_mc_samples: number of Monte Carlo samples to compute the KL divergence.
        :params training: if True apply dropout.
        """
        # split keys
        keys = jax.random.split(key, num=3)

        # Project out to higher dim
        intermediate_output, kl_div_linear1 = self.linear1(x, keys[0], kl_mc_samples)

        # Apply gelu nonlinearity
        intermediate_output = gelu(intermediate_output)

        # Project back down to hidden size
        output, kl_div_linear2 = self.linear2(intermediate_output, keys[1], kl_mc_samples)
  
        # Apply dropout at training time
        if training:
            output = hk.dropout(
                rng=keys[2], 
                rate=self.config['fully_connected_drop_rate'],
                x=output
            )

        # Compute kl divergence 
        kl_div = kl_div_linear1 + kl_div_linear2

        return output, kl_div

class BertLayer(hk.Module):
    """
    Bert Layer composed of multihead self-attention.
    """
    def __init__(self, config, n):
        """
        Build Layer block.
        :params config: configuration dictionary.
        :params n: identifier of the block.
        """
        super().__init__(name=f"TransformerBlock_{n}")
        self.config = config
        self.mha = MultiHeadAttention(self.config, n)
        self.mlp = BertMLP(self.config, n)

    def __call__(self, x, key, mask, kl_mc_samples, training=True):
        """
        Forward pass on the transformer block.
        :params x: input. 
        :params key: random key.
        :params mask: self-attention mask.
        :params kl_mc_samples: number of posterior Monte Carlo samples to compute the KL divergence.
        :params training: if True, apply dropout.
        """
        # Split keys
        keys = jax.random.split(key, 2)

        # Feed our input through a multi-head attention operation
        attention_output, kl_div_attention = self.mha(x, keys[0], mask, kl_mc_samples, training)

        # Add a residual connection with the input to the layer
        residual = attention_output + x

        # Apply layer norm to the combined output
        attention_output = hk.LayerNorm(
            axis=-1,
            create_scale=True,
            create_offset=True,
            scale_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform"),
            offset_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform"),
            name="mha_layernorm",
        )(residual)

        # Project out to a larger dim, apply a gelu, and then project back down to our hidden dim
        mlp_output, kl_div_mlp = self.mlp(attention_output, keys[1], kl_mc_samples, training)

        # Residual connection to the output of the attention operation
        output_residual = mlp_output + attention_output

        # Apply another LayerNorm
        layer_output = hk.LayerNorm(
            axis=-1,
            create_scale=True,
            create_offset=True,
            scale_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform"),
            offset_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform"),
            name="mlp_layernorm",
        )(output_residual) 

        # Compute KL divergence 
        kl_div = kl_div_attention + kl_div_mlp

        return layer_output, kl_div

class BertEncoder(hk.Module):
    """
    Transformer trained with Variational Inference.
    """
    def __init__(self, config):

        self.config = config
        super().__init__(name="BertEncoder")
        self.transformer_blocks = [
            BertLayer(self.config, n) for n in range(self.config["n_layers"])
        ]
        self.mask = jnp.zeros((self.config["max_length"], self.config["max_length"]))
    

    def __call__(self, x, key, rate, kl_mc_samples, training=True):

         # Split keys
        keys = jax.random.split(key, num=self.config["n_layers"]+2)
        
        # Embedding : (batch, n_seq, hidden_size)
        # x, kl_div_embedding = self.embedding(x, keys[0], training) 

        # Transformer blocks
        kl_div_transformer = 0.
        for n, transformer_layer in enumerate(self.transformer_blocks):
            x, kl_div = transformer_layer(x, keys[n+1], self.mask, kl_mc_samples, training)
            kl_div_transformer += kl_div

        # Dropout
        if training:
            x = hk.dropout(rng=keys[-2], rate=rate, x=x)
        
        # # Project to the output size
        # x, kl_div_projection_head = self.projection_head(x, keys[-1], kl_mc_samples)  

        # Compute kl divergence
        kl_div =  kl_div_transformer 

        return x, kl_div


class BertPooler(hk.Module):
    
    def __init__(self, config):

        self.config = config
        super().__init__(name="BertPooler")
        self.linear = LinearLayer(
            input_size=self.config['hidden_size'], 
            output_size=self.config['hidden_size'], 
            bayesian=config[f'bayesian_bert_pooler'], 
            prior_distribution=config[f'proj_prior_distribution'], 
            prior_params=config[f'proj_prior_params'],  
            with_bias=True,
            name="dense",
        )

    def __call__(self, x, key, kl_mc_samples, training=True):
        """
        Forward pass on the BertMLP MLP layer.
        :params x: input x.
        :params key: random key.
        :parmas kl_mc_samples: number of Monte Carlo samples to compute the KL divergence.
        :params training: if True apply dropout.
        """
        # split keys
        keys = jax.random.split(key, num=2)

        # Pool output and take last hidden state
        pooler_in = jnp.expand_dims(x[:,0], axis=1)
        # print("pooler_in ", pooler_in.shape)

        # Project out to higher dim
        pooler_output, kl_div_linear = self.linear(pooler_in, keys[0], kl_mc_samples)

        # Apply gelu nonlinearity
        pooler_output = jnp.squeeze(gelu(pooler_output), axis=1)
  
        # Apply dropout at training time
        if training:
            pooler_output = hk.dropout(
                rng=keys[1], 
                rate=self.config['fully_connected_drop_rate'],
                x=pooler_output
            )

        # Compute kl divergence 
        kl_div = kl_div_linear

        return pooler_output, kl_div
        

class BertModule(hk.Module):

    def __init__(self, config):

        self.config = config
        super().__init__(name="BertModule")
        self.embeddings = BertEmbeddings(self.config)
        self.encoder = BertEncoder(self.config)
        self.pooler= BertPooler(self.config)


    def __call__(self, 
                input_ids, 
                token_type_ids, 
                key, rate, kl_mc_samples, training=True):

        # Split keys
        keys = jax.random.split(key, num=3)
        # Embedding : (batch, n_seq, hidden_size)
        x, kl_div_embedding = self.embeddings(input_ids, 
                                            token_type_ids, 
                                            keys[0], training) 

        x, kl_div_encoder = self.encoder(x, keys[1], rate, kl_mc_samples, training)

        output, kl_pooler = self.pooler(x, keys[2], kl_mc_samples, training)

        kl_div = kl_div_embedding + kl_div_encoder + kl_pooler

        return output, kl_div

class BertForSequenceClassification(hk.Module):

    def __init__(self, config):

        self.config = config
        super().__init__(name="BertForSequenceClassification")

        self.bert = BertModule(self.config)

        self.classifier =  LinearLayer(
            input_size=self.config['hidden_size'], 
            output_size=self.config['num_labels'], 
            bayesian=config[f'bayesian_proj'], 
            prior_distribution=config[f'proj_prior_distribution'], 
            prior_params=config[f'proj_prior_params'],  
            with_bias=True,
            name="classifier",
        )
        self.vectorized_forward = jax.vmap(
            self.forward, in_axes=(None, None, 0, None, None, None), out_axes=(0, 0)
        )

    def forward(self, input_ids, 
            token_type_ids, 
            key, rate, kl_mc_samples, training=True):

        keys = jax.random.split(key, num=3)

        output, kl_bert = self.bert(input_ids, token_type_ids,
                            keys[0], rate, kl_mc_samples, training)

        output = hk.dropout(
            rng=keys[1], 
            rate=self.config['fully_connected_drop_rate'],
            x=output
        )
        # Final Classifier Layer
        logits, kl_classifier = self.classifier(jnp.expand_dims(output, axis=1), keys[2], kl_mc_samples)
        # print("logits ", logits.shape)  # we need to expand_dims since, einsum for Linear expects three dimensions
        logits = jnp.squeeze(logits, axis=1)
        # print("logits ", logits.shape) 

        kl_div = kl_bert + kl_classifier

        return logits, kl_div
    
    def __call__(self, input_ids, 
            token_type_ids, 
            pred_mc_samples,
            kl_mc_samples, training=True):
        """
        Forward pass on the model. This function vectorizes the computation.
        :params x: input tensor.
        :params pred_mc_samples: number of samples from predictive distribution.
        :params kl_mc_samples: number of Monte Carlo samples to compute the KL divergence.
        :params training: if True apply dropout.
        """
        # # Get keys
        # keys = jnp.array(hk.next_rng_keys(pred_mc_samples))
        
        # # Forward pass : (mc_samples, batch, max_len, n_outputs)
        # logits, kl_divs = self.vectorized_forward(
        #     input_ids, token_type_ids, keys, self.config['regressor_drop_rate'], kl_mc_samples, training
        # )

        # return logits, kl_divs.mean()
        print(input_ids.shape, token_type_ids.shape)
        return self.forward(input_ids, 
                            token_type_ids, hk.next_rng_key(), 
                            self.config['regressor_drop_rate'], kl_mc_samples, training)