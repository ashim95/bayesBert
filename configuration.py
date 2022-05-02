configuration = {
    # Train set
    "language":"English",
    "split":"train",
    "batch_size":8,
    "vocab_size":28996, #3444,
    "n_epochs": 5, #100,

    # Positional embedding
    'min_scale': 1.,
    'max_scale': 10000.,

    # Transformer hyperparameters
    'n_layers': 12,
    'max_length': 512, #40,
    'embed_dropout_rate': 0.1,
    'fully_connected_drop_rate': 0.1,
    'attention_drop_rate': 0.1,
    'hidden_size': 768,
    'intermediate_size': 3072,
    'n_heads': 12,
    'weight_stddev': 0.02,
    'n_outputs': 17, # mean + var of prediction
    'regressor_drop_rate': 0.1,
    'type_vocab_size':2,

    # sampling parameters
    'pred_mc_samples': 1, 
    'kl_mc_samples': 1,
}
