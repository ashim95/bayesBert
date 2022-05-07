from scipy.sparse import data
import cloudpickle
import pickle
import jax
import pickle
import argparse
import os

import haiku as hk
import numpy as np
import jax.numpy as jnp
import jax.scipy as jsp

from jax.experimental import optimizers

# Project module imports
import evaluation
from jax.experimental import optimizers
from configuration import configuration
from dataloader import DataLoader
# from model.modeling_bayes_bert import BertForSequenceClassification
from model.modeling_bayes_bert_wo_dropout import BertForSequenceClassification
from experiments import ood_entropy_split, data_uncertainty, pred_ambiguity_from_entropy
from pprint import pprint
from utils import get_dataset, glue_train_data_collator, glue_eval_data_collator, set_pretrained_params

from tqdm import tqdm
#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

jax.config.update("jax_debug_nans", True)
jax.config.update("jax_enable_x64", True)


# Parse arguments
parser = argparse.ArgumentParser()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Configuration
parser.add_argument(
    '--config_name', required=True, default='mle',
    help="Name of the configuration: mle, embedding, attention, final, full."
)
parser.add_argument(
    "--perform_experiments", type=str2bool, default=True,
    help="Perform experiments or not."
)
# Save model
parser.add_argument(
    "--save_model", type=str2bool, default=True,
    help="Save model or not."
)
parser.add_argument(
    "--load_pretrained", type=str2bool, default=False,
    help="Load pretrained model or not."
)
# Sampling arguments
parser.add_argument(
    "--pred_mc_samples", type=int, default=configuration['pred_mc_samples'],
    help="Number of samples to compute predictive distribution."
)
parser.add_argument(
    "--kl_mc_samples", type=int, default=configuration['kl_mc_samples'],
    help="Number of samples to compute the KL divergence."
)
# Training arguments
parser.add_argument(
    "--n_epochs", type=int, default=configuration['n_epochs'],
    help="Number of training epochs."
)
# Variationial Inference argument
parser.add_argument(
    "--priors", type=str, required=True,
    help="Type of priors to use: gaussian or improved."
)

# Dataset Arguments
parser.add_argument(
    "--task_name", type=str, required=True,
    help="The GLUE task name."
)
parser.add_argument(
    "--data_dir", type=str, required=False,
    help="The input data dir. Should contain the .tsv files (or other data files) for the task."
)
parser.add_argument(
    "--cache_dir", type=str, required=False, default=None,
    help="The cache_dir data dir. "
)
parser.add_argument(
    "--overwrite_cache", type=bool, default=False,
    help="Overwrite the cached training and evaluation sets"
)
parser.add_argument(
    "--train_file", default=None, type=str,
)
parser.add_argument(
    "--validation_file", default=None, type=str,
)
parser.add_argument(
    "--test_file", default=None, type=str,
)
parser.add_argument(
    "--model_name_or_path", default="bert-base-cased", type=str,
)
parser.add_argument(
    "--use_slow_tokenizer", type=bool, default=False,
    help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library)."
)
parser.add_argument(
    "--max_seq_length", type=int, default=None,
    help="The maximum total input sequence length after tokenization. If set, sequences longer "
            "than this will be truncated, sequences shorter will be padded."
)

ARGS = parser.parse_args()
configuration['max_position_embeddings'] = ARGS.max_seq_length
# Update arguments based on input arguments
configuration['load_pretrained'] = ARGS.load_pretrained
configuration['save_model'] = ARGS.save_model
configuration['pred_mc_samples'] = ARGS.pred_mc_samples
configuration['kl_mc_samples'] = ARGS.kl_mc_samples
configuration['n_epochs'] = ARGS.n_epochs

if ARGS.priors == 'gaussian':
    # Embeddings
    configuration['emb_prior_distribution'] = "gaussian"
    configuration['emb_prior_params'] = {"loc":0, "scale":1e0}
    # Transformer block 1
    configuration['mlp_prior_distribution_0'] = "gaussian"
    configuration['mlp_prior_params_0'] = {"loc":0, "scale":1e-2}
    configuration['mhsa_prior_distribution_0'] = "gaussian"
    configuration['mhsa_prior_params_0'] = {"loc":0, "scale":1e-2}
    # Projection head
    configuration['proj_prior_distribution'] = "gaussian"
    configuration['proj_prior_params'] = {"loc":0, "scale":1e-1}
elif ARGS.priors == 'improved':
    # Embeddings
    configuration['emb_prior_distribution'] = "gaussian"
    configuration['emb_prior_params'] = {"loc":0, "scale":0.3}
    # Transformer block 1
    configuration['mlp_prior_distribution_0'] = "t"
    configuration['mlp_prior_params_0'] = {'df':10000,"loc":0, "scale":0.00025}
    configuration['mhsa_prior_distribution_0'] = "t"
    configuration['mhsa_prior_params_0'] = {'df':12,"loc":0, "scale":0.0001}
    # Projection head
    configuration['proj_prior_distribution'] = "gaussian"
    configuration['proj_prior_params'] = {"loc":0, "scale":0.3}
else:
    raise Exception("Invalide provided priors argument. Must be gaussian or improved")

if ARGS.config_name == "mle":
    configuration.update(
        {
            'save_path':f"saved_model/mle_{ARGS.priors}_transformer.bin",
            'bayesian_emb': False,
            'bayesian_mlp_0': False,
            'bayesian_mhsa_0': False,
            'bayesian_proj': False,
            'bayesian_bert_pooler': False,
        }
    )
elif ARGS.config_name == "embedding":
    configuration.update(
        {
            'save_path':f"saved_model/emb_{ARGS.priors}_transformer.bin",
            'bayesian_emb': True,
            'bayesian_mlp_0': False,
            'bayesian_mhsa_0': False,
            'bayesian_proj': False,
            'bayesian_bert_pooler': False,
        }
    )
elif ARGS.config_name == "attention":
    configuration.update(
        {
            'save_path':f"saved_model/attention_{ARGS.priors}_transformer.bin",
            'bayesian_emb': False,
            'bayesian_mlp_0': True,
            'bayesian_mhsa_0': True,
            'bayesian_proj': False,
            'bayesian_bert_pooler': False,
        }
    )
elif ARGS.config_name == "final":
    configuration.update(
        {
            'save_path':f"saved_model/final_{ARGS.priors}_transformer.bin",
            'bayesian_emb': False,
            'bayesian_mlp_0': False,
            'bayesian_mhsa_0': False,
            'bayesian_proj': True,
            'bayesian_bert_pooler': False,
        }
    )
elif ARGS.config_name == "full":
    configuration.update(
        {
            'save_path':f"saved_model/full_{ARGS.priors}_transformer.bin",
            'bayesian_emb': True,
            'bayesian_mlp_0': True,
            'bayesian_mhsa_0': True,
            'bayesian_proj': True,
            'bayesian_bert_pooler': True,
        }
    )
else:
    raise Exception("Invalide config name must be mle, embedding, attention, final, full")



def build_model(configuration, key):
    """
    Build the Bayesian Transformer.
    :params configuration: configuration dictionary.
    :params key: random key.
    """
    # Define the model
    def bayesian_transformer_fn(input_ids, token_type_ids,
                                pred_mc_samples=configuration['pred_mc_samples'],
                                kl_mc_samples=configuration['kl_mc_samples'], training=True):
        """
        Define Transformer function.
        """
        transformer = BertForSequenceClassification(configuration)
        fwd, kl_div = transformer(input_ids, token_type_ids, pred_mc_samples, kl_mc_samples, training)

        return fwd, kl_div

    key, sub_key = jax.random.split(key)
    transformer = hk.transform(bayesian_transformer_fn, apply_rng=True)

    # Inititialize model
    sample_input_ids = jnp.zeros((configuration['batch_size'], configuration['max_length']), dtype=jnp.int32)
    sample_token_type_ids = jnp.zeros((configuration['batch_size'], configuration['max_length']), dtype=jnp.int32)
    transformer_params = transformer.init(sub_key, sample_input_ids, sample_token_type_ids, training=True)

    # Load saved model
    load_model = configuration["load_pretrained"]
    if load_model:
        print(f"Loading saved weights from {configuration['save_path']}", flush=True)
        with open(configuration["save_path"],"rb") as f:
            transformer_params = pickle.load(f)

    return transformer, transformer_params

def load_optimizers(configuration, transformer_params):
    """
    Load optimizers.
    :params configuration: configuration dictionary.
    :params transformer_params: Transformer parameters.
    """
    # Define optimizers
    def make_lr_schedule(warmup_steps, d_model):
        """
        Define a triangular learning-rate schedule as in Attention is All you Need.
        :params warmup_steps: number of warm-up steps - learning rate increases
        :params d_model: hidden size of the model.
        """
        def lr_schedule(step):
            lr = d_model**(-0.5) * jax.lax.cond(step**(-0.5) <= step*warmup_steps**(-1.5),
                                                lambda _: step**(-0.5),
                                                lambda _: step*warmup_steps**(-1.5),
                                                operand=None)
            return lr
        return lr_schedule

    opt_init, opt_update, get_params = optimizers.adam(
        step_size=make_lr_schedule(warmup_steps=4000,
        d_model=configuration["hidden_size"]),
        b1=0.9, b2=0.98, eps=1e-09
    )
    opt_state = opt_init(transformer_params)

    return opt_init, opt_update, get_params, opt_state

# Training functions
def evaluate(params, step, key, dataloader):
    """
    Evaluate model during training.
    :params params: Transformer model parameters.
    :params step: current epoch.
    :params key: random key.
    :params dataloader: placeholder for data.
    """
    # Select validation data
    dataloader.change_split("dev")

    # Compute metrics
    metrics = np.stack([loss_eval(params, key, x, y) for x,y in dataloader])
    kl_div, cross_entropy, ece_mean, accuracy, mean_entropy = jnp.sum(metrics, axis=0)
    n_distributions = np.prod(dataloader.dev['text'].shape[:3])
    n_samples = np.prod(dataloader.dev['text'].shape[:2])
    cross_entropy /= n_samples
    kl_div /= dataloader.dev['text'].shape[0]
    ece_mean /= dataloader.dev['text'].shape[0]*dataloader.dev['text'].shape[2] # divide by # time steps
    accuracy /= n_distributions
    mean_entropy /= n_distributions
    loss = kl_div + cross_entropy*n_samples

    print(f"[Epochs {step}] - loss: {loss:.4f} - kl_div: {kl_div:.4f} - ", end="", flush=True)
    print(f"cross_entropy: {cross_entropy:.4f} - ece_mean: {ece_mean:4f} - ", end="", flush=True)
    print(f"accuracy: {accuracy:.4f} - mean_entropy: {mean_entropy:.4f}", flush=True)

@jax.jit
def ece(probs, labels):
    """
    Computes the Expected Calibration Error (ECE).
    :params probs: vector of probabilies. Has shape [n_examples, n_classes].
    :params labels: labels of targets. Has shape [n_class]
    """
    n_bins=10

    n_examples, n_classes = probs.shape

    # Assume that the prediction is the class with the highest prob.
    preds = jnp.argmax(probs, axis=1)
    onehot_labels = jnp.eye(n_classes)[labels]
    predicted_class_probs = jnp.max(probs, axis=1)

    # Use uniform bins on the range of probabilities, i.e. closed interval [0.,1.]
    bin_upper_edges = jnp.histogram_bin_edges(jnp.array([]), bins=n_bins, range=(0., 1.))
    bin_upper_edges = bin_upper_edges[1:] # bin_upper_edges[0] = 0.

    probs_as_bin_num = jnp.digitize(predicted_class_probs, bin_upper_edges)
    sums_per_bin = jnp.bincount(probs_as_bin_num, minlength=n_bins, weights=predicted_class_probs, length=n_bins)
    sums_per_bin = sums_per_bin.astype(jnp.float32)

    total_per_bin = jnp.bincount(probs_as_bin_num, minlength=n_bins, length=n_bins) \
        + jnp.finfo(sums_per_bin.dtype).eps # division by zero
    avg_prob_per_bin = sums_per_bin / total_per_bin

    accuracies = jnp.array([onehot_labels[i, preds[i]] for i in range(preds.shape[0])]) # accuracies[i] is 0 or 1
    accuracies_per_bin = jnp.bincount(probs_as_bin_num, weights=accuracies, minlength=n_bins, length=n_bins) \
        / total_per_bin

    prob_of_being_in_a_bin = total_per_bin / float(n_examples)

    ece_ret = jnp.abs(accuracies_per_bin - avg_prob_per_bin) * prob_of_being_in_a_bin
    ece_ret = jnp.sum(ece_ret)

    return ece_ret

@jax.jit
def loss_eval(params, key, x, y):
    """
    Compute evaluation metrics.
    :params params: Transformer model parameters.
    :params key: random key.
    :params x: model input tensor.
    :params y: target output tensor.
    """
    # Forward pass : (mc_samples, batch, max_len, n_outputs)
    logits, kl_div = transformer.apply(params, key, x, training=False) # (mc_samples, batch, max_len, n_outputs)
    preds = jnp.mean(jax.nn.softmax(logits, axis=-1), axis=0)
    labels = jax.nn.one_hot(y, logits.shape[-1])
    cross_entropy = -jnp.sum(labels * jnp.mean(jax.nn.log_softmax(logits, -1), axis=0))
    accuracy = (jnp.argmax(preds, -1) == y).sum()
    ece_sum = jax.vmap(ece, in_axes=(0,0))(preds.transpose((1,0,2)), y.transpose((1,0))).sum()
    entropy_sum = jnp.sum(jsp.special.entr(preds))

    return [kl_div, cross_entropy, ece_sum, accuracy, entropy_sum]

@jax.jit
def loss_fn(params, key, batch, n_samples):
    """
    Loss function.
    :params params: Transformer model parameters.
    :params key: random key.
    :params x: model input tensor.
    :params y: target output tensor.
    :params n_samples: number of training samples.
    """
    # Forward pass : (mc_samples, batch, max_len, n_outputs)
    logits, kl_div = transformer.apply(params, key, batch['input_ids'], batch['token_type_ids'], training=True)
    # print(logits.shape) logits.shape = mc_samples, batch, num_classes
    # sss
    labels = jax.nn.one_hot(batch.pop("labels"), logits.shape[-1])
    cross_entropy = -jnp.sum(labels * jnp.mean(jax.nn.log_softmax(logits, -1), axis=0))

    return cross_entropy * n_samples / batch['input_ids'].shape[0] + kl_div

@jax.jit
def update(step, opt_state, batch, key, n_samples):
    """
    Update weights.
    :params step: Current optimizer iteration.
    :params opt_state: Current state of the optimizer
    :params x: model input tensor.
    :params y: target output tensor.
    :params key: random key.
    :params n_samples: number of training samples.
    """
    params = get_params(opt_state)
    value, grads = jax.value_and_grad(loss_fn)(params, key, batch, n_samples)
    opt_state = opt_update(step, grads, opt_state)

    return opt_state, value

# Training loop
def train_old(opt_state, key, dataloader_dict, n_epochs):
    """
    Train model.
    :params opt_state: Current state of the optimizer
    :params key: random key.
    :params dataloader: placeholder for data.
    :params n_epochs: number of training epochs.
    """
    step = 0
    # n_batchs = dataloader.train["text"].shape[0]
    n_batchs = len(dataloader_dict['train'])
    for epoch in range(n_epochs):
        # dataloader.change_split("train")
        key, sub_key = jax.random.split(key)
        p = jax.random.permutation(sub_key, n_batchs)
        shuffled_data = dataloader[p][0], dataloader[p][1]
        n_train_samples = np.prod(dataloader.train["tags"].shape)
        for x, y in zip(*shuffled_data):
            key, sub_key = jax.random.split(key)
            opt_state, loss = update(step+1, opt_state, x, y, sub_key, n_train_samples)
            step += 1
        if epoch % 1 == 0 or epoch == n_epochs-1:
            dataloader.change_split("dev")
            key, sub_key = jax.random.split(key)
            params = get_params(opt_state)
            evaluate(params, epoch, sub_key, dataloader)

    return opt_state

def train(opt_state, key, dataset_dict, n_epochs):
    """
    Train model.
    :params opt_state: Current state of the optimizer
    :params key: random key.
    :params dataloader: placeholder for data.
    :params n_epochs: number of training epochs.
    """
    step = 0
    n_epochs = 1
    train_dataset = dataset_dict['train']
    eval_dataset = dataset_dict['dev']
    test_dataset = dataset_dict['test']
    steps_per_epoch = len(train_dataset) // configuration['batch_size']
    eval_loader = glue_eval_data_collator(eval_dataset, configuration['batch_size'])
    test_loader = glue_eval_data_collator(test_dataset, configuration['batch_size'])
    total_steps = steps_per_epoch * n_epochs
    epochs = tqdm(range(n_epochs), desc=f"Epoch ... (0/{n_epochs})", position=0, leave=True)
    for epoch in epochs:
        key, data_key = jax.random.split(key)
        train_loader = glue_train_data_collator(data_key, train_dataset, configuration['batch_size'])
        for step, batch in enumerate(
                tqdm(
                    train_loader,
                    total=steps_per_epoch,
                    desc="Training...",
                    position=1,
                ),
            ):
            # dataloader.change_split("train")
            # key, sub_key = jax.random.split(key)
            # p = jax.random.permutation(sub_key, n_batchs)
            # shuffled_data = dataloader[p][0], dataloader[p][1]
            # n_train_samples = np.prod(dataloader.train["tags"].shape)
            # for x, y in zip(*shuffled_data):
            key, sub_key = jax.random.split(key)
            n_train_samples = batch['input_ids'].shape[0]
            opt_state, loss = update(step+1, opt_state, batch, sub_key, n_train_samples)
            # epochs.write(
            #         f"Step... ({step}/{total_steps} | Training Loss: {loss})"
            #     )
            step += 1
            # if epoch % 1 == 0 or epoch == n_epochs-1:
            #     dataloader.change_split("dev")
            #     key, sub_key = jax.random.split(key)
            #     params = get_params(opt_state)
            #     evaluate(params, epoch, sub_key, dataloader)
    return opt_state


if __name__ == "__main__":
    # Define random key
    key = jax.random.PRNGKey(0)
    key, sub_key, data_key = jax.random.split(key, num=3)

    print('Printing model configuration ')
    pprint(configuration)
    ARGS.n_layers = configuration['n_layers']

    # Load data
    print("Loading the data...", flush=True)
    train_dataset, eval_dataset, test_dataset, num_labels = get_dataset(ARGS)
    configuration["num_labels"] = num_labels

    # print(len(train_dataset), len(eval_dataset))
    # print(list(train_dataset)[0])
    # dataloader = DataLoader(configuration)
    # dataloader.change_split("train")
    train_loader = glue_train_data_collator(data_key, train_dataset, configuration['batch_size'])
    eval_loader = glue_eval_data_collator(eval_dataset, configuration['batch_size'])
    test_loader = glue_eval_data_collator(test_dataset, configuration['batch_size'])

    dataset_dict = {
        'train': train_dataset,
        'dev': eval_dataset,
        'test': test_dataset,
    }

    # Build model
    print("Loading model...", flush=True)
    transformer, transformer_params = build_model(configuration, sub_key)
    # print(transformer)
    # print(transformer_params.keys())
    # # posterior_w_mean
    # # print(transformer_params['BertForSequenceClassification/~/LinearLayer/~/BayesianLinear'].keys())
    # print(jnp.sum(transformer_params['BertForSequenceClassification/~/classifier/~/BayesianLinear']['posterior_w_mean']))
    # transformer_params['BertForSequenceClassification/~/classifier/~/BayesianLinear']['posterior_w_mean'] = jnp.zeros_like(transformer_params['BertForSequenceClassification/~/classifier/~/BayesianLinear']['posterior_w_mean'])
    # print(jnp.sum(transformer_params['BertForSequenceClassification/~/classifier/~/BayesianLinear']['posterior_w_mean']))
    # print(transformer_params['BertForSequenceClassification/~/BertModule/~/bert_embeddings/layer_norm'].keys())
    # print(transformer_params['BertForSequenceClassification/~/classifier/~/StandardLinear'].keys())
    # print(jnp.shape(transformer_params['BertForSequenceClassification/~/BertModule/~/bert_embeddings/layer_norm']['scale']))
    # print(jnp.shape(transformer_params['BertForSequenceClassification/~/BertModule/~/bert_embeddings/layer_norm']['offset']))
    # aaa
    set_pretrained_params(ARGS, transformer_params)
    # print(transformer_params['BertForSequenceClassification/~/BertModule/~/bert_embeddings/~/word_embeddings/~/LearnableEmbedding/embed'].keys())
    # print(jnp.sum(transformer_params['BertForSequenceClassification/~/BertModule/~/bert_embeddings/~/word_embeddings/~/LearnableEmbedding/embed']))
    # pprint(transformer_params.keys())
    # for key in transformer_params.keys():
    #     print(key)
    # aaa
    print('Platform used by jax')
    from jax.lib import xla_bridge
    print(xla_bridge.get_backend().platform)

    # Load optimizers
    print("Define optimizers...", flush=True)
    opt_init, opt_update, get_params, opt_state = load_optimizers(configuration, transformer_params)
    # print(get_params(opt_state).keys())

    # Train
    print("Training model...", flush=True)
    n_epochs = configuration["n_epochs"]
    opt_state = train(opt_state, key, dataset_dict, n_epochs=n_epochs)

    # Save model
    if configuration["save_model"]:
        print(f"Saving model weigths at {configuration['save_path']}", flush=True)
        with open(configuration["save_path"],"wb") as f:
            cloudpickle.dump(get_params(opt_state), file=f)

    # Evaluate model
    print("Evaluating the model...", flush=True)
    key, sub_key = jax.random.split(key)
    params = get_params(opt_state)
    dataloader.change_split("test")
    evaluation.evaluate(dataloader, transformer, params, sub_key, configuration, n=3)

    # Perform experiments
    if ARGS.perform_experiments:
        print("Performing experiments...", flush=True)
        # pred_ambiguity_from_entropy(dataloader, transformer, params, sub_key, configuration)
        data_uncertainty(dataloader, transformer, params, sub_key, configuration)
        ood_entropy_split(dataloader, transformer, params, sub_key)
