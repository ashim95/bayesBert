import datasets
import logging
import numpy as np
import jax
import jax.numpy as jnp
from datasets import load_dataset, load_metric
from typing import Any, Callable, Dict, Optional, Tuple
from transformers import (
    AdamW,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
import torch

logger = logging.getLogger(__name__)
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}
Dataset = datasets.arrow_dataset.Dataset
PRNGKey = Any

def get_dataset(data_args):
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset("glue", data_args.task_name, cache_dir=data_args.cache_dir)
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        # if training_args.do_predict:
        #     if data_args.test_file is not None:
        #         train_extension = data_args.train_file.split(".")[-1]
        #         test_extension = data_args.test_file.split(".")[-1]
        #         assert (
        #             test_extension == train_extension
        #         ), "`test_file` should have the same extension (csv or json) as `train_file`."
        #         data_files["test"] = data_args.test_file
        #     else:
        #         raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            raw_datasets = load_dataset("csv", data_files=data_files, cache_dir=model_args.cache_dir)
        else:
            # Loading a dataset from local json files
            raw_datasets = load_dataset("json", data_files=data_files, cache_dir=model_args.cache_dir)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.


    label_list = raw_datasets["train"].features["label"].names
    num_labels = len(label_list)
    # Preprocessing the raw_datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None
    label_to_id = None
    # label_to_id = {v: i for i, v in enumerate(sorted(label_list))}
    # train_dataset = raw_datasets["train"]
    # eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
    # test_dataset = raw_datasets["test_matched" if data_args.task_name == "mnli" else "test"]

    tokenizer = AutoTokenizer.from_pretrained(
        data_args.model_name_or_path, use_fast=not data_args.use_slow_tokenizer
    )  
    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding="max_length", max_length=data_args.max_seq_length, truncation=True)
        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result
    processed_datasets = raw_datasets.map(
        preprocess_function, batched=True, remove_columns=raw_datasets["train"].column_names
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
    test_dataset = processed_datasets["test_matched" if data_args.task_name == "mnli" else "test"]
    return train_dataset, eval_dataset, test_dataset, num_labels


def glue_train_data_collator(rng: PRNGKey, dataset: Dataset, batch_size: int):
    """Returns shuffled batches of size `batch_size` from truncated `train dataset`, sharded over all local devices."""
    steps_per_epoch = len(dataset) // batch_size
    perms = jax.random.permutation(rng, len(dataset))
    perms = perms[: steps_per_epoch * batch_size]  # Skip incomplete batch.
    perms = perms.reshape((steps_per_epoch, batch_size))

    for perm in perms:
        batch = dataset[perm]
        batch = {k: np.array(v) for k, v in batch.items()}

        yield batch

def glue_eval_data_collator(dataset: Dataset, batch_size: int):
    """Returns batches of size `batch_size` from `eval dataset`, sharded over all local devices."""
    for i in range(len(dataset) // batch_size):
        batch = dataset[i * batch_size : (i + 1) * batch_size]
        batch = {k: np.array(v) for k, v in batch.items()}

        yield batch

def find_key(keys, keywords):
    for key in keys:
        if all(keyword in key for keyword in keywords):
            return key
    return None

def set_pretrained_embeddings(transformer_params, embed_weights, embed_type):
    
    if embed_type == 'word':
        key = find_key(transformer_params.keys(), ['word_embeddings'])
    elif embed_type == 'position':
        key = find_key(transformer_params.keys(), ['position_embeddings'])
    elif embed_type == 'token_type':
        key = find_key(transformer_params.keys(), ['token_type_embeddings'])
    elif embed_type == 'layernorm':
        key = find_key(transformer_params.keys(), ['bert_embeddings', "layernorm"])
    
    if key is None:
        raise ValueError("Key not found in parameter list for embedding type: ", embed_type)
    print("Setting weight for {}".format(key))
    if 'BayesianEmbedding' in key:
        # print(jnp.sum(transformer_params[key]['w_mean']))
        transformer_params[key]['w_mean'] = jnp.array(embed_weights.cpu().detach().numpy())
        # print(jnp.sum(transformer_params[key]['w_mean']))
    elif 'LearnableEmbedding' in key:
        # print(jnp.sum(transformer_params[key]['embeddings']))
        transformer_params[key]['embeddings'] = jnp.array(embed_weights.cpu().detach().numpy())
        # print(jnp.sum(transformer_params[key]['embeddings']))
    elif 'layernorm' in key:
        # print(jnp.sum(transformer_params[key]['scale']), jnp.sum(transformer_params[key]['offset']))
        transformer_params[key]['scale'] = jnp.array(embed_weights[0].cpu().detach().numpy())
        transformer_params[key]['offset'] = jnp.array(embed_weights[1].cpu().detach().numpy())
        # print(jnp.sum(transformer_params[key]['scale']), jnp.sum(transformer_params[key]['offset']))


def set_mha_weights(transformer_params, weights, weight_type, layer_number):

    if weight_type in ['query', 'key', 'value', 'output']:
        key = find_key(transformer_params.keys(), [f'MultiHeadAttention_{layer_number}', weight_type])
    print("Setting weight for {}".format(key))
    if 'BayesianLinear' in key:
        # print(jnp.sum(transformer_params[key]['posterior_w_mean']))
        # print(jnp.sum(transformer_params[key]['posterior_b_mean']))
        transformer_params[key]['posterior_w_mean'] = jnp.array(weights[0].cpu().detach().numpy())
        transformer_params[key]['posterior_b_mean'] = jnp.array(weights[1].cpu().detach().numpy()) # for setting bias
        # print(jnp.sum(transformer_params[key]['posterior_w_mean']))
        # print(jnp.sum(transformer_params[key]['posterior_b_mean']))
    elif 'StandardLinear' in key:
        # print(jnp.sum(transformer_params[key]['w']))
        # print(jnp.sum(transformer_params[key]['b']))
        # transformer_params[key][]
        transformer_params[key]['w'] = jnp.array(weights[0].cpu().detach().numpy())
        transformer_params[key]['b'] = jnp.array(weights[1].cpu().detach().numpy()) # for setting bias
        # print(jnp.sum(transformer_params[key]['w']))
        # print(jnp.sum(transformer_params[key]['b']))

def set_bert_layer_layernorm(transformer_params, weights, layernorm_type, layer_number):
    # layernorm_type can be mha_layernorm or mlp_layernorm
    key = find_key(transformer_params.keys(), [f'TransformerBlock_{layer_number}', layernorm_type])
    print("Setting weight for {}".format(key))
    transformer_params[key]['scale'] = jnp.array(weights[0].cpu().detach().numpy())
    transformer_params[key]['offset'] = jnp.array(weights[1].cpu().detach().numpy())

def set_bert_layer_mlp_weights(transformer_params, weights, weight_type, layer_number):
    # weight_type can be intermediate_dense or output_dense
    key = find_key(transformer_params.keys(), [f'TransformerBlock_{layer_number}', weight_type])
    print("Setting weight for {}".format(key))
    if 'BayesianLinear' in key:
        transformer_params[key]['posterior_w_mean'] = jnp.array(weights[0].cpu().detach().numpy())
        transformer_params[key]['posterior_b_mean'] = jnp.array(weights[1].cpu().detach().numpy()) # for setting bias
    elif 'StandardLinear' in key:
        # transformer_params[key][]
        # print(transformer_params[key]['w'].shape)
        # print(transformer_params[key]['b'].shape)
        transformer_params[key]['w'] = jnp.array(weights[0].cpu().detach().numpy())
        transformer_params[key]['b'] = jnp.array(weights[1].cpu().detach().numpy()) # for setting bias
        # print(transformer_params[key]['w'].shape)
        # print(transformer_params[key]['b'].shape)

def set_bert_pooler_weights(transformer_params, weights):
    # weight_type can be intermediate_dense or output_dense
    key = find_key(transformer_params.keys(), ['BertPooler', 'dense'])
    print("Setting weight for {}".format(key))
    if 'BayesianLinear' in key:
        transformer_params[key]['posterior_w_mean'] = jnp.array(weights[0].cpu().detach().numpy())
        transformer_params[key]['posterior_b_mean'] = jnp.array(weights[1].cpu().detach().numpy()) # for setting bias
    elif 'StandardLinear' in key:
        # transformer_params[key][]
        transformer_params[key]['w'] = jnp.array(weights[0].cpu().detach().numpy())
        transformer_params[key]['b'] = jnp.array(weights[1].cpu().detach().numpy()) # for setting bias

def set_pretrained_params(args,transformer_params):

    model = AutoModel.from_pretrained(args.model_name_or_path,
                                     cache_dir=args.cache_dir, 
                                     )

    # print(model)
    # print('embeddings shape ')
    # print(type(model.embeddings.word_embeddings))
    # print((model.embeddings.word_embeddings.weight.shape))
    # print('Torch values ')
    # print(torch.sum(model.embeddings.word_embeddings.weight))
    # print(torch.sum(model.embeddings.position_embeddings.weight))
    # print(model.embeddings.LayerNorm.weight.shape)
    # print(model.embeddings.LayerNorm.bias.shape)
    # print(torch.sum(model.embeddings.LayerNorm.weight))
    # print(torch.sum(model.embeddings.LayerNorm.bias))
    # print(torch.sum(model.encoder.layer[0].attention.self.query.weight))
    # print(torch.sum(model.encoder.layer[0].attention.self.query.bias))
    # print(model.encoder.layer[0].attention.self.query.weight.shape)
    # print(model.encoder.layer[0].intermediate.dense.weight.shape)
    # print(model.encoder.layer[0].output.dense.weight.shape)
    # set BERT Embeddings (Word, Position, Token Type, and LayerNorm)
    set_pretrained_embeddings(transformer_params, model.embeddings.word_embeddings.weight, "word")
    set_pretrained_embeddings(transformer_params, model.embeddings.position_embeddings.weight, "position")
    set_pretrained_embeddings(transformer_params, model.embeddings.token_type_embeddings.weight, "token_type")
    set_pretrained_embeddings(transformer_params, (model.embeddings.LayerNorm.weight, model.embeddings.LayerNorm.bias), "layernorm")

    # set BERTLayer Weights (SelfAttention (Q,K,V,O), 2 MLPs, 2 LayerNorms)
    # Recall that when accessing weights with .weight in pytorch, the matrices are saved as transpose
    for layer_num in range(args.n_layers):
        set_mha_weights(transformer_params, 
                        (model.encoder.layer[layer_num].attention.self.query.weight.t(), 
                        model.encoder.layer[layer_num].attention.self.query.bias),
                        'query',
                        layer_num,
                        )
        set_mha_weights(transformer_params, 
                        (model.encoder.layer[layer_num].attention.self.key.weight.t(), 
                        model.encoder.layer[layer_num].attention.self.key.bias),
                        'key',
                        layer_num,
                        )
        set_mha_weights(transformer_params, 
                        (model.encoder.layer[layer_num].attention.self.value.weight.t(), 
                        model.encoder.layer[layer_num].attention.self.value.bias),
                        'value',
                        layer_num,
                        )
        set_mha_weights(transformer_params, 
                        (model.encoder.layer[layer_num].attention.output.dense.weight.t(), 
                        model.encoder.layer[layer_num].attention.output.dense.bias),
                        'output',
                        layer_num,
                        )
        set_bert_layer_mlp_weights(
                        transformer_params,
                        (model.encoder.layer[layer_num].intermediate.dense.weight.t(), 
                        model.encoder.layer[layer_num].intermediate.dense.bias), 
                        "intermediate_dense", layer_num,
        )
        set_bert_layer_mlp_weights(
                        transformer_params,
                        (model.encoder.layer[layer_num].output.dense.weight.t(), 
                        model.encoder.layer[layer_num].output.dense.bias), 
                        "output_dense", layer_num,
        )
        set_bert_layer_layernorm(
                        transformer_params,
                        (model.encoder.layer[layer_num].attention.output.LayerNorm.weight,
                        model.encoder.layer[layer_num].attention.output.LayerNorm.bias),
                        "mha_layernorm",
                        layer_num,
        )
        set_bert_layer_layernorm(
                        transformer_params,
                        (model.encoder.layer[layer_num].output.LayerNorm.weight,
                        model.encoder.layer[layer_num].output.LayerNorm.bias),
                        "mlp_layernorm",
                        layer_num,
        )
    
    # set BertPooler Params
    set_bert_pooler_weights(transformer_params, (model.pooler.dense.weight.t(), model.pooler.dense.bias))

    del model