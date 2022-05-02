import jax
import jax.numpy as jnp
import numpy as np

from sklearn.metrics import f1_score

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

def test_ce_acc_ece(test_data, model, params, key):
    """
    Compute the test cross-entropy, accuracy and expected 
    calibration error of the model.
    :params test_data: data on which to compute scores.
    :params model: model to score.
    :params params: parameters of the model.
    :params key: random key.
    """
    ce_loss, accuracy, ece_mean, n = 0., 0., 0., 0.

    for x, y in test_data:
        # Forward pass : (mc_samples, batch, max_len, n_outputs)
        logits = model.apply(params, key, x, training=False)[0]  
        preds = jnp.mean(jax.nn.softmax(logits, axis=-1), axis=0) # (batch, max_len, n_outputs)
        # Test cross-entropy
        labels = jax.nn.one_hot(y, logits.shape[-1])
        ce_loss += -jnp.sum(labels * jnp.mean(jax.nn.log_softmax(logits, -1), axis=0))
        # Test accuracy
        accuracy += (jnp.argmax(preds, -1) == y).sum()
        # Test ece 
        ece_mean += jax.vmap(ece, in_axes=(0,0))(
            preds.transpose((1,0,2)), y.transpose((1,0))
        ).sum()
        n += x.shape[1]
    accuracy /= np.prod(test_data.test["text"].shape[:3])
    ce_loss /= np.prod(test_data.test["text"].shape[:2])
    ece_mean /= n

    return jnp.array([accuracy, ce_loss, ece_mean])

def test_f1_score(test_data, model, params, key, configuration, n):
    """
    Compute n passes of the f1 score of the model. 
    :params test_data: data on which to compute scores.
    :params model: model to score.
    :params params: parameters of the model.
    :params key: random key.
    :params configuration: configuration dictionary.
    :params n: number of data passes. 
    """
    f1 = []
    for _ in range(n):
        y_preds, y_true = [], []
        for x, y in test_data:
            # Forward pass : (mc_samples, batch, max_len, n_outputs)
            logits = model.apply(params, key, x, training=False)[0]
            preds = jnp.mean(jax.nn.softmax(logits, axis=-1), axis=0)
            y_preds += [np.argmax(preds, -1).reshape(-1)]
            y_true += [y.reshape(-1)]
        y_preds = np.concatenate(y_preds)
        y_true = np.concatenate(y_true)
        f1 += [f1_score(y_true, y_preds, average='weighted', labels=list(range(configuration["n_outputs"])))]

    return np.mean(f1), np.std(f1)

def evaluate(dataloader, model, params, key, configuration, n):
    """
    Evaluate the model.
    :params dataloader: dataloader containing the data.
    :params model: model to score.
    :params params: parameters of the model.
    :params key: random key.
    :params configuration: configuration dictionary.
    :params n: number of data passes. 
    """
    # Get the test split of the data
    dataloader.change_split("test")

    # Split the keys
    keys = jax.random.split(key, n)
    
    # Compute log-likelihood, accuracy and ECE scores on test data
    losses = jax.vmap(test_ce_acc_ece, in_axes=(None, None, None, 0))(dataloader, model, params, keys)

    acc_mean, ce_mean, ece_mean = losses.mean(axis=0)
    acc_std, ce_std, ece_std = losses.std(axis=0)

    # Compute f1 score on test data.
    mean_f1, std_f1 = test_f1_score(dataloader, model, params, key, configuration, n)
    
    print(f"Cross-entropy loss: {ce_mean:.4f} $\pm$ {ce_std:.4f}")
    print(f"Accuracy: {acc_mean:.4f} $\pm$ {acc_std:.4f}")
    print(f"ECE mean: {ece_mean:.4f} $\pm$ {ece_std:.4f}")
    print(f"F1: {mean_f1:.4f} $\pm$ {std_f1:.4f}")

