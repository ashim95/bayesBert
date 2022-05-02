import jax
import scipy
import heapq
import spacy 
import string
 
import numpy as np
import jax.numpy as jnp
import seaborn as sns
import matplotlib.pyplot as plt

from conllu import parse
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

# import from project
from ood_samples import samples

# Load lemmatizer from scipy
lemmatizer = spacy.load('en_core_web_sm')

def data_uncertainty(dataloader, transformer, params, key, configuration, k=9):
    """
    Output the k images the model is most confident
    and unconfident of.
    Evaluate the model.
    :params dataloader: dataloader containing the data.
    :params transformer: model to test.
    :params params: parameters of the model.
    :params key: random key.
    :params configuration: configuration dictionary.
    :params k: number of images to display.
    """
    dataloader.change_split('test')
    dataloader.rebatch_data(1)
    
    entropies = []
    heapq.heapify(entropies)
    for i, (x,y) in enumerate(dataloader):
        key, sub_key = jax.random.split(key)
        logits = transformer.apply(params, key, x, training=False)[0] # (mc_samples, batch, max_len, n_outputs)
        preds = jnp.mean(jax.nn.softmax(logits, axis=-1), axis=0) # (batch, max_len, n_outputs)
        padded_entropies = scipy.stats.entropy(preds, axis=-1)
        non_padded_preds = preds[x != dataloader.vocab['PAD']].reshape(x.shape[0], -1, configuration["n_outputs"])
        if non_padded_preds.shape[1] >= 3:
            non_padded_entropy = scipy.stats.entropy(non_padded_preds, axis=-1).mean(axis=-1)
            heapq.heappush(entropies, (non_padded_entropy, i, 0, padded_entropies))

    k_largest = heapq.nlargest(k, entropies)
    k_smallest = heapq.nsmallest(k, entropies)

    print("K largest entropies:")
    fig, axs = plt.subplots(nrows=k, figsize=(50,k))
    for idx, e in enumerate(k_largest):
        non_padded_entropy, i, j, padded_entropies = e
        text = [dataloader.reverse_vocab[word] for word in dataloader[i][0][j]]
        #sns.heatmap(
        #    data=padded_entropies, vmin=0, vmax=np.log2(17), cmap="coolwarm", annot=np.expand_dims(np.array(text), 0), 
        #    fmt='', linewidths=1, ax=axs[idx], linecolor='black', cbar=False
        #)
        print(f"{idx} - entropy: {non_padded_entropy} - text: {' '.join(text)}")
    #fig.colorbar(ax=axs)
    #fig.tight_layout()
    #plt.savefig(f"{configuration['save_path'][12:-4]}_k_largest_entropies.png")

    print("K smallest entropies:")
    fig, axs = plt.subplots(nrows=k, figsize=(50,k))
    for idx, e in enumerate(k_smallest):
        non_padded_entropy, i, j, padded_entropies = e
        text = [dataloader.reverse_vocab[word] for word in dataloader[i][0][j]]
        #sns.heatmap(
        #    data=padded_entropies, vmin=0, vmax=np.log2(17), cmap="coolwarm", annot=np.expand_dims(np.array(text), 0), 
        #    fmt='', linewidths=1, ax=axs[idx], linecolor='black', cbar=False
        #)
        print(f"{idx} - entropy: {non_padded_entropy} - text: {' '.join(text)}")
    #fig.colorbar(ax=axs)
    #fig.tight_layout()
    #plt.savefig(f"{configuration['save_path'][12:-4]}_k_smallest_entropies.png")


def pred_ambiguity_from_entropy(dataloader, transformer, params, key, configuration):
    """
    Given one ambiguous and non-ambiguous dataset, 
    we compare how predictive of the dataset is the entropy.
    We expect the non-ambiguous dataset to have lower entropy and
    the ambiguous dataset to have high entropy.
    :params dataloader: dataloader containing the data.
    :params transformer: model to test.
    :params params: parameters of the model.
    :params key: random key.
    :params configuration: configuration dictionary.
    """
    dataloader.change_split('test')
    dataloader.rebatch_data(1)

    # Entropy inspection 
    def load_and_process_data(key):
        """
        Encode and pad samples read from the file.
        :params key: a random key.
        """
        # Read file
        file = open("../Data/ambiguous_test.conll", "r")
        raw_data = parse(file.read())
        results = []
        sentence_entropy_log = []
        for sentence in raw_data:
            tag_id = -1
            text, tags, words = [], [], []
            unkn = 0.
            for i, word in enumerate(sentence):
                if word["form"] not in string.punctuation and all(w not in string.digits for w in word["form"]):    
                    lemma = lemmatizer(word["form"].lower())[0].lemma_  
                    if lemma == "-PRON-":
                        continue
                    try:
                        id = dataloader.vocab[lemma]
                    except:
                        # map unknown words to the unknown token
                        id = 0
                        unkn += 1
                    finally:
                        text += [id]
                    if word["upos"] == "_":
                        tags += [dataloader.tag_vocab["X"]] 
                    else:
                        tag_id = i
                        tags += [dataloader.tag_vocab["NOUN"] if word["upos"] == "NN" else dataloader.tag_vocab["VERB"]]
                    words += [word["form"]]
            if len(text) == 0 or tag_id >= dataloader.max_length:
                continue  
            elif tag_id < dataloader.max_length:
                if len(text) <= dataloader.max_length:
                    # pad
                    text += [dataloader.vocab["PAD"]]*(dataloader.max_length-len(text))
                    tags += [dataloader.tag_vocab["X"]]*(dataloader.max_length-len(tags))
                else:
                    text = text[:dataloader.max_length]
                    tags = tags[:dataloader.max_length]
            
            encoded_text = jnp.expand_dims(jnp.array(text), axis=0)

            logits = transformer.apply(params, key, encoded_text, training=False)[0]
            probs = jnp.mean(jax.nn.softmax(logits, axis=-1), axis=0)[:,tag_id]
            entropy = scipy.stats.entropy(probs, axis=-1).mean()
            results += [(entropy, 1)]
            sentence_entropy_log += [(entropy, unkn / len(words), ' '.join(words))]
        
        file.close()

        return results, sentence_entropy_log

    def entropy_inspection(data, model, params, key):
        """
        Compute and return per word entropy mean of each sentence.
        :params data: input data to the model.
        :params model: considered model to compute the output entropy.
        :params params: parameters of the model.
        :params key: random key.
        """
        results = []
        for x, y in data:
            logits = model.apply(params, key, x, training=False)[0] # (mc_samples, batch, max_len, n_outputs)
            preds = jnp.mean(jax.nn.softmax(logits, axis=-1), axis=0) # (batch, max_len, n_outputs)
            preds = preds[jnp.logical_or(y == data.tag_vocab["NOUN"], y == data.tag_vocab["VERB"])].reshape(-1, configuration["n_outputs"])
            for i in range(preds.shape[0]):
                entropy = scipy.stats.entropy(preds[i,:], axis=-1).mean()
                results += [(entropy, 0)]
        
        return results

    ambiguous_results, ambiguous_sentence_entropy = load_and_process_data(key)
    print(len(ambiguous_results))
    data_results = entropy_inspection(dataloader, transformer, params, key)
    print(len(data_results))
    
    # Perform decision tree classification to predict the dataset given the entropy.
    data = jnp.array([(e[0], e[1]) for e in ambiguous_results])
    amb_X, amb_y = data[:,0].reshape(-1, 1), data[:,1]

    data = jnp.array([(e[0], e[1]) for e in data_results])
    norm_X, norm_y = data[:,0].reshape(-1, 1), data[:,1]
    print(amb_X.shape, norm_X.shape)
    X = np.concatenate([amb_X, norm_X], axis=0)
    y = np.concatenate([amb_y, norm_y], axis=0)

    idx = np.random.permutation(len(X))
    X = X[idx]
    y = y[idx]

    classifier = DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2)
    classifier.fit(X, y)

    accuracy = classifier.score(X, y)

    y_pred = classifier.predict(X)
    auroc = roc_auc_score(y, y_pred)

    print('How predicitive of the dataset is given the entropy of a word:')
    print(f"Accuracy on train set: {accuracy}")
    print(f"AUROC on train set: {auroc}")
    print(f"Classifier threshold on entropy feature : {classifier.tree_.threshold}")

    # Perform decision tree classification to examen how predictive the entropy is from the unkn rate
    data = jnp.array([(e[1], e[0]) for e in ambiguous_sentence_entropy])
    X, y = data[:,0].reshape(-1, 1), data[:,1]

    idx = np.random.permutation(len(X))
    X = X[idx]
    y = y[idx]

    regressor = LinearRegression()
    regressor.fit(X, y)

    accuracy = regressor.score(X, y)
    
    print('How predicitive the entropy is given the unkown word rate:')
    print(f"R2 score on train set: {accuracy}")

    # plt.figure(figsize=(10, 10))
    # plt.hist(norm_X.flatten(), bins=200, alpha=0.5, label='test samples', color='b', density=True)
    # plt.hist(amb_X.flatten(), bins=200, alpha=0.5, label='ambiguous samples', color='r', density=True)
    # plt.grid()
    # plt.legend()
    # plt.savefig(f"entropy_split_{configuration['save_path'][12:-4]}.png")


def ood_entropy_split(dataloader, transformer, params, key):
    """
    Experiment in which we study how predicitve of the ambiguouity 
    of a sentence the model entropy is. 
    :params dataloader: dataloader containing the data.
    :params transformer: model to test.
    :params params: parameters of the model.
    :params key: random key.
    """
    # Entropy inspection 
    def encode_and_pad(samples):
        """
        Encode and pad samples given as input.
        :params samples: list of strings.
        """
        data = []
        for e in samples:
            # Unpack e
            sample, label = e
            # Lemmatize the sentences
            sentence = lemmatizer(sample.lower())
            lemmas = [word.lemma_ for word in sentence if word.lemma_ != "-PRON-"]
            # Pad the sentences of lemmas
            lemmas += ["PAD"]*(dataloader.max_length-len(lemmas))
            # Encode the processed text
            encoded_text = jnp.expand_dims(
                jnp.array([dataloader.vocab[word] for word in lemmas]), 
                axis=0
            )
            data += [(sample, encoded_text, label)]
            
        return data

    def entropy_inspection(data, model, params, key):
        """
        Compute and return per word entropy mean of each sentence.
        :params data: input data to the model.
        :params model: considered model to compute the output entropy.
        :params params: parameters of the model.
        :params key: random key.
        """
        results = []
        for e in data:
            text, encoded_text, label = e
            logits = model.apply(params, key, encoded_text, training=False)[0][0,:len(text),:] 
            probs = jnp.mean(jax.nn.softmax(logits, axis=-1), axis=0) 
            entropy = scipy.stats.entropy(probs, axis=-1).mean()
            results += [(entropy, text, label)]
        for (entropy, text, _) in sorted(results):
            print(f"entropy : {entropy} - {text}")
        
        return results

    dataloader.change_split("test")
    dataloader.rebatch_data(1)

    data = encode_and_pad(samples)
    results = entropy_inspection(data, transformer, params, key)

    # Perform decision tree classification
    data = jnp.array([(e[0], e[2]) for e in results])
    X, y = data[:,0].reshape(-1, 1), data[:,1]

    classifier = DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2)
    classifier.fit(X, y)

    accuracy = classifier.score(X, y)

    y_pred = classifier.predict(X)
    auroc = roc_auc_score(y, y_pred)

    print(f"Accuracy on train set: {accuracy}")
    print(f"AUROC on train set: {auroc}")
    print(f"Classifier threshold on entropy feature : {classifier.tree_.threshold}")
    