import jax
import string
import numpy as np
from conllu import parse
from glob import glob

class DataLoader():
    """
    Object which loads, pre-processes and batches the data.
    """
    def __init__(self, configuration):
        """
        Build DataLoader.
        :params configuration: configuration dictionary.
        """
        self.config = configuration
        # Sanitize input
        languages = [language[38:] for language in glob("../Data/universal-dependencies-1.2/*")]
        if self.config["language"] not in languages:
            raise Exception(f'Language must be chosen among {languages}')
        # Model params
        self.batch_size = self.config["batch_size"]
        self.max_length = self.config["max_length"]
        self.split = self.config["split"]
        self.skip = 5 # map words which appear up to skip times to its occurence
        self.tag_vocab = {
            "ADJ": 0, "ADP": 1, "ADV": 2, "AUX": 3, "CONJ": 4, 
            "DET": 5, "INTJ": 6, "NOUN": 7, "NUM": 8, "PART": 9, 
            "PRON": 10, "PROPN": 11, "PUNCT": 12, "SCONJ": 13, 
            "SYM": 14, "VERB": 15, "X": 16
        } # CONJ == CCONJ
        # Load data
        train_data, dev_data, test_data = self._load_data()
        # Tags & text
        self.train, self.dev, self.test = {}, {}, {}
        self.train["text"], self.train["tags"], train_sentence_len = self._get_tags_and_text(train_data)
        self.dev["text"], self.dev["tags"], dev_sentence_len = self._get_tags_and_text(dev_data)
        self.test["text"], self.test["tags"], test_sentence_len = self._get_tags_and_text(test_data)
        self.sentence_len = train_sentence_len + dev_sentence_len + test_sentence_len
        # Fetch vocabulary
        train_word_occur = self._get_word_occurences(self.train["text"])
        dev_word_occur = self._get_word_occurences(self.dev["text"])
        test_word_occur = self._get_word_occurences(self.test["text"])
        self.vocab, self.word_occur = self._get_vocabulary(train_word_occur, dev_word_occur, test_word_occur)
        self.reverse_vocab = {
            (v if v > self.skip else (v-1)):(k if v > self.skip else f"UKN{v}") 
            for k,v in self.vocab.items()
        }
        self.vocab_size = len(set(self.vocab.values()))
        # Encode text
        self.train["text"] = self._encode_text(self.train["text"])
        self.dev["text"] = self._encode_text(self.dev["text"])
        self.test["text"] = self._encode_text(self.test["text"])
        # Encode tags
        self.train["tags"] = self._encode_tags(self.train["tags"])
        self.dev["tags"] = self._encode_tags(self.dev["tags"])
        self.test["tags"] = self._encode_tags(self.test["tags"])
        # Batch 
        self.train["text"], self.train["tags"] = self._batch_data(self.train["text"], self.train["tags"])
        self.dev["text"], self.dev["tags"] = self._batch_data(self.dev["text"], self.dev["tags"])
        self.test["text"], self.test["tags"] = self._batch_data(self.test["text"], self.test["tags"])
        # Iterable init
        self.index = 0
        # Statistics 
        print(f"Number of different tokens: {self.vocab_size}", flush=True)
        print(f"Number of different words: {len(self.vocab.values())}", flush=True)
        print(f"Number of training samples: {len(train_sentence_len)}", flush=True)
        print(f"Number of validation samples: {len(dev_sentence_len)}", flush=True)
        print(f"Number of test samples: {len(test_sentence_len)}", flush=True)
    
    def _load_data(self):
        """
        Read data from file.
        """
        path = "../Data/universal-dependencies-1.2/UD_" + self.config["language"] + "/*"
        for file_path in glob(path):
            file = open(file_path, "r")
            if "train" in file_path:
                train_data = parse(file.read())
            elif "dev" in file_path:
                dev_data = parse(file.read())
            elif "test" in file_path:
                test_data = parse(file.read())
        
        return train_data, dev_data, test_data
    
    def _get_tags_and_text(self, data):
        """
        Extract text & tags.
        :params data: list of TokenList.
        """
        text_list, tag_list, sentence_len = [], [], []
        for sentence in data:
            text, tags = [], []
            sentence_len += [len(sentence)]
            for word in sentence:
                # Underscore _ is used to denote unspecified values
                if word["form"] != "_" and word["form"] not in string.punctuation and all(w not in string.digits for w in word["form"]):        
                    text += [word["lemma"]]
                    tags += [word["upos"]]
            if len(text) == 0:
              continue  
            elif len(text) <= self.max_length:
                # Pad
                text_list += [text + ["PAD"]*(self.max_length-len(text))] 
                tag_list += [tags + ["X"]*(self.max_length-len(tags))]
            else: 
                # Split
                n = len(text) // self.max_length
                for i in range(n):
                    text_list += [text[i*self.max_length:(i+1)*self.max_length]]
                    tag_list += [tags[i*self.max_length:(i+1)*self.max_length]]
                # Pad the remains
                text_list += [text[n*self.max_length:] + ["PAD"]*(self.max_length-len(text[n*self.max_length:]))] 
                tag_list += [tags[n*self.max_length:] + ["X"]*(self.max_length-len(text[n*self.max_length:]))]

        return text_list, tag_list, sentence_len

    def _update_dict(self, d, v, n=1):
        """
        Increment element count dictionary by n.
        :params d: dictionary to update.
        :params v: key.
        :params n: amount by which to increment d[v].
        """
        d[v] = d[v] + n if v in d else n

        return d
    
    def _get_word_occurences(self, text):
        """
        Return the number of times each word occurs in our text.
        :params text: list of list containing words.
        """
        word_occurences = {}
        # Populate words and lengths
        for sentence in text:
            for word in sentence:
                word_occurences = self._update_dict(word_occurences, word)
        return word_occurences
            
    def _get_vocabulary(self, w1, w2, w3):
        """
        Return text vocabulary and word occurences.
        :params w1: dictionary mapping words to their number of occurences in train set.
        :params w2: dictionary mapping words to their number of occurences in dev set.
        :params w3: dictionary mapping words to their number of occurences in test set.
        """
        vocab, words = {}, {}
        # Populate words and lengths
        for w in (w1, w2, w3):
            for k,v in w.items():
                words = self._update_dict(words, k, n=v)
        # Map all words for which # of occurences <= skip, to its # of occurences
        next = self.skip
        for k,v in words.items():
            if v <= self.skip:
                vocab[k] = v-1
            else:
                vocab[k] = next 
                next += 1

        return vocab, words

    def _encode_text(self, text_list):
        """
        Encode words into token ids.
        :params text_list: list of list containing words.
        """
        text = []
        for sentence in text_list:
            encoded_sentence = []
            for word in sentence:
                if word in self.vocab:
                    encoded_sentence += [self.vocab[word]]
            text += [encoded_sentence]
        
        return np.array(text)

    def _encode_tags(self, tag_list):
        """
        Encode tags into class ids.
        :params tag_list: list of list containing tags.
        """
        tags = []
        for sentence in tag_list:
            encoded_sentence_tags = []
            for word in sentence:
                if word in self.tag_vocab:
                    encoded_sentence_tags += [self.tag_vocab[word]]
            tags += [encoded_sentence_tags]
        
        return np.array(tags)   

    def _batch_data(self, text, tags):
        """
        Split data into batches.
        :params text: text data tensor.
        :params tags: tag data tensor.
        """
        n = text.shape[0] // self.batch_size
        b_text = text[:n*self.batch_size].reshape(n, self.batch_size, self.max_length)
        b_tags = tags[:n*self.batch_size].reshape(n, self.batch_size, self.max_length)
        
        return b_text, b_tags

    def __getitem__(self, index):
        """
        Return batch of data at index conditioned on split.
        :params index: index of data to be returned.
        """
        if self.split == "train":
            return self.train["text"][index], self.train["tags"][index]
        elif self.split == "dev":
            return self.dev["text"][index], self.dev["tags"][index]
        elif self.split == "test":
            return self.test["text"][index], self.test["tags"][index]
    
    def __len__(self):
        """
        Return length of data conditioned on split.
        """
        if self.split == "train":
            return self.train["text"].shape[0]
        elif self.split == "dev":
            return self.dev["text"].shape[0]
        elif self.split == "test":
            return self.test["text"].shape[0]

    def __iter__(self):
        """
        Return an iterator on the selected data split.
        """ 
        return self

    def __next__(self): 
        """
        When using an iterator, return next element.
        """
        if self.index < self.__len__():
            x, y = self[self.index]
            self.index += 1
            return x, y
        self.index = 0
        raise StopIteration

    def change_split(self, new_split):
        """
        Change the data split among train, dev and test.
        :params new_split: new data split.
        """
        assert new_split in ["train", "dev", "test"]
        self.split = new_split

    def rebatch_data(self, batch_size):
        """
        Change the batch size of the data.
        :params batch_size: new batch size
        """
        self.batch_size = batch_size

        n = np.prod(self.train["text"].shape[:2]) // batch_size
        self.train["text"] = self.train["text"][:n*batch_size].reshape(n, batch_size, self.max_length)
        self.train["tags"] = self.train["tags"][:n*batch_size].reshape(n, batch_size, self.max_length)
        
        n = np.prod(self.dev["text"].shape[:2]) // batch_size
        self.dev["text"] = self.dev["text"][:n*batch_size].reshape(n, batch_size, self.max_length)
        self.dev["tags"] = self.dev["tags"][:n*batch_size].reshape(n, batch_size, self.max_length)

        n = np.prod(self.test["text"].shape[:2]) // batch_size
        self.test["text"] = self.test["text"][:n*batch_size].reshape(n, batch_size, self.max_length)
        self.test["tags"] = self.test["tags"][:n*batch_size].reshape(n, batch_size, self.max_length)
    
    def shuffle(self, key):
        """
        Shuffle data.
        :params key: random key.
        """
        # Flatten data
        batch_size = self.batch_size
        self.rebatch_data(1)

        # Shuffle train data
        p = jax.random.permutation(key, self.train["text"].shape[0])
        self.train["text"] = self.train["text"][p]
        self.train["tags"] = self.train["tags"][p]

        # Shuffle val data
        p = jax.random.permutation(key, self.dev["text"].shape[0])
        self.dev["text"] = self.dev["text"][p]
        self.dev["tags"] = self.dev["tags"][p]

        # Shuffle test data
        p = jax.random.permutation(key, self.test["text"].shape[0])
        self.test["text"] = self.test["text"][p]
        self.test["tags"] = self.test["tags"][p]

        # Rebatch the data
        self.rebatch_data(batch_size)
