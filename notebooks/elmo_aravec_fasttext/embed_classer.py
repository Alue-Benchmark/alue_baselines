import gensim 
import numpy as np
import fasttext
from elmoformanylangs import Embedder

class embed(object):
    def __init__(self, source, path):
        self.source = source
        self.path = path
        self._load_model()
    def _aravec(self, path):
        self.model = gensim.models.Word2Vec.load(path)
        self.vector_size = self.model.vector_size
    def _fasttext(self, path):
        self.model = fasttext.load_model(path)
        self.vector_size = self.model.get_dimension()
    def _elmo(self, path):
        self.model = Embedder(path, 64)
        
    def _load_model(self):
        if self.source == "aravec":
            self._aravec(self.path)
        elif self.source == "fasttext":
            self._fasttext(self.path)
        elif self.source == "elmo":
            self._elmo(self.path)
        else:
            raise ValueError("Model not supported. Please select either aravec, fasttext or elmo")
   
    def _embed_single(self, text, max_len):
        if self.source == "aravec":
            embedding = [self.model.wv[i].reshape(-1, self.vector_size) for i in text.split() if i in self.model.wv]
            if len(embedding) == 0:
                return self._pad(np.zeros((1, self.vector_size)), max_len)
            embedding = np.concatenate(embedding, axis=0)
            return self._pad(embedding, max_len)
        if self.source == "fasttext":
            embedding = [self.model.get_word_vector(i).reshape(-1, self.vector_size) for i in text.split()]
            embedding = np.concatenate(embedding, axis=0)
            return self._pad(embedding, max_len)
                    
    def embed_batch(self, text_list, max_len):
        if self.source == "elmo":
            input_segmented = [i.split() for i in text_list]
            embedding = self.model.sents2elmo(input_segmented)
            embedding = [self._pad(i, max_len) for i in embedding]            
            return np.concatenate(embedding, axis=0)
        else:    
            batch = [self._embed_single(i, max_len) for i in text_list]
            return np.concatenate(batch)
    
    def _pad(self, array, max_len):
        if array.shape[0] >= max_len:
            return np.expand_dims(array[:max_len],0)
        else:
            padding_size = max_len - array.shape[0]
            return np.expand_dims(np.pad(array, [(0, padding_size), (0, 0)], mode='constant', constant_values=0), 0)
