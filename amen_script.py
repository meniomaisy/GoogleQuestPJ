import pandas as pd
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
from bert_tokenization import bertTokenization as tokenization
import tensorflow.keras.backend as K
from scipy.stats import spearmanr
from math import floor, ceil
from tqdm.notebook import tqdm

np.set_printoptions(suppress=True)

PATH = 'input/Data/'
BERT_PATH = 'input/bert-base-from-tfhub/bert_en_uncased_L-12_H-768_A-12'
tokenizer = tokenization.FullTokenizer(BERT_PATH+'/assets/vocab.txt', True)
MAX_SEQUENCE_LENGTH = 512

def _get_masks(tokens, max_seq_length):
    """Mask for padding"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))

def _get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    segments = []
    first_sep = True
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            if first_sep:
                first_sep = False 
            else:
                current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))

def _get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
    return input_ids

def _trim_input(title, answer, sub_category, q_class, q_keywords, max_sequence_length, 
                t_max_len=150, a_max_len=196, s_c_max_len=25, q_c_max_len=10, q_k_max_len=125):

    t   = tokenizer.tokenize(title)
    a   = tokenizer.tokenize(answer)
    s_c = tokenizer.tokenize(sub_category)
    q_c = tokenizer.tokenize(q_class)
    q_k = tokenizer.tokenize(q_keywords)
    
    t_len   = len(t)
    a_len   = len(a)
    s_c_len = len(s_c)
    q_c_len = len(q_c)
    q_k_len = len(q_k)

  
    if s_c_max_len > s_c_len:
        s_c_new_len = s_c_len
    else:
        s_c_new_len = s_c_max_len

    if q_c_max_len > q_c_len:
        q_c_new_len = q_c_len
    else:
        q_c_new_len = q_c_max_len

    if q_k_max_len > q_k_len:
        q_k_new_len = q_k_len
    else:
        q_k_new_len = q_k_max_len

    if t_max_len > t_len:
        t_new_len = t_len
    else:
        t_new_len = t_max_len

    a_new_len = max_sequence_length - (t_new_len+s_c_new_len+q_c_new_len+q_k_new_len+6)
            
    if t_new_len+s_c_new_len+a_new_len+q_c_new_len+q_k_new_len+6 != max_sequence_length:
        raise ValueError("New sequence length should be %d, but is %d" 
                         % (max_sequence_length, (t_new_len+s_c_new_len+a_new_len+q_c_new_len+q_k_new_len+4)))
    
    a   = a[:a_new_len]
    t   = t[:t_new_len]
    s_c = s_c[:s_c_new_len]
    q_c = q_c[:q_c_new_len]
    q_k = q_k[:q_k_new_len]
    
    return t, a, s_c, q_c, q_k

def _convert_to_bert_inputs(title, answer, sub_category, q_class, q_keywords, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for BERT"""
    
    stoken = ["[CLS]"] + title + ["[SEP]"] + answer + ["[SEP]"] + sub_category + ["[SEP]"] + q_class + ["[SEP]"] + q_keywords + ["[SEP]"]
    
    input_ids = _get_ids(stoken, tokenizer, max_sequence_length)
    input_masks = _get_masks(stoken, max_sequence_length)
    input_segments = _get_segments(stoken, max_sequence_length)

    return [input_ids, input_masks, input_segments]

def compute_input_arays(df, columns, tokenizer, max_sequence_length):
    input_ids, input_masks, input_segments = [], [], []

    for _, instance in tqdm(df[columns].iterrows()):
        t, a, s_c, q_c, q_k = instance.question_title, instance.answer, instance.sub_category, instance.q_class, instance.q_keywords

        t, a, s_c, q_c, q_k = _trim_input(t, a, s_c, q_c, q_k, max_sequence_length)
       
       
        ids, masks, segments = _convert_to_bert_inputs(t, a, s_c, q_c, q_k, tokenizer, max_sequence_length)
        
        input_ids.append(ids)
        input_masks.append(masks)
        input_segments.append(segments)
        
    return [np.asarray(input_ids, dtype=np.int32), 
            np.asarray(input_masks, dtype=np.int32), 
            np.asarray(input_segments, dtype=np.int32)]


def compute_output_arrays(df, columns):
    return np.asarray(df[columns])

def compute_spearmanr(trues, preds):
    rhos = []
    for col_trues, col_pred in zip(trues.T, preds.T):
        rhos.append(
            spearmanr(col_trues, col_pred + np.random.normal(0, 1e-7, col_pred.shape[0])).correlation)
    return np.mean(rhos)


class CustomCallback(tf.keras.callbacks.Callback):
    
    def __init__(self, valid_data, test_data, batch_size=16, fold=None):

        self.valid_inputs = valid_data[0]
        self.valid_outputs = valid_data[1]
        self.test_inputs = test_data
        
        self.batch_size = batch_size
        self.fold = fold
        
    def on_train_begin(self, logs={}):
        self.valid_predictions = []
        self.test_predictions = []
        
    def on_epoch_end(self, epoch, logs={}):
        self.valid_predictions.append(
            self.model.predict(self.valid_inputs, batch_size=self.batch_size))
        
        rho_val = compute_spearmanr(
            self.valid_outputs, np.average(self.valid_predictions, axis=0))
        
        print("\nvalidation rho: %.4f" % rho_val)
        
        if self.fold is not None:
            self.model.save_weights(f'bert-base-{fold}-{epoch}.h5py')
        
        self.test_predictions.append(
            self.model.predict(self.test_inputs, batch_size=self.batch_size)
        )

def bert_model():
    
    input_word_ids = tf.keras.layers.Input(
        (MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_word_ids')
    input_masks = tf.keras.layers.Input(
        (MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_masks')
    input_segments = tf.keras.layers.Input(
        (MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_segments')
    
    bert_layer = hub.KerasLayer(BERT_PATH, trainable=True)
    
    _, sequence_output = bert_layer([input_word_ids, input_masks, input_segments])
    
    x = tf.keras.layers.GlobalAveragePooling1D()(sequence_output)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(30, activation="sigmoid", name="dense_output")(x)

    model = tf.keras.models.Model(
        inputs=[input_word_ids, input_masks, input_segments], outputs=out)
    
    return model    
        
def train_and_predict(model, train_data, valid_data, test_data, 
                      learning_rate, epochs, batch_size, loss_function, fold):
        
    custom_callback = CustomCallback(
        valid_data=(valid_data[0], valid_data[1]), 
        test_data=test_data,
        batch_size=batch_size,
        fold=None)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=loss_function, optimizer=optimizer)
    model.fit(train_data[0], train_data[1], epochs=epochs, 
              batch_size=batch_size, callbacks=[custom_callback])
    
    return custom_callback
