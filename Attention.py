'''
Created on Jan 5, 2018

@author: purbasha
'''
import numpy as np
import keras.backend as K
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import merge, Merge
from keras.models import Model
from keras.layers.convolutional import Conv2D
from keras.optimizers import SGD, Adagrad, Adam
from keras.layers import Activation, Input, Embedding, Dropout, GlobalMaxPooling1D, Flatten
from keras.layers.core import Dense, Lambda, Reshape, Permute
from keras.engine.topology import Layer
import tensorflow as tf 

emb_dim = 20
MAX_NB_WORDS = 40000
MAX_SEQUENCE_LENGTH = 60
print('Indexing word vectors.')

texts = []  # list of text samples
filenameTrain = 'SemEvalTrain.csv'
filereader_Train = pd.read_csv(filenameTrain, sep=',')

filenameDev = 'SemEvalDev.csv'
filereader_Dev = pd.read_csv(filenameDev, sep=',')

filenameTest = 'SemEvalTest.csv'
filereader_Test = pd.read_csv(filenameDev, sep=',')

filereader_Train.columns = ['QuesID','AnsID','QuesSubject','QuesBody', 'Answers','Score'] 
filereader_Dev.columns = ['QuesID','AnsID','QuesSubject','QuesBody', 'Answers','Score']         
filereader_Test.columns = ['QuesID','AnsID','QuesSubject','QuesBody', 'Answers','Score'] 

def normScore(filereader):
    '''
    Normalizing the score and creating lists of comments,topic and labels
    '''
    filereader["QuesBody"] = np.where(filereader["QuesBody"].isnull()==True ,"NA", filereader["QuesBody"])
    filereader["QuesSubject"] = np.where(filereader["QuesSubject"].isnull()==True ,"NA", filereader["QuesSubject"])
    filereader["Score"] = np.where(filereader["Score"]=="Bad" ,0, filereader["Score"])
    filereader["Score"] = np.where(filereader["Score"]=="Good" ,1, filereader["Score"])
    filereader["Score"] = np.where(filereader["Score"]=="PotentiallyUseful" ,2, filereader["Score"])
    txt_cmt = filereader["Answers"].tolist()
    txt_topic_sub = filereader["QuesSubject"].tolist() 
    txt_topic_body = filereader["QuesBody"].tolist()         
    txt_label = filereader["Score"].tolist()
    return txt_cmt, txt_topic_sub, txt_topic_body, txt_label 


tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
def tokenizing(filereader):  
    txt_cmt, txt_topic_sub, txt_topic_body, txt_label = normScore(filereader) 
    tokenizer.fit_on_texts(txt_cmt)
    sequences_cmt = tokenizer.texts_to_sequences(txt_cmt)
    tokenizer.fit_on_texts(txt_topic_sub)
    sequences_topic_sub = tokenizer.texts_to_sequences(txt_topic_sub)
    tokenizer.fit_on_texts(txt_topic_body)
    sequences_topic_body = tokenizer.texts_to_sequences(txt_topic_body)
    labels = np.array(txt_label)
    return sequences_cmt, sequences_topic_sub, sequences_topic_body, labels

## Train Data
sequences_cmt, sequences_topic_sub, sequences_topic_body, labels = tokenizing(filereader_Train) 
data_cmt = pad_sequences(sequences_cmt, maxlen=MAX_SEQUENCE_LENGTH)
data_topic_sub = pad_sequences(sequences_topic_sub, maxlen=10)
data_topic_body = pad_sequences(sequences_topic_body, maxlen=MAX_SEQUENCE_LENGTH)

## Dev Data
sequences_cmt_dev, sequences_topic_sub_dev, sequences_topic_body_dev, labels_dev = tokenizing(filereader_Dev)
data_cmt_dev = pad_sequences(sequences_cmt_dev, maxlen=MAX_SEQUENCE_LENGTH)
data_topic_sub_dev = pad_sequences(sequences_topic_sub_dev, maxlen=10)
data_topic_body_dev = pad_sequences(sequences_topic_body_dev, maxlen=MAX_SEQUENCE_LENGTH)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

## Test Data
sequences_cmtTest, sequences_topic_subTest, sequences_topic_bodyTest, labelsTest = tokenizing(filereader_Test)
data_cmt_test = pad_sequences(sequences_cmtTest, maxlen=MAX_SEQUENCE_LENGTH)
data_topic_sub_test = pad_sequences(sequences_topic_subTest, maxlen=10)
data_topic_body_test = pad_sequences(sequences_topic_bodyTest, maxlen=MAX_SEQUENCE_LENGTH)

from keras.utils import np_utils
score_train = np_utils.to_categorical(labels)
score_val = np_utils.to_categorical(labels_dev)
score_test = np_utils.to_categorical(labelsTest)
    
class QACNN(Layer):
    
    def __init__(self, vocab_size=MAX_NB_WORDS, embedding_size=emb_dim,filter_sizes=[1,2,3], num_filters=400, dropout_keep_prob=1.0,paras=None,learning_rate=1e-2,embeddings=None,trainable=True, **kwargs):
        self.learning_rate=learning_rate
        self.paras=paras
        self.filter_sizes=filter_sizes
        self.num_filters=num_filters
        self.dropout_keep_prob = dropout_keep_prob
        self.embeddings=embeddings


        self.embedding_size=embedding_size
        self.model_type="base"
        self.num_filters_total=self.num_filters * len(self.filter_sizes)        
        
        # Embedding layer
        self.updated_paras=[]
        with tf.name_scope("embedding"):
            if self.paras==None:
                if self.embeddings ==None:
                    print ("random embedding")
                    self.Embedding_W = tf.Variable(
                        tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                        name="random_W")
                else:
                    self.Embedding_W = tf.Variable(np.array(self.embeddings),name="embedding_W" ,dtype="float32",trainable=trainable)
            else:
                print ("load embeddings")
                self.Embedding_W=tf.Variable(self.paras[0],trainable=trainable,name="embedding_W")
            self.updated_paras.append(self.Embedding_W)
        super(QACNN, self).__init__(** kwargs)
        
        
    def build(self, input_shape): 
            self.kernels=[]  
            self.input_dim = input_shape[-1]  
            for i, filter_size in enumerate(self.filter_sizes):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                    if self.paras==None:
                        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="kernel_W")
                        b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="kernel_b")
                        self.kernels.append((W,b))
                    else:
                        _W,_b=self.paras[1][i]
                        W=tf.Variable(_W)                
                        b=tf.Variable(_b)
                        self.kernels.append((W,b))   
                    self.updated_paras.append(W)
                    self.updated_paras.append(b)

    def compute_mask(self, inputs, mask=None):
        mask = super(QACNN, self).compute_mask(inputs, mask)
        return mask                

    def call(self,sentence, mask=None):
        embedded_chars_1 = tf.nn.embedding_lookup(self.Embedding_W, sentence)
        print "embedded::",embedded_chars_1
        embedded_chars_expanded_1 = tf.expand_dims(embedded_chars_1, -1)
        print "expanded embedded::",embedded_chars_expanded_1
        output=[]
        for i, filter_size in enumerate(self.filter_sizes): 
            conv = tf.nn.conv2d(
                embedded_chars_expanded_1,
                self.kernels[i][0],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="conv-1"
            )
            print "conv::",conv
            h = tf.nn.relu(tf.nn.bias_add(conv, self.kernels[i][1]), name="relu-1")
            print "relu::",h
            output.append(h)
        tf_reshape = tf.reshape(conv,[-1,int(h.shape[1]), int(h.shape[3])])    
        self.output_dim = [int(tf_reshape.shape[1]), int(tf_reshape.shape[2])]
        return tf_reshape

    def compute_output_shape(self, input_shape):
        newShape = list(input_shape)
        newShape[1] = self.output_dim[0]
        newShape.append(self.output_dim[1])
        return tuple(newShape)


sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

qns_cnn = QACNN()(sequence_1_input)
ans_cnn = QACNN()(sequence_2_input)
print qns_cnn, ans_cnn 

## calculating Similarity
qns_perm = Permute((2,1))(qns_cnn)
dens_qns = Dense(58, kernel_initializer='random_uniform')(qns_perm)
ans_perm = Permute((2,1))(ans_cnn)
qns_ans_dens = merge([dens_qns, ans_perm], mode='dot', dot_axes=1)
print qns_ans_dens

## Column max pooling
col_max = Permute((2,1))(qns_ans_dens)
col_max = GlobalMaxPooling1D()(col_max)
print col_max

## Row-max pooling
row_max = GlobalMaxPooling1D()(qns_ans_dens)
print row_max

## Attention
soft_col = Activation('softmax')(col_max)
soft_row = Activation('softmax')(row_max)

## Final Representation
qns_soft = merge([qns_cnn, soft_col], mode='dot', dot_axes=1)
ans_soft = merge([ans_cnn, soft_row], mode='dot', dot_axes=1)

## Cosine similarity
qns_ans_tensors = merge([qns_soft, ans_soft], mode='cos', dot_axes=-1)
dist = Lambda(lambda x: 1-x)(qns_ans_tensors)
dist = Dense(3,activation = 'softmax')(dist)

## Model Training
model = Model(inputs = [sequence_1_input, sequence_2_input], outputs = dist)    
print model.summary()
adagrad = Adagrad(lr = 0.1)
sgd = SGD()
adam = Adam(lr = 0.01)
model.compile(optimizer=adagrad, loss='categorical_crossentropy', metrics=['acc'])

model.fit([data_topic_body,data_cmt], score_train,
                       validation_data=[[data_topic_body_dev,data_cmt_dev], score_val],
                       epochs=80, batch_size=64)    
      