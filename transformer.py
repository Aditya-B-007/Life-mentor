import numpy as np
import pandas as pd

"""## ***Encoder Block***

Word embeddings
"""

encode={}
decode={}
max_value=50

keys=np.random.rand(50,50)
queries=np.random.rand(50,50)
values=np.random.rand(50,50)

def layer_norm_numpy(x, epsilon=1e-6):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    normalized_x = (x - mean) / np.sqrt(variance + epsilon)
    return normalized_x

class encoder():
  def __init__(self,string,flag):
    self.string=string
    self.encoder(self.string)
  def get_positional_encoding(seq_length, d_model):
    positional_encoding = np.zeros((seq_length, d_model))
    for pos in range(seq_length):
        for i in range(0, d_model, 2):
            denominator = np.power(10000, 2 * i / d_model)
            positional_encoding[pos, i] = np.sin(pos / denominator)
            if i + 1 < d_model:
                positional_encoding[pos, i + 1] = np.cos(pos / denominator)
    return positional_encoding
  def encoder(self,string):
    global max_value
    cleaned_string=''.join(character for character in string if character.isalnum())
    cleaned_string=cleaned_string.lower()
    word_list=cleaned_string.split()
    global encode
    global decode
    for index,word in enumerate(word_list):
      if word not in encode:
        encode[word]=np.array([index])
        continue
      encode[word]=np.append(encode[word],index)
      encode[word]=np.unique(encode[word])
      current_length = len(encode[word])
      if current_length<max_value:
        encode[word]=np.pad(encode[word],(0,max_value-current_length))
      else:
        encode[word]=encode[word][:max_value]
    encode=self.addition(encode)
    for keys, values in encode.items():
      decode[values]=keys
  def addition(self,encode):
    for keys,values in encode.items():
      new_embedding=np.add(values,self.get_positional_encoding(50,50))
      encode[keys]=new_embedding
    return encode

#Self Attention object creation
class self_attention():
  def __init__(self,matrix):
    self.matrix=matrix
    self.self_attention(self.matrix)
  def self_attention(self,matrix):
    Q=np.dot(matrix,queries)
    K=np.dot(matrix,keys)
    V=np.dot(matrix,values)
    scores = np.dot(Q, K.T)
    d_k = K.shape[1]
    scaled_scores = scores / np.sqrt(d_k)
    exp_scores = np.exp(scaled_scores)
    attention_weights = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    output = np.dot(attention_weights, V)
    return output

class decoder():
  def __init__(self,vector,string):
    self.string=string
    self.decoder(vector,self.string)
  def cosine_similarity(self,vector1,vector2):
    return np.dot(vector1,vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2)) if np.linalg.norm(vector1)*np.linalg.norm(vector2)!=0 else 0
  def decoder(self,vector,string):
    max_value=-float('inf')
    to_be_added=""
    for keys,values in decode.items:
      if self.cosine_similarity(vector,values)>max_value:
        max_value=self.cosine_similarity(vector,values)
        to_be_added=keys
    return string+to_be_added

class Feed_Forward_Network():
    def __init__(self):
        self.input_size = 50
        self.hidden_size1 = 64
        self.hidden_size2 = 32
        self.hidden_size3 = 16
        self.hidden_size4 = 8
        self.output_size = 50
        self.W1 = np.random.randn(self.input_size, self.hidden_size1)
        self.b1 = np.ones((1, self.hidden_size1))
        self.W2 = np.random.randn(self.hidden_size1, self.hidden_size2)
        self.b2 = np.ones((1, self.hidden_size2))
        self.W3 = np.random.randn(self.hidden_size2, self.hidden_size3)
        self.b3 = np.ones((1, self.hidden_size3))
        self.W4 = np.random.randn(self.hidden_size3, self.hidden_size4)
        self.b4 = np.ones((1, self.hidden_size4))
        self.W5 = np.random.randn(self.hidden_size4, self.output_size)
        self.b5 = np.ones((1, self.output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        residual = X
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = self.sigmoid(self.z3)
        self.z4 = np.dot(self.a3, self.W4) + self.b4
        self.a4 = self.sigmoid(self.z4)
        self.z5 = np.dot(self.a4, self.W5) + self.b5
        output_before_norm = self.z5
        output = self.add_and_norm(residual, output_before_norm)
        return output

    def add_and_norm(self, x, sublayer_output):
        added_output = x + sublayer_output
        norm_output = layer_norm_numpy(added_output)
        return norm_output

    def back_prop(self, X, y, output, learning_rate=0.01):
        m = X.shape[0]
        self.error = y - output
        self.delta_output = self.error / m
        self.W5_grad = np.dot(self.a4.T, self.delta_output)
        self.b5_grad = np.sum(self.delta_output, axis=0, keepdims=True)
        self.error_hidden4 = np.dot(self.delta_output, self.W5.T)
        self.delta_hidden4 = self.error_hidden4 * self.sigmoid_derivative(self.a4)
        self.W4_grad = np.dot(self.a3.T, self.delta_hidden4)
        self.b4_grad = np.sum(self.delta_hidden4, axis=0, keepdims=True)
        self.error_hidden3 = np.dot(self.delta_hidden4, self.W4.T)
        self.delta_hidden3 = self.error_hidden3 * self.sigmoid_derivative(self.a3)
        self.W3_grad = np.dot(self.a2.T, self.delta_hidden3)
        self.b3_grad = np.sum(self.delta_hidden3, axis=0, keepdims=True)
        self.error_hidden2 = np.dot(self.delta_hidden3, self.W3.T)
        self.delta_hidden2 = self.error_hidden2 * self.sigmoid_derivative(self.a2)
        self.W2_grad = np.dot(self.a1.T, self.delta_hidden2)
        self.b2_grad = np.sum(self.delta_hidden2, axis=0, keepdims=True)
        self.error_hidden1 = np.dot(self.delta_hidden2, self.W2.T)
        self.delta_hidden1 = self.error_hidden1 * self.sigmoid_derivative(self.a1)
        self.W1_grad = np.dot(X.T, self.delta_hidden1)
        self.b1_grad = np.sum(self.delta_hidden1, axis=0, keepdims=True)
        self.W1 -= learning_rate * self.W1_grad
        self.b1 -= learning_rate * self.b1_grad
        self.W2 -= learning_rate * self.W2_grad
        self.b2 -= learning_rate * self.b2_grad
        self.W3 -= learning_rate * self.W3_grad
        self.b3 -= learning_rate * self.b3_grad
        self.W4 -= learning_rate * self.W4_grad
        self.b4 -= learning_rate * self.b4_grad
        self.W5 -= learning_rate * self.W5_grad
        self.b5 -= learning_rate * self.b5_grad







