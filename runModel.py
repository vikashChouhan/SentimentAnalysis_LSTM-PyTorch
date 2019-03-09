import string
import lstmModel
import sys
import numpy as np
import pickle

try:
    global importSucces
    import torch
    import torch.nn as nn
    importSuccess = 1
except:
    print('Error ')
    print('Make sure you have installed torch module.')

#padding text
def pad_features(reviews_ints, seq_length):
    ''' Return features of review_ints, where each review is padded with 0's
        or truncated to the input seq_length.
    '''
    ## implement function
    features = np.zeros((len(reviews_ints),seq_length),dtype=np.int32)

    for i in range(len(reviews_ints)):

      features[i,-len(reviews_ints[i]):] = reviews_ints[i][:seq_length]

    return features

#function to load model
def loadModel():
    vocab_size = 74073
    output_size = 1
    embedding_dim = 128
    hidden_dim = 256
    n_layers = 2

    net = None
    try:
        net = lstmModel.SentimentLSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
        net.load_state_dict(torch.load('model/LSTM_Model.pt',map_location='cpu'))
    except:
        print('...Error in loading model...')
        net = None

    return net

#code to predict sentiment
def predict(net, test_review, sequence_length=200):
  ''' Prints out whether a give review is predicted to be
  positive or negative in sentiment, using a trained model.

  params:
  net - A trained net
  test_review - a review made of normal text and punctuation
  sequence_length - the padded length of a review
  '''
  try:
      with open('model/vocab2int','rb') as vocab:
          vocab_to_int = pickle.load(vocab)
  except:
      return 'Error does not foud vocab file'

  test_review = ''.join([ch for ch in test_review if ch not in string.punctuation])
  encoded_words = [vocab_to_int[word] for word in test_review.split() if(vocab_to_int.get(word,False)>0)]

  features = pad_features(np.array(encoded_words).reshape(1,-1),sequence_length)
  features = torch.Tensor(features)

  net.to('cpu')

  #hidden = net.init_hidden(1)
  output,h = net(features.long(),None)

  # print custom response based on whether test_review is pos/neg
  return 'Positive Sentence' if output>0.5 else 'Negative Sentence'


#main function
def main():
    global importSuccess
    if(importSuccess):
        net = loadModel()

        if(net is not None):
            print('Enter any text')
            text = input()
            result = predict(net,text)
            print(result)

if __name__ == '__main__':
    main()
