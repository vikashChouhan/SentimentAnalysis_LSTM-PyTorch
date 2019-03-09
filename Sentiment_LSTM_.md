
# Sentiment Analysis with an RNN

In this notebook, we'll implement a recurrent neural network that performs sentiment analysis.
>Using an RNN rather than a strictly feedforward network is more accurate since we can include information about the *sequence* of words.

Here we'll use a dataset of movie reviews, accompanied by sentiment labels: positive or negative.

<img src="assets/reviews_ex.png" width=40%>

### Network Architecture

The architecture for this network is shown below.

<img src="assets/network_diagram.png" width=40%>

>**First, we'll pass in words to an embedding layer.** We need an embedding layer because we have tens of thousands of words, so we'll need a more efficient representation for our input data than one-hot encoded vectors. We can actually train an embedding with the Skip-gram Word2Vec model and use those embeddings as input, here. However, it's good enough to just have an embedding layer and let the network learn a different embedding table on its own. *In this case, the embedding layer is for dimensionality reduction, rather than for learning semantic representations.*

>**After input words are passed to an embedding layer, the new embeddings will be passed to LSTM cells.** The LSTM cells will add *recurrent* connections to the network and give us the ability to include information about the *sequence* of words in the movie review data.

>**Finally, the LSTM outputs will go to a sigmoid output layer.** We're using a sigmoid function because positive and negative = 1 and 0, respectively, and a sigmoid will output predicted, sentiment values between 0-1.

We don't care about the sigmoid outputs except for the **very last one**; we can ignore the rest. We'll calculate the loss by comparing the output at the last time step and the training label (pos or neg).

---
### Load in and visualize the data


```python
import numpy as np

# read data from text files
with open('data/reviews.txt', 'r') as f:
    reviews = f.read()
with open('data/labels.txt', 'r') as f:
    labels = f.read()
```


```python
print(reviews[:2000])
print()
print(labels[:20])
```

    bromwell high is a cartoon comedy . it ran at the same time as some other programs about school life  such as  teachers  . my   years in the teaching profession lead me to believe that bromwell high  s satire is much closer to reality than is  teachers  . the scramble to survive financially  the insightful students who can see right through their pathetic teachers  pomp  the pettiness of the whole situation  all remind me of the schools i knew and their students . when i saw the episode in which a student repeatedly tried to burn down the school  i immediately recalled . . . . . . . . . at . . . . . . . . . . high . a classic line inspector i  m here to sack one of your teachers . student welcome to bromwell high . i expect that many adults of my age think that bromwell high is far fetched . what a pity that it isn  t   
    story of a man who has unnatural feelings for a pig . starts out with a opening scene that is a terrific example of absurd comedy . a formal orchestra audience is turned into an insane  violent mob by the crazy chantings of it  s singers . unfortunately it stays absurd the whole time with no general narrative eventually making it just too off putting . even those from the era should be turned off . the cryptic dialogue would make shakespeare seem easy to a third grader . on a technical level it  s better than you might think with some good cinematography by future great vilmos zsigmond . future stars sally kirkland and frederic forrest can be seen briefly .  
    homelessness  or houselessness as george carlin stated  has been an issue for years but never a plan to help those on the street that were once considered human who did everything from going to school  work  or vote for the matter . most people think of the homeless as just a lost cause while worrying about things such as racism  the war on iraq  pressuring kids to succeed  technology  the elections  inflation  or worrying if they  ll be next to end up on the streets .  br    br   but what if y

    positive
    negative
    po


## Data pre-processing

The first step when building a neural network model is getting our data into the proper form to feed into the network. Since we're using embedding layers, we'll need to encode each word with an integer. We'll also want to clean it up a bit.

We can see an example of the reviews data above. Here are the processing steps, we'll want to take:
>* We'll want to get rid of periods and extraneous punctuation.
* Also, one might notice that the reviews are delimited with newline characters `\n`. To deal with those, I'm going to split the text into each review using `\n` as the delimiter.
* Then I can combined all the reviews back together into one big string.

First, let's remove all punctuation. Then get all the text without the newlines and split it into individual words.


```python
from string import punctuation

print(punctuation)

# get rid of punctuation
reviews = reviews.lower() # lowercase, standardize
all_text = ''.join([c for c in reviews if c not in punctuation])
```

    !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~



```python
# split by new lines and spaces
reviews_split = all_text.split('\n')
all_text = ' '.join(reviews_split)

# create a list of words
words = all_text.split()

```


```python
words[:30]
```




    ['bromwell',
     'high',
     'is',
     'a',
     'cartoon',
     'comedy',
     'it',
     'ran',
     'at',
     'the',
     'same',
     'time',
     'as',
     'some',
     'other',
     'programs',
     'about',
     'school',
     'life',
     'such',
     'as',
     'teachers',
     'my',
     'years',
     'in',
     'the',
     'teaching',
     'profession',
     'lead',
     'me']



### Encoding the words

The embedding lookup requires that we pass in integers to our network. The easiest way to do this is to create dictionaries that map the words in the vocabulary to integers. Then we can convert each of our reviews into integers so they can be passed into the network.

>  Now we're going to encode the words with integers. Build a dictionary that maps words to integers. Later we're going to pad our input vectors with zeros, so make sure the integers **start at 1, not 0**.
> Also, convert the reviews to integers and store the reviews in a new list called `reviews_ints`.


```python

from collections import Counter

## Build a dictionary that maps words to integers
count = Counter(words)
vocab = sorted(count,key=count.get,reverse=True)
vocab_to_int = {word:i for i,word in enumerate(vocab,1)}

## use the dict to tokenize each review in reviews_split
## store the tokenized reviews in reviews_ints
reviews_ints = [[vocab_to_int[word] for word in review.split()] for review in reviews_split]


```

**Testing our code**

As a text that we've implemented the dictionary correctly, print out the number of unique words in our vocabulary and the contents of the first, tokenized review.


```python
# stats about vocabulary
print('Unique words: ', len((vocab_to_int)))  # should ~ 74000+
print()

# print tokens in first review
print('Tokenized review: \n', reviews_ints[0][:10])
```

    Unique words:  74072

    Tokenized review:
     [21025, 308, 6, 3, 1050, 207, 8, 2138, 32, 1]


### Encoding the labels

Our labels are "positive" or "negative". To use these labels in our network, we need to convert them to 0 and 1.

> We are going  to convert labels from `positive` and `negative` to 1 and 0, respectively, and place those in a new list, `encoded_labels`.


```python
# 1=positive, 0=negative label conversion
encoded_labels = [1  if(label=='positive') else 0 for label in labels.split('\n')]
```
### Removing Outliers

As an additional pre-processing step, we want to make sure that our reviews are in good shape for standard processing. That is, our network will expect a standard input text size, and so, we'll want to shape our reviews into a specific length. We'll approach this task in two main steps:

1. Getting rid of extremely long or short reviews; the outliers
2. Padding/truncating the remaining data so that we have reviews of the same length.

<img src="assets/outliers_padding_ex.png" width=40%>

Before we pad our review text, we should check for reviews of extremely short or long lengths; outliers that may mess with our training.


```python
# outlier review stats
review_lens = Counter([len(x) for x in reviews_ints])
print("Zero-length reviews: {}".format(review_lens[0]))
print("Maximum review length: {}".format(max(review_lens)))
```

    Zero-length reviews: 1
    Maximum review length: 2514


Okay, a couple issues here. We seem to have one review with zero length. And, the maximum review length is way too many steps for our RNN. We'll have to remove any super short reviews and truncate super long reviews. This removes outliers and should allow our model to train more efficiently.

>  First, we'll remove *any* reviews with zero length from the `reviews_ints` list and their corresponding label in `encoded_labels`.


```python
print('Number of reviews before removing outliers: ', len(reviews_ints))

## remove any reviews/labels with zero length from the reviews_ints list.
review_nonzero_idx = [i  for i,review in enumerate(reviews_ints) if(len(review)!=0)]

reviews_ints = [reviews_ints[i] for i in review_nonzero_idx]
encoded_labels = [encoded_labels[i] for i in review_nonzero_idx]

print('Number of reviews after removing outliers: ', len(reviews_ints))
```

    Number of reviews before removing outliers:  25000
    Number of reviews after removing outliers:  25000


---
## Padding sequences

To deal with both short and very long reviews, we'll pad or truncate all our reviews to a specific length. For reviews shorter than some `seq_length`, we'll pad with 0s. For reviews longer than `seq_length`, we can truncate them to the first `seq_length` words. A good `seq_length`, in this case, is 200.

> We'll  define a function that returns an array `features` that contains the padded data, of a standard size, that we'll pass to the network.
* The data should come from `review_ints`, since we want to feed integers to the network.
* Each row should be `seq_length` elements long.
* For reviews shorter than `seq_length` words, **left pad** with 0s. That is, if the review is `['best', 'movie', 'ever']`, `[117, 18, 128]` as integers, the row will look like `[0, 0, 0, ..., 0, 117, 18, 128]`.
* For reviews longer than `seq_length`, use only the first `seq_length` words as the feature vector.

As a small example, if the `seq_length=10` and an input review is:
```
[117, 18, 128]
```
The resultant, padded sequence should be:

```
[0, 0, 0, 0, 0, 0, 0, 117, 18, 128]
```

**Our final `features` array should be a 2D array, with as many rows as there are reviews, and as many columns as the specified `seq_length`.**

This isn't trivial and there are a bunch of ways to do this. But, if we're going to be building your own deep learning networks, we're going to have to get used to preparing our data.


```python
def pad_features(reviews_ints, seq_length):
    ''' Return features of review_ints, where each review is padded with 0's
        or truncated to the input seq_length.
    '''
    ## implement function
    features = np.zeros((len(reviews_ints),seq_length),dtype=np.int32)

    for i in range(len(reviews_ints)):

      features[i,-len(reviews_ints[i]):] = reviews_ints[i][:seq_length]

    return features
```


```python
# Test your implementation!

seq_length = 200

features = pad_features(reviews_ints, seq_length=seq_length)

## test statements - do not change - ##
assert len(features)==len(reviews_ints), "Your features should have as many rows as reviews."
assert len(features[0])==seq_length, "Each feature row should contain seq_length values."

# print first 10 values of the first 30 batches
print(features[:30,:10])
```

    [[    0     0     0     0     0     0     0     0     0     0]
     [    0     0     0     0     0     0     0     0     0     0]
     [22382    42 46418    15   706 17139  3389    47    77    35]
     [ 4505   505    15     3  3342   162  8312  1652     6  4819]
     [    0     0     0     0     0     0     0     0     0     0]
     [    0     0     0     0     0     0     0     0     0     0]
     [    0     0     0     0     0     0     0     0     0     0]
     [    0     0     0     0     0     0     0     0     0     0]
     [    0     0     0     0     0     0     0     0     0     0]
     [   54    10    14   116    60   798   552    71   364     5]
     [    0     0     0     0     0     0     0     0     0     0]
     [    0     0     0     0     0     0     0     0     0     0]
     [    0     0     0     0     0     0     0     0     0     0]
     [    1   330   578    34     3   162   748  2731     9   325]
     [    9    11 10171  5305  1946   689   444    22   280   673]
     [    0     0     0     0     0     0     0     0     0     0]
     [    1   307 10399  2069  1565  6202  6528  3288 17946 10628]
     [    0     0     0     0     0     0     0     0     0     0]
     [   21   122  2069  1565   515  8181    88     6  1325  1182]
     [    1    20     6    76    40     6    58    81    95     5]
     [   54    10    84   329 26230 46427    63    10    14   614]
     [   11    20     6    30  1436 32317  3769   690 15100     6]
     [    0     0     0     0     0     0     0     0     0     0]
     [    0     0     0     0     0     0     0     0     0     0]
     [   40    26   109 17952  1422     9     1   327     4   125]
     [    0     0     0     0     0     0     0     0     0     0]
     [   10   499     1   307 10399    55    74     8    13    30]
     [    0     0     0     0     0     0     0     0     0     0]
     [    0     0     0     0     0     0     0     0     0     0]
     [    0     0     0     0     0     0     0     0     0     0]]


## Training, Validation, Test

With our data in nice shape, we'll split it into training, validation, and test sets.

> We'll create the training, validation, and test sets.
* We'll need to create sets for the features and the labels, `train_x` and `train_y`, for example.
* Define a split fraction, `split_frac` as the fraction of data to **keep** in the training set. Usually this is set to 0.8 or 0.9.
* Whatever data is left will be split in half to create the validation and *testing* data.


```python
split_frac = 0.8

## split data into training, validation, and test data (features and labels, x and y)
encoded_labels = np.array(encoded_labels)


length = len(features)
split = int(split_frac*length)

train_x,valid_x = features[:split],features[split:]
train_y,valid_y = encoded_labels[:split],encoded_labels[split:]

test_x,test_y = valid_x[len(valid_x)//2:],valid_y[len(valid_y)//2:]
valid_x,valid_y = valid_x[:len(valid_x)//2],valid_y[:len(valid_y)//2]
## print out the shapes of your resultant feature data
print('Train set : ',train_x.shape,train_y.shape)
print('Validation set : ',valid_x.shape,valid_y.shape)
print('Test set : ',test_x.shape,test_y.shape)

```

    Train set :  (20000, 200) (20000,)
    Validation set :  (2500, 200) (2500,)
    Test set :  (2500, 200) (2500,)


**Checking our work**

With train, validation, and test fractions equal to 0.8, 0.1, 0.1, respectively, the final, feature data shapes should look like:
```
                    Feature Shapes:
Train set: 		 (20000, 200)
Validation set: 	(2500, 200)
Test set: 		  (2500, 200)
```

---
## DataLoaders and Batching

After creating training, test, and validation data, we can create DataLoaders for this data by following two steps:
1. We'll create a known format for accessing our data, using [TensorDataset](https://pytorch.org/docs/stable/data.html#) which takes in an input set of data and a target set of data with the same first dimension, and creates a dataset.
2. Create DataLoaders and batch our training, validation, and test Tensor datasets.

```
train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
train_loader = DataLoader(train_data, batch_size=batch_size)
```

This is an alternative to creating a generator function for batching our data into full batches.


```python
import torch
from torch.utils.data import TensorDataset, DataLoader

# create Tensor datasets
train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
valid_data = TensorDataset(torch.from_numpy(valid_x), torch.from_numpy(valid_y))
test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

# dataloaders
batch_size = 50

# make sure to SHUFFLE your data
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)
```


```python
# obtain one batch of training data
dataiter = iter(train_loader)
sample_x, sample_y = dataiter.next()

print('Sample input size: ', sample_x.size()) # batch_size, seq_length
print('Sample input: \n', sample_x)
print()
print('Sample label size: ', sample_y.size()) # batch_size
print('Sample label: \n', sample_y)
```

    Sample input size:  torch.Size([50, 200])
    Sample input:
     tensor([[   0,    0,    0,  ...,  416,   37,   50],
            [   0,    0,    0,  ...,    3, 2176,  841],
            [   0,    0,    0,  ...,    4, 6130,  511],
            ...,
            [  30,    4,  147,  ...,  137,   70, 3701],
            [  11,    6,  690,  ..., 1499,    5, 7590],
            [   0,    0,    0,  ...,    2,   85, 1964]], dtype=torch.int32)

    Sample label size:  torch.Size([50])
    Sample label:
     tensor([1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1,
            0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
            0, 0])


---
# Sentiment Network with PyTorch

Below is where we'll define the network.

<img src="assets/network_diagram.png" width=40%>

The layers are as follows:
1. An [embedding layer](https://pytorch.org/docs/stable/nn.html#embedding) that converts our word tokens (integers) into embeddings of a specific size.
2. An [LSTM layer](https://pytorch.org/docs/stable/nn.html#lstm) defined by a hidden_state size and number of layers
3. A fully-connected output layer that maps the LSTM layer outputs to a desired output_size
4. A sigmoid activation layer which turns all outputs into a value 0-1; return **only the last sigmoid output** as the output of this network.

### The Embedding Layer

We need to add an [embedding layer](https://pytorch.org/docs/stable/nn.html#embedding) because there are 74000+ words in our vocabulary. It is massively inefficient to one-hot encode that many classes. So, instead of one-hot encoding, we can have an embedding layer and use that layer as a lookup table. We could train an embedding layer using Word2Vec, then load it here. But, it's fine to just make a new layer, using it for only dimensionality reduction, and let the network learn the weights.


### The LSTM Layer(s)

We'll create an [LSTM](https://pytorch.org/docs/stable/nn.html#lstm) to use in our recurrent network, which takes in an input_size, a hidden_dim, a number of layers, a dropout probability (for dropout between multiple layers), and a batch_first parameter.

Most of the time, our network will have better performance with more layers; between 2-3. Adding more layers allows the network to learn really complex relationships.

> We'll omplete the `__init__`, `forward`, and `init_hidden` functions for the SentimentRNN model class.

Note: `init_hidden` should initialize the hidden and cell state of an lstm layer to all zeros, and move those state to GPU, if available.


```python
# First checking if GPU is available
train_on_gpu=torch.cuda.is_available()

if(train_on_gpu):
    print('Training on GPU.')
else:
    print('No GPU available, training on CPU.')
```

    Training on GPU.



```python
import torch.nn as nn

class SentimentRNN(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.7):
        """
        Initialize the model by setting up the layers.
        """
        super(SentimentRNN, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # define all layers
        self.embed = nn.Embedding(vocab_size,embedding_dim)
        self.lstm = nn.LSTM(embedding_dim,hidden_dim,n_layers,dropout=drop_prob,batch_first=True)
        self.fc = nn.Linear(hidden_dim,output_size)
        self.sigmoid = nn.Sigmoid()
        self.drp = nn.Dropout(p=0.7)

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size=x.shape[0]

        x = self.embed(x)

        x,hidden = self.lstm(x,hidden)

        x = x.reshape(-1,self.hidden_dim)

        x = self.drp(x)

        x = self.fc(x)

        sig_out = self.sigmoid(x)

        # return last sigmoid output and hidden state
        sig_out = sig_out.reshape(batch_size,-1)
        sig_out = sig_out[:,-1]

        return sig_out, hidden


    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())


        return hidden

```

## Instantiate the network

Here, we'll instantiate the network. First up, defining the hyperparameters.

* `vocab_size`: Size of our vocabulary or the range of values for our input, word tokens.
* `output_size`: Size of our desired output; the number of class scores we want to output (pos/neg).
* `embedding_dim`: Number of columns in the embedding lookup table; size of our embeddings.
* `hidden_dim`: Number of units in the hidden layers of our LSTM cells. Usually larger is better performance wise. Common values are 128, 256, 512, etc.
* `n_layers`: Number of LSTM layers in the network. Typically between 1-3

> We'll  define the model  hyperparameters.



```python
# Instantiate the model w/ hyperparams
vocab_size = len(vocab)+1
output_size = 1
embedding_dim = 128
hidden_dim = 256
n_layers = 2

net = None
net = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)

print(net)
```

    SentimentRNN(
      (embed): Embedding(74073, 128)
      (lstm): LSTM(128, 256, num_layers=2, batch_first=True, dropout=0.7)
      (fc): Linear(in_features=256, out_features=1, bias=True)
      (sigmoid): Sigmoid()
      (drp): Dropout(p=0.7)
    )


---
## Training

Below is the typical training code.

>We'll also be using a new kind of cross entropy loss, which is designed to work with a single Sigmoid output. [BCELoss](https://pytorch.org/docs/stable/nn.html#bceloss), or **Binary Cross Entropy Loss**, applies cross entropy loss to a single value between 0 and 1.

We also have some data and training hyparameters:

* `lr`: Learning rate for our optimizer.
* `epochs`: Number of times to iterate through the training dataset.
* `clip`: The maximum gradient value to clip at (to prevent exploding gradients).


```python
# loss and optimization functions
lr=0.001

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

```


```python
# training params
validLoss,trainLoss = [],[]
epochs = 10 # 3-4 is approx where I noticed the validation loss stop decreasing

counter = 0
print_every = 100
clip=5 # gradient clipping
minValidLoss = np.inf #use for saving model whenever valid loss becomes less than min valid loss

# move model to GPU, if available
device = 'cuda' if(torch.cuda.is_available()) else 'cpu'
net.to(device)
print("Running on",device)

net.train()
# train for some number of epochs
for e in range(epochs):
    # initialize hidden state
    h = net.init_hidden(batch_size)

    # batch loop
    print()
    for batch,(inputs, labels) in enumerate(train_loader,1):
        print(f'\rBatch : {batch}/{len(train_loader)}',end='')
        counter += 1

        if(train_on_gpu):
            inputs, labels = inputs.cuda().long(), labels.cuda().long()

        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])

        # zero accumulated gradients
        net.zero_grad()

        # get the output from the model
        output, h = net(inputs, h)

        # calculate the loss and perform backprop
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()

        # loss stats
    else:
        # Get validation loss
        print()
        val_h = net.init_hidden(batch_size)
        val_losses = []
        net.eval()
        for inputs, labels in valid_loader:

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            val_h = tuple([each.data for each in val_h])

            if(train_on_gpu):
                inputs, labels = inputs.cuda().long(), labels.cuda().long()

            output, val_h = net(inputs, val_h)
            val_loss = criterion(output.squeeze(), labels.float())

            val_losses.append(val_loss.item())

        net.train()
        print("Epoch: {}/{}...".format(e+1, epochs),
              "Step: {}...".format(counter),
              "Loss: {:.6f}...".format(loss.item()),
              "Val Loss: {:.6f}".format(np.mean(val_losses)))

        if(np.mean(val_losses)<minValidLoss):
          print('\t\t...Saving Model...')
          torch.save(net.state_dict(),'LSTM_Model.pt')
          minValidLoss = np.mean(val_losses)

        trainLoss.append(loss.item())
        validLoss.append(np.mean(val_losses))

```

    Running on cuda

    Batch : 400/400
    Epoch: 1/10... Step: 400... Loss: 0.694966... Val Loss: 0.689454
    		...Saving Model...

    Batch : 400/400
    Epoch: 2/10... Step: 800... Loss: 0.546926... Val Loss: 0.595986
    		...Saving Model...

    Batch : 400/400
    Epoch: 3/10... Step: 1200... Loss: 0.333596... Val Loss: 0.551114
    		...Saving Model...

    Batch : 400/400
    Epoch: 4/10... Step: 1600... Loss: 0.482259... Val Loss: 0.461121
    		...Saving Model...

    Batch : 400/400
    Epoch: 5/10... Step: 2000... Loss: 0.325998... Val Loss: 0.431142
    		...Saving Model...

    Batch : 400/400
    Epoch: 6/10... Step: 2400... Loss: 0.334639... Val Loss: 0.476345

    Batch : 400/400
    Epoch: 7/10... Step: 2800... Loss: 0.250438... Val Loss: 0.535614

    Batch : 400/400
    Epoch: 8/10... Step: 3200... Loss: 0.142953... Val Loss: 0.590295

    Batch : 400/400
    Epoch: 9/10... Step: 3600... Loss: 0.074845... Val Loss: 0.579512

    Batch : 400/400
    Epoch: 10/10... Step: 4000... Loss: 0.187632... Val Loss: 0.798425



---
## Testing

There are a few ways to test your network.

* **Test data performance:** First, we'll see how our trained model performs on all of our defined test_data, above. We'll calculate the average loss and accuracy over the test data.

* **Inference on user-generated data:** Second, we'll see if we can input just one example review at a time (without a label), and see what the trained model predicts. Looking at new, user input data like this, and predicting an output label, is called **inference**.


```python
#Loading saved model
net.load_state_dict(torch.load('LSTM_Model.pt'))

if(train_on_gpu):
  net.to('cuda')
# Get test data loss and accuracy

test_losses = [] # track loss
num_correct = 0

# init hidden state
h = net.init_hidden(batch_size)

net.eval()
# iterate over test data
for inputs, labels in test_loader:

    # Creating new variables for the hidden state, otherwise
    # we'd backprop through the entire training history
    h = tuple([each.data for each in h])

    if(train_on_gpu):
        inputs, labels = inputs.cuda().long(), labels.cuda().long()

    # get predicted outputs
    output, h = net(inputs, h)

    # calculate loss
    test_loss = criterion(output.squeeze(), labels.float())
    test_losses.append(test_loss.item())

    # convert output probabilities to predicted class (0 or 1)
    pred = torch.round(output.squeeze())  # rounds to the nearest integer

    # compare predictions to true label
    correct_tensor = pred.eq(labels.float().view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    num_correct += np.sum(correct)


# -- stats! -- ##
# avg test loss
print("Test loss: {:.3f}".format(np.mean(test_losses)))

# accuracy over all test data
test_acc = num_correct/len(test_loader.dataset)
print("Test accuracy: {:.3f}".format(test_acc))
```

    Test loss: 0.431
    Test accuracy: 0.805


### Inference on a test review



>We'll write a `predict` function that takes in a trained net, a plain text_review, and a sequence length, and prints out a custom statement for a positive or negative review!
* we can use any functions that we've already defined or define any helper functions you want to complete `predict`, but it should just take in a trained net, a text review, and a sequence length.



```python
# negative test review
test_review_neg = 'The worst movie I have seen; acting was terrible and I want my money back. This movie had bad acting and the dialogue was slow.'

```


```python
import string
def predict(net, test_review, sequence_length=200):
  ''' Prints out whether a give review is predicted to be
  positive or negative in sentiment, using a trained model.

  params:
  net - A trained net
  test_review - a review made of normal text and punctuation
  sequence_length - the padded length of a review
  '''
  test_review = ''.join([ch for ch in test_review if ch not in string.punctuation])
  encoded_words = [vocab_to_int[word] for word in test_review.split() if(vocab_to_int.get(word,False)>0)]

  features = pad_features(np.array(encoded_words).reshape(1,-1),sequence_length)
  features = torch.Tensor(features)

  net.to('cpu')

  hidden = net.init_hidden(1)
  output,h = net(features.long(),(hidden[0].cpu(),hidden[1].cpu()))

  # print custom response based on whether test_review is pos/neg
  print('Positive' if output>0.5 else 'Negative')


predict(net,test_review_neg)
```

    Negative



```python
# positive test review
test_review_pos = 'This movie had the best acting and the dialogue was so good. I loved it.'

```


```python
# call function
# try negative and positive reviews!
seq_length=200
predict(net, test_review_pos, seq_length)
```

    Positive
