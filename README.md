## Requirements

Tested with Python 3. 

Use virtualenv to install the packages listed in requirements.txt.

## Running Bayesian Sequence Combination

The main method implementation is in src/algorithm/bsc.py. It can be run as follows. First,
construct the object for a BIO tagging problem:
~~~
num_classes = 3 # B, I and O

bac_model = bsc.BAC(L=num_classes, K=num_annotators, worker_model='seq',
            data_model=['IF'])

bac_model.verbose = True # to get extra debugging info
~~~

In the next call, you can fit the model to the data and predict the aggregate labels.

### Data Format

The crowdsourced data is an annotation matrix where columns correspond to workers, and rows correspond
to tokens. Where a worker has not labeled a token, the entry in the matrix is -1. 
Rows containing the annotations for all documents/sentences (i.e. the blocks of text
that the annotators labeled) should be concatenated. The start points of each block of text
is indicated by a vector, doc_start, which has '1' to indicate that a token is the first in
a new block of text and '0' for all other tokens.

The default setup in the example above assumes that the labels in the annotation matrix  
are as follows:
   * 0: Inside
   * 1: Outside
   * 2: Beginning, to be used at the start of every span

If you have multiple annotation types you may have more class labels,
for example, instead of simply I and B, you may have I-type1, B-type1, I-type2, B-type2.
In this case you must pass pass in a list of values for the inside and beginning tokens to 
the constructor.

### Training and Aggregation

A single call is used to train the model and produce the aggregated labels:
~~~
probs, agg = bac_model.run(annotations, doc_start, features)
~~~
The features object is a numpy array that typically contains strings, i.e. a column vector of text tokens.
Other features can also be provided as additional columns. The shape of features is num_tokens x num_features_per_token.
If we consider only the text tokens as features, num_features_per_token == 1.

The method outputs both 'probs', i.e. the class label probabilities for each token, and 
'agg', which is the most likely sequence of labels aggregated from the crowdsourced data. 

### Annotator models

The constructor takes an argument, 'worker_model', which determines the type of 
annotator model. In the example above it is set to "worker_model='seq'", which
uses the default sequential annotator model. However, if the results are not looking good
on some datasets, you may also wish to try "worker_model='ibcc''", which will use
a simpler annotator model. A further simplification to the model is "worker_model='vec'".

### Sequential label model

The method can also be applied to non-sequential classification tasks by turning off the 
sequential label model. This is achieved by passing an optional constructor argument
"transition_model='noHMM'". 

## Third-pary code included in this repository:

* MACE
* src/baselines/hmm.py
 
## Competence Scores

Most of the annotator models used here to not learn a single competence score
for each worker. However, a competence score or similar can be useful for ranking
workers to decide who to exclude from a task.
We can compute a competence measure as follows:

~~~
model = bsc.BAC(L=num_classes, K=num_annotators, worker_model='seq',
            data_model=['IF'])

probs, agg = model.run(annotations, doc_start, features)

EPi = model.A._calc_EPi(self.alpha) # gives us a 2D or 3D matrix

if EPi.ndim == 1:
    competence = EPi[1, :]
elif EPi.ndim == 2:
    competence = np.mean(EPi[range(EPi.shape[0]), range(EPi.shape[0]), :], axis=0)
else:
    competence = np.mean(np.mean(EPi[range(EPi.shape[0]), range(EPi.shape[0]), :, :], axis=0), axis=0)
    
print(competence)
~~~

In this case, the competence is the probability of the annotator giving the 
correct answer, averaged over all the different cases considered by the model.
