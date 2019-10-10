# Bayesian Sequence Combination

Aggregate sequential annotations from a crowd of annotators, accounting for their unreliability.
The implementation allows you to test different configurations of annotator models + optional
LSTM sequence tagger. The crowd of annotators plus automated taggers together form an ensemble.

This is the implementation of the method discussed in the following paper -- 
please use the following citation:

@article{DBLP:journals/corr/abs-1811-00780,
  author    = {Edwin Simpson and
               Iryna Gurevych},
  title     = {Bayesian Ensembles of Crowds and Deep Learners for Sequence Tagging},
  journal   = {CoRR},
  volume    = {abs/1811.00780},
  year      = {2018},
  url       = {http://arxiv.org/abs/1811.00780},
  archivePrefix = {arXiv},
  eprint    = {1811.00780},
  timestamp = {Thu, 22 Nov 2018 17:58:30 +0100},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1811-00780},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}

> **Abstract:** 
Current methods for sequence tagging, a core task in NLP, are data hungry, which motivates the use of crowdsourcing as a cheap way to obtain labelled data. However, annotators are often unreliable and current aggregation methods cannot capture common types of span annotation errors. To address this, we propose a Bayesian method for aggregating sequence tags that reduces errors by modelling sequential dependencies between the annotations as well  as  the  ground-truth  labels. By  taking a  Bayesian  approach, we account for uncertainty in the model due to both annotator errors and the lack of data for modelling annotators who complete few tasks. We evaluate our model on crowdsourced data for named entity recognition, information extraction and argument mining, showing that our sequential model outperforms the previous state of the art.   We also find that our approach can reduce crowdsourcing costs through more effective active learning, as it better captures uncertainty in the sequence labels when there are few annotations.

Contact person:

Edwin Simpson, simpson@ukp.informatik.tu-darmstadt.de

https://www.ukp.tu-darmstadt.de/

Please send us an e-mail or report an issue, if something is broken or if you have further questions.

## Project structure

   * `src` -- contains all the python code, with the scripts for running experiments in the paper at the top level
   * `src/baselines` -- implementations of the methods we evaluate BSC against
   * `src/bsc` -- the core implementation of our method, including various annotator models
   * `src/data` -- simualted data generation and loaders for our test datasets
   * `src/evaluation` -- experimental code for calling all aggregation methods, running the active learning simulation 
   and computing evaluation metrics
   * `src/lample_lstm_tagger` -- LSTM-based model for sequence tagging, see NOTICE.txt
   * `src/models` -- temporary storage of LSTM models
   * `src/test` -- simple unit tests
   * `batch_scripts` -- for running on a cluster with SLURM
   * `config` -- config files for the simulations
   * `data` -- outputs from error analysis
   * `documents` -- tex files for the paper. 
   * `MACE` -- MACE implementation in Java, see NOTICE.txt

## Third-pary code included in this repository:

* MACE -- by Dirk Hovy, https://github.com/dirkhovy/MACE 
* src/baselines/hmm.py -- by An Tanh Nguyen, https://github.com/thanhan

## Requirements and installation.

Python 3.5 or higher. 

Use virtualenv to install the packages listed in requirements.txt -- you can use the script `setup.sh` to automatically set up a new 
virtual environment in the directory `env`.
Please also run:
~~~
python -m spacy download en_core_web_sm
~~~

### Obtaining Datasets

If you wish to reproduce the experiments in the paper, follow these steps
to install the datasets.

1. Create the directories `../../data/bayesian_sequence_combination/data`  and 
`../../data/bayesian_sequence_combination/output`. 
The `../../` is relative to the directory containing this repository.
If you want to put the data or results somewhere else, 
you can replace all references to '../../data' in the code with the desired location.

2. Extract the NER crowdsourced data from `data/crf-ma-NER-task1.zip`. 
Move the directory to `../../data/bayesian_sequence_combination/data`.

3. Extract the NER test data from `data/English NER.zip` and 
move it to `../../data/bayesian_sequence_combination/data`.

4. For the PICO dataset, 
checkout the git repo https://github.com/yinfeiy/PICO-data/ . 

5. Create a link to the PICO data:
`ln -s ~/git/PICO-data/ ../../data/bayesian_sequence_combination/data/bio-PICO`, 
where the path `~/git/PICO-data/` should be replaced with location where you checked out the PICO-data.

6. For the ARG dataset, unzip the data from `data/argmin_LMU.zip` and move it to `../../data/bayesian_sequence_combination/data`.
 
## Running the experiments

 If you don't need to reproduce the experiments from the paper, you can skip this part and go to
 the next section.
 
 For all of these scripts, please run them from the root
 directory of this repository. No command line arguments are needed,
 instead you should comment and uncomment the  code
 to disable or enable the methods you want to test.
 
### NER experiments

You can run the table 2 and table 4 experiments using the following:

`src/run_ner_experiments.py` for non-LSTM methods and 
`src/run_ner_experiments_gpu.py` for those that can make use of a GPU.

Uncomment methods in the code to enable or disable them.
To tune the method on the dev set, see the comments in the code
and uncomment the appropriate bit. 

Results are saved to `~\data\bayesian_sequence_combination\output\ner\result_started_<date>_<time>...csv`.

For the active learning simulation:

Run `src/run_ner_active_learning.py` 
and `src/run_ner_active_learning_nogpu.py`.

Now edit `src/evaluation/plot_active_learning.py` to uncomment
the block referring to 'NER_AL'. Run this script to output the plots
to the specified directory.

### PICO experiments

You can get the table 2 results using the following:

`src/run_pico_experiments.py` for non-LSTM methods and 
`src/run_pico_experiments_gpu.py` for those that can make use of a GPU.

Uncomment methods in the code to enable or disable them.
To tune the method on the dev set, see the comments in the code
and uncomment the appropriate bit.

Results are saved to `~\data\bayesian_sequence_combination\output\pico\result_started_<date>_<time>...csv`.

For table 4:

`src/run_pico_experiments_task2.py`

Results are saved to `~\data\bayesian_sequence_combination\output\pico_task2\result_nocrowd_started_<date>_<time>...csv`.

For the active learning simulation:

Run `src/run_pico_active_learning.py` 
and `src/run_pico_active_learning_nogpu.py`.

Now edit `src/evaluation/plot_active_learning.py` to uncomment
the block referring to 'PICO_AL'. Run this script to output the plots
to the specified directory.

### Error analysis

First you need to run one of the experiment scripts above, 
e.g. `src/run_ner_experiments.py`,
to obtain some results.

Then, modify the error analysis script,
 `src/evaluation/error_analysis.py`, 
 to point to the `pred_start_<date>_<time>...csv`
file output by your experiment in `~\data\bayesian_sequence_combination\output\`.

Now, run `src/evaluation/error_analysis.py`.

### Annotator models

The plots in Figure 1 were made using `src/plot_pico_worker_models.py`.

## Running Bayesian Sequence Combination

### Data Format

The crowdsourced data is an annotation matrix where columns correspond to workers, and rows correspond
to tokens. Where a worker has not labeled a token, the entry in the matrix is -1. 
Rows containing the annotations for all documents/sentences (i.e. the blocks of text
that the annotators labeled) should be concatenated. The start points of each block of text
is indicated by a vector, doc_start, which has '1' to indicate that a token is the first in
a new block of text and '0' for all other tokens.

The default setup in the example below assumes that the labels in the annotation matrix  
are as follows:
   * 0: Inside
   * 1: Outside
   * 2: Beginning, to be used at the start of every span

If you have multiple annotation types you may have more class labels,
for example, instead of simply I and B, you may have I-type1, B-type1, I-type2, B-type2.
In this case you must pass pass in a list of values for the inside and beginning tokens to 
the constructor.

### Aggregating Crowdsourced Data

The main method implementation is in src/algorithm/bsc.py. It can be run as follows. First,
construct the object for an IOB tagging problem:
~~~
num_classes = 3 # B, I and O

bsc_model = bsc.BSC(L=num_classes, K=num_annotators, worker_model='seq')

bsc_model.verbose = True # to get extra debugging info
~~~
The constructor takes an argument, 'worker_model', which determines the type of 
annotator model. In the example above it is set to "worker_model='seq'", which
uses the default sequential annotator model. However, if the results are not looking good
on some datasets, you may also wish to try "worker_model='ibcc''" (this is the confusion matrix or 'cm' model in the paper), 
which will use a simpler annotator model. 
Further alternatives are "worker_model='vec'" (called 'cv' or confusion vector in the paper),
 "worker_model='mace'" (called 'spam' in the paper),
and "worker_model='acc' (a single accuracy value for each worker).

A single call is used to train the model and produce the aggregated labels:
~~~
probs, agg = bsc_model.run(annotations, doc_start, features)
~~~
The features object is a numpy array that typically contains strings, i.e. a column vector of text tokens.
Other features can also be provided as additional columns. The shape of features is num_tokens x num_features_per_token.
If we consider only the text tokens as features, num_features_per_token == 1.

The method outputs both 'probs', i.e. the class label probabilities for each token, and 
'agg', which is the most likely sequence of labels aggregated from the crowdsourced data. 

### Sequential label model

The method can also be applied to non-sequential classification tasks by turning off the 
sequential label model. This is achieved by passing an optional constructor argument
"transition_model='noHMM'". 
 
## Competence Scores

A competence score that rates the ability of an annotator can be useful for selecting the best workers for future tasks.
Given a `bsc_model` object, after calling `run()` as described above, 
we can compute the annotator accuracy as follows:
~~~
acc = bsc_model.annotator_accuracy()
print(acc)
~~~
This returns a vector of accuracies for the annotators. 
The accuracy has a couple of downsides as a metric for worker competence:
(1) if the classes are not balanced (e.g. lots of 'O' tokens), the accuracy can be very
high if the worker selects the same label all the time;
(2) annotators can still be informative, even if they are inaccurate, e.g. if 
a worker consistently confuses one class for another.
A better metric is based on the expected information gain, i.e. the reduction in uncertainty
about a true label given the label from a particular worker. We call this 'informativeness':
~~~
infor = bsc_model.informativeness()
print(infor)
~~~
The informativeness is also returned as a vector and can also be used to rank workers (higher is better).

