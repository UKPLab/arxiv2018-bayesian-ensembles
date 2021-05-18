# Bayesian Combination

## Contents

 * [Introduction](#introduction)
 * [Publications](#publications)
 * [Project Structure](#project-structure)
 * [Requirements and Installation](#requirements-and-installation)
 * [Running the experiments from EMNLP 2019](#running-the-experiments-from-emnlp-2019)
 * [Running the Experiments from AAAI 2020](#running-the-experiments-from-aaai-2020)
 * [How to Use Bayesian Combination (BC)](#how-to-use-bayesian-combination-bc)

## Introduction

This package implements several key methods for aggregating annotations from a 
multiple annotators, accounting for each individual's reliability.
The annotations may be sequence labels (e.g., text span annotations) or 
classifications of independent data points (e.g., image or document classification).
The annotators may be experts, crowd workers, or automated classifiers in an ensemble.

The methods that we implement are:
   * *BSC (all variants)* -- "A Bayesian Approach for Sequence Tagging with Crowds", Simpson & Gurevych (2019), EMNLP; See below for details on experiments in this paper; 
   see `src/bayesian_combination/bsc.py`.
   * *Variational combined supervision (VCS)* -- "Low Resource Sequence Tagging with Weak Labels", Simpson et al. (2020), AAAI; See below for details on experiments;
   * *Dawid and Skene* -- "Maximum Likelihood Estimation of Observer Error-Rates Using the EM Algorithm" (1979);
   see `src/baselines/dawid_and_skene.py`.
   * *Independent Bayesian Classifier Combination (IBCC-VB)* -- Simpson, et al. "Dynamic Bayesian combination of multiple imperfect classifiers", Simpson et al. (2013);
   this is an approximate Bayesian version of Dawid and Skene;
   see `src/bayesian_combination/ibcc.py`.
   * *MACE* -- "Learning Whom to Trust with MACE", Hovy et al. (2013), NAACL;
   see `src/bayesian_combination/MACE.py`.
   * *Dynamic IBCC (DynIBCC)* -- Simpson, et al. "Dynamic Bayesian combination of multiple imperfect classifiers", Simpson et al. (2013);
   this is a version of IBCC that tracks changing reliability of annotators;
   see `src/bayesian_combination/ibcc.py`.
   * *BCCWords* -- "Language understanding in the wild: Combining crowdsourcing and machine learning.", Simpson et al. (2015), WWW;
   this is IBCC with bag-of-words features (*note this implementation is incomplete at the moment as each data point is limited to one feature!*);
    see `src/bayesian_combination/ibcc.py`.
   * *HeatmapBCC* -- "Bayesian heatmaps: probabilistic classification with multiple unreliable information sources.", Simpson et al. (2017), ECML-PKDD; 
   IBCC combined with a Gaussian process classifier; 
   see `src/bayesian_combination/ibcc.py`.

This implementation uses a common variational Bayes framework to infer gold labels using
 different annotator models and labelling models. 
 It can also integrate sequence taggers or automated classifiers to improve performance, 
 which it directly trains from the crowd's annotations. 
 The crowd of annotators plus automated taggers together form an ensemble.

## Publications

This is the first implementation of the method presented
in 'A Bayesian Approach for Sequence Tagging with Crowds'
and extended in 'Low Resource Sequence Tagging with Weak Labels'.
The intructions for running the experiments for each are outlined below.

For the BSC method, please use the following citation:
```bibtex
@inproceedings{simpson2019bayesian,
    title = "A {B}ayesian Approach for Sequence Tagging with Crowds",
    author = "Simpson, Edwin D.  and
      Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-1101",
    doi = "10.18653/v1/D19-1101",
    pages = "1093--1104",
    abstract = "Current methods for sequence tagging, a core task in NLP, are data hungry, which motivates the use of crowdsourcing as a cheap way to obtain labelled data. However, annotators are often unreliable and current aggregation methods cannot capture common types of span annotation error. To address this, we propose a Bayesian method for aggregating sequence tags that reduces errors by modelling sequential dependencies between the annotations as well as the ground-truth labels. By taking a Bayesian approach, we account for uncertainty in the model due to both annotator errors and the lack of data for modelling annotators who complete few tasks. We evaluate our model on crowdsourced data for named entity recognition, information extraction and argument mining, showing that our sequential model outperforms the previous state of the art, and that Bayesian approaches outperform non-Bayesian alternatives. We also find that our approach can reduce crowdsourcing costs through more effective active learning, as it better captures uncertainty in the sequence labels when there are few annotations.",
}
```

> **Abstract:**
Current methods for sequence tagging, a core task in NLP, are data hungry, which motivates the use of crowdsourcing as a cheap way to obtain labelled data. However, annotators are often unreliable and current aggregation methods cannot capture common types of span annotation errors. To address this, we propose a Bayesian method for aggregating sequence tags that reduces errors by modelling sequential dependencies between the annotations as well  as  the  ground-truth  labels. By  taking a  Bayesian  approach, we account for uncertainty in the model due to both annotator errors and the lack of data for modelling annotators who complete few tasks. We evaluate our model on crowdsourced data for named entity recognition, information extraction and argument mining, showing that our sequential model outperforms the previous state of the art.   We also find that our approach can reduce crowdsourcing costs through more effective active learning, as it better captures uncertainty in the sequence labels when there are few annotations.


This software was extended for the following paper.
When citing the work on combining automated sequence taggers
with human annotators,
please cite the following:
```bibtex
@inproceedings{simpson2020lowresource,
    title = "Low Resource Sequence Tagging with Weak Labels",
    author = "Simpson, Edwin D. and Pfeiffer, Jonas and
      Gurevych, Iryna",
    booktitle = "Proceedings of the Thirty-fourth AAAI Conference on Artificial Intelligence (AAAI)",
    month = feb,
    year = "2020",
    address = "New York, USA",
    publisher = "AAAI",
    url = "https://public.ukp.informatik.tu-darmstadt.de/UKP_Webpage/publications/2020/2020_AAAI_SE_LowResourceSequence.pdf",
    pages = "to appear",
}
```
> **Abstract:**
Current methods for sequence tagging depend on large quantities of domain-specific training data, limiting their use in new, user-defined tasks with few or no annotations. While crowdsourcing can be a cheap source of labels, it often introduces errors that degrade the performance of models trained on such crowdsourced data. Another solution is to use transfer learning to tackle low resource sequence labelling, but current approaches rely heavily on similar high resource datasets in different languages. In this paper, we propose a domain adaptation method using Bayesian sequence combination to exploit pretrained models and unreliable crowdsourced data that does not require high resource data in a different language. Our method boosts performance by learning the relationship between each labeller and the target task and trains a sequence labeller on the target domain with little or no gold-standard data. We apply our approach to labelling diagnostic classes in medical and educational case studies, showing that the model achieves strong performance though zero-shot transfer learning and is more effective than alternative ensemble methods. Using NER and information extraction tasks, we show how our approach can train a model directly from crowdsourced labels, outperforming pipeline approaches that first aggregate the crowdsourced data, then train on the aggregated labels.


For both cases, the contact person is:

Edwin Simpson, edwin.simpson@gmail.com (previously: simpson@ukp.informatik.tu-darmstadt.de).

https://www.ukp.tu-darmstadt.de/

Please send us an e-mail or report an issue, if something is broken or if you have further questions on how to
reproduce the experiments or run the method in a new project.

## Project structure

   * `src` -- contains all the python code
   * `src/baselines` -- implementations of the methods we evaluate BSC against, including the MACE implementation in Java, see NOTICE.txt
   * `src/bayesian_combination` -- the core implementation of the variational Bayes method
   * `src/bayesian_combination/annotator_models` -- various models of annotator reliability
   * `src/bayesian_conbination/emission_models` -- models for generating the features from true label
   * `src/bayesian_combination/label_models` -- models for sequential labels, classification without features, and classification with features
   * `src/bayesian_combination/tagger_wrappers` -- wrapper classes for integrating automated classifiers using variational combined supervision (VCS)
   * `src/data` -- simualted data generation and loaders for our test datasets
   * `src/evaluation` -- experimental code for calling all aggregation methods, running the active learning simulation
   and computing evaluation metrics
   * `src/experiments` -- scripts for running experiments in the papers, see below
   * `src/taggers` -- Implementations of automated taggers for inclusion using VCS, including the BiLSTM-CRF for sequence tagging, see NOTICE.txt
   * `src/test` -- simple unit tests
   * `batch_scripts` -- for running on a cluster with SLURM
   * `config` -- config files for the simulations
   * `data` -- datasets and outputs from error analysis
   * `documents` -- tex files for the EMNLP paper.

### Third-pary code included in this repository

* src/baselines/MACE -- by Dirk Hovy, https://github.com/dirkhovy/MACE
* src/baselines/hmm.py -- by An Tanh Nguyen, https://github.com/thanhan
* src/taggers/lample_lstm_tagger -- by Guillaume Lample, https://github.com/glample/tagger 

## Requirements and Installation

Python 3.5 or higher.

Installation: there are two installation scripts. If you want to run the
experiments from either of the papers, use:
~~~
chmod +x setup_with_expts.sh
./setup_with_expts.sh
~~~

If you don't need to run the experiments and just want to use the methods,
you can use:
~~~
chmod +x setup.sh
./setup.sh
~~~

These scripts use virtualenv to set up a new
virtual environment in the directory `env` and 
install the packages listed in requirements.txt.
The script `setup_with_expts.sh` also adds the paths for experiment scripts and installs the Spacy packages
required for some experiments.

Set data path: in `src/evaluation/experiments.py`, change `data_root_dir` to point
to the location where you would like to store datasets and results.
Create the directories `<data_root_dir>/data`  and
`<data_root_dir>/output`.

### Obtaining Datasets

If you wish to reproduce the experiments in the EMNLP 2019 paper,
or the experiments in the crowdsourcing experiments in AAAI 2020 paper,
follow these steps
to install the datasets. The data for the transfer learning experiments in the
AAAI 2020 paper is already in the correct place.  

1. Set the data_root_dir, data and output paths as described above.

2. Extract the NER crowdsourced data from `data/crf-ma-NER-task1.zip`.
Move the directory to `<data_root_dir>/data`.

3. Extract the NER test data from `data/English NER.zip` and
move it to `<data_root_dir>/data`.

4. For the PICO dataset,
extract the data from `data/bio.zip` and move it to `<data_root_dir>/data`.

5. If you want to get the original data that `bio/` 
folder was derived from,
checkout the git repo https://github.com/yinfeiy/PICO-data/ (you don't need these
to repeat the experiments). You can regenerate the files by deleting the `bio/` 
folder, creating a link to the PICO-data repo, and running the PICO experiments
as instructed below. The link should be:
`ln -s ~/git/PICO-data/ <data_root_dir>/data/bio-PICO`,
where the path `~/git/PICO-data/` should be replaced with location where you 
checked out PICO-data.

6. For the ARG dataset, unzip the data from `data/argmin_LMU.zip` and move it to `<data_root_dir>/data`.

## Running the Experiments from EMNLP 2019

These instructions are for "A Bayesian Approach To Sequence Tagging with Crowds".

 If you just want to use BSC for a new project,
  you don't need to reproduce the experiments from the paper,
  so you can skip this part
  and the next part, and go to
 the section "Using Bayesian Sequence Combination (BSC)".

 For all of these scripts, please run them from the root
 directory of this repository. No command line arguments are needed,
 instead you should comment and uncomment the  code
 to disable or enable the methods you want to test.

### First step

Run the experiments from the root directory of the repo: `cd .`

Set the Python path to the src directory:

`export PYTHONPATH=$PYTHONPATH:"./src"`

### NER experiments

You can run the table 2 experiments using the following:

Run `python src/experiments/EMNLP2019/run_ner_experiments.py`.

Uncomment methods in the code to enable or disable them.
The code will tune the methods on the dev set (or for speed, a subsample of the dev set),
comment this section and uncomment code to repeat selected methods with pre-determined hyperparameters.

Results are saved to `<data_root_dir>\output\ner\result_started_<date>_<time>...csv`.

For the active learning simulation:

Edit `src/experiments/EMNLP2019/run_ner_active_learning.py` to enable or disable the methods you want to test.

Run `python src/experiments/EMNLP2019/run_ner_active_learning.py`.

Run `python src/experiments/EMNLP2019/plot_active_learning.py`
to output plots for active learning (the plots are output to the directory specified in this script).

### PICO experiments

You can get the table 2 results using the following:

Run `python src/experiments/EMNLP2019/run_pico_experiments.py`.

Uncomment methods in the code to enable or disable them.
The code will tune the methods on the dev set (or for speed, a subsample of the dev set),
comment this section and uncomment code to repeat selected methods with pre-determined hyperparameters.

Results are saved to `<data_root_dir>\output\pico\result_started_<date>_<time>...csv`.

### ARG experiments

For the ARG results in table 2, the scripts follow the same pattern as for NER
and PICO. Please run `src/experiments/EMNLP2019/run_arg_experiments.py`.

Results are saved to `<data_root_dir>\output\arg_LMU_<hyperparameters>\result_started_<date>_<time>...csv`.

### Error analysis

To reproduce the results for Table 3.
First you need to run one of the experiment scripts above,
e.g. `python src/run_ner_experiments.py`,
to obtain some results.

Then, modify the error analysis script,
 `src/evaluation/error_analysis.py`,
 to point to the `pred_start_<date>_<time>...csv`
file that was output by your experiment in `<data_root_dir>\output\`.

Now, run `python src/experiments/error_analysis.py`.

### Annotator models

The plots in Figure 1 were made using `src/plot_pico_worker_models.py`.

## Running the Experiments from AAAI 2020

These instructions are for the experiments in
"Low Resource Sequence Tagging with Weak Labels".

### First step

Run the experiments from the root directory of the repo: `cd .`

Set the Python path to the src directory:

`export PYTHONPATH=$PYTHONPATH:"./src"`

### Transfer Learning by Combining Models

These are the experiments with FAMULUS data.

First, download the German Fasttext embeddings:
 `wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.de.300.vec.gz` 
 then unzip to `./data/famulus`:
 
 `mv cc.de.300.vec.gz ./data/famulus; cd ./data/famulus; gunzip cc.de.300.vec.gz; cd ../..`

To run the zero-shot experiments (Table 3):

Run `python src/experiments/AAAI2020/run_famulus_unsupervised.py <"TEd"|"Med">`.

The results will be printed to the console. Change line 49 `basemodels = ` to
modify whether CRF or LSTM base models are used.
Choose either "TEd" or "Med" to run with one of the two
FAMULUS datasets.

To reprint the key results:

Run `python src/experiments/AAAI2020/give_me_the_headlines.py 1 "'bilstm-crf', 'crf'" <"TEd"|"Med">`

To run the semi-supervised learning experiments (Figure 2):

Run `python src/experiments/AAAI2020/run_famulus_semisupervised.py`.

Results are plotted and saved to `results/test3_spanf1_<class ID>.pdf`.


### Enhancing a Crowd with an Automated Sequence Tagger

These are the experiments with NER and PICO crowdsourced data.

For the methods that do not involve an LSTM component, 
the scripts are the same as for EMNLP 2019:
 `python src/experiments/EMNLP2019/run_ner_experiments.py` and
`python src/experiments/EMNLP2019/run_pico_experiments.py`.  

For those with the LSTM, run `python src/experiments/AAAI2020/run_ner_experiments_gpu.py` and
`python src/experiments/AAAI2020/run_pico_experiments_gpu.py`.  

Uncomment methods in the code to enable or disable them.
The commented code will tune the methods on the dev set (or for speed, a subsample of the dev set).
The uncommented code repeats selected methods with previously-tuned hyperparameters.

Results are saved to `<data_root_dir>\output\ner3\result_started_<date>_<time>...csv`
and `<data_root_dir>\pico3\result_started_<date>_<time>...csv`.

## How to Use Bayesian Combination (BC)

### Data Format

The crowdsourced data is an annotation matrix, `annotations`,
 where columns correspond to workers, and rows correspond
to data points. 
For sequence labelling of text, each data point is a token, 
and their corresponding rows should be ordered as in the documents.
the annotations for all documents should be concatenated into a single matrix.
Where a worker has not labeled a data point, 
the entry in the matrix is -1.
The start points of each sequence (e.g., block of text)
are indicated by a vector, `doc_start`, which has '1' to indicate that a token is the first in
a new block of text and '0' for all other tokens.
If there is no sequence to the labels (i.e., for standard classification tasks),
then `doc_start=None`.

The default setup in the example below assumes that the labels in the annotation matrix  
are as follows:
   * 0: Inside
   * 1: Outside
   * 2: Beginning, to be used at the start of every span

If you have multiple annotation types you may have more class labels,
for example, instead of simply I and B, you may have I-type1, B-type1, I-type2, B-type2.
In this case you must pass in a list of values for the inside and beginning tokens to
the constructor.

### Aggregating Crowdsourced Data

The main method implementation is the BC class in src/bayesian_combination/bc.py. 
The bayesian_combination package also contains several subclasses of 
BC that implement BSC, MACE, BCCWords, IBCC-VB (in ibcc module), HeatmapBCC (also in the ibcc module) 
and DynIBCC (in ibcc module). 
Each of these methods is implemented by configuring BC in a particular way.
See the doc strings 
in bayesian_combination.py for more details on configuration.

BC can be run as follows. First,
construct the object for a labelling task. Let's say we wish to use the BSC-seq method.
 We can either use the BSC subclass:
~~~python
num_classes = 5 # number of different tags/labels/states/classes or whatever your application calls them

bc_model = BSC(L=num_classes, K=num_annotators, annotator_model='seq', tagging_scheme='none')
~~~
Or, the same thing can be achieved by instantiating Bayesian Combination directly with 
the correct arguments:
~~~python
bc_model = BC(L=num_classes, K=num_annotators, annotator_model='seq', tagging_scheme='none', 
              true_label_model='HMM', discrete_feature_likelihoods=True)
~~~
These two are equivalent.
In the example above, the annotator model is set to 'seq', which
uses the default sequential annotator model. However, if the results are not looking good
on some datasets, you may also wish to try "annotator_model='CM'" (this is the confusion matrix in the EMNLP 2019 paper),
which will use a simpler annotator model.
Further alternatives are "annotator_model='CV'" (confusion vector),
 "annotator_model='spam'" (as used by MACE),
and "annotator_model='acc' (a single accuracy value for each worker).

For many NLP tasks like named entity recogintion, we need to use a tagging scheme like 
BIO to extract spans of multiple tokens. 
In that case, we can set the tagging scheme differently to set up the correct priors:
~~~python
num_classes = 3 # Beginning, Inside and Outside

bc_model = BSC(L=num_classes, K=num_annotators, annotator_model='seq', tagging_scheme='IOB2')
~~~
Or alternatively:
~~~python
bc_model = BC(L=num_classes, K=num_annotators, annotator_model='seq', tagging_scheme='IOB2', 
              true_label_model='HMM', discrete_feature_likelihoods=True)
~~~

Different models for the methods listed at the top
of this page can also be constructed using the 
classes in bayesian_combination package. Alternatively,
you can create new configurations by instantiating BC directly, 
e.g., if you want to use dynIBCC with BSC,
set annotator_model='dyn' and true_label_model='HMM'.

A single call is used to train the model and produce the aggregated labels:
~~~python
probs, agg = bc_model.fit_predict(annotations, doc_start, features)
~~~
For sequence labelling, doc_start is an optional array of zeros and ones, as above.
The features object is a numpy array that typically contains strings, 
i.e. a column vector of text tokens.
Other features can also be provided as additional columns if using 
the GP label model. 
The shape of features is num_tokens x num_features_per_token.
If we consider only the text tokens as features, num_features_per_token == 1.

The `fit_predict` method outputs 'probs',
 i.e. the class label probabilities for each token, 
'agg', which is the most likely sequence of labels aggregated 
from the crowd's data,
and 'plabels', which is the probability of the most likely labels.

### Label model

The method can also be applied to non-sequential classification tasks by turning off the
sequential label model. This is achieved by passing an optional constructor argument
"true_label_model='noHMM'". Alternatively, if you have continuous feature data,
you can use "true_label_model='GP'" to use a Gaussian process label model, as in the
HeatmapBCC method.

## Competence Scores

A competence score that rates the ability of an annotator can be useful for selecting the best workers for future tasks.
Given a `bsc_model` object, after calling `run()` as described above,
we can compute the annotator accuracy as follows:
~~~python
acc = bc_model.A.annotator_accuracy()
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
~~~python
infor = bc_model.A.informativeness()
print(infor)
~~~
The informativeness is also returned as a vector and can also be used to rank workers (higher is better).
