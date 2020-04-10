#!/bin/bash


virtualenv env -p python3
source env/bin/activate
pip install -r requirements.txt
source virtualenvwrapper.sh
add2virtualenv src
add2virtualenv src/experiments/EMNLP2019
add2virtualenv src/experiments/AAAI2020
python -m spacy download en_core_web_sm
python -m spacy download en
