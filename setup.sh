#!/bin/bash


virtualenv env -p python3
source env/bin/activate
pip install -r requirements.txt
source virtualenvwrapper.sh
add2virtualenv src

