#!/bin/bash
sudo apt-get update
sudo apt-get install python3-pip
sudo apt install redis-server
sudo pip3 install flask redis rq pandas scipy emcee tqdm corner matplotlib flask_mail boto boto3 aws-ssh
sudo apt-get install python3-venv
sudo apt install python3-flask
sudo apt install python3-dev build-essential libssl-dev libffi-dev python3-setuptools
sudo apt install python3-venv
sudo apt install nginx
export FLASK_APP=run.py
export FLASK_ENV=development
