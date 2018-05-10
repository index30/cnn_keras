#!/bin/sh

cd $(dirname `pwd`)
if [ `which virtualenv` ]; then
  echo "Check installed virtualenv"
else
  if [ "$(uname)" == 'Darwin' ]; then
    # your OS is Mac
    sudo pip install virtualenv
  else
    # your OS is Linux
    sudo apt-get install python-pip
    sudo pip install virtualenv
    sudo pip install virtualenvwrapper
  fi
fi

python3 -m virtualenv cnn_keras

#echo $(pwd)
cd "$(pwd)/cnn_keras"
. bin/activate
pip3 install -r requirements.txt
pip3 list

echo "Finish building environment"
