#/bin/bash

if ! which >/dev/null 2>&1 python2.7;
then
    echo "Python 2.7 is required but not found"
    exit 1
fi

pip install virtualenv

# Create and activate virtual environment
echo "Creating virtual environment"
virtualenv -p $(which python2.7) venv2
source ./venv2/bin/activate

# Install python dependencies
pip install -r requirements2.txt
deactivate

virtualenv -p $(which python3) venv3
source ./venv3/bin/activate
pip3 install -r requirements3.txt
deactivate
