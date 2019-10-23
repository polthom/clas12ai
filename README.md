# clas12ai
AI tools for CLAS12 reconstruction
## How to use
First you need to install the dependencies of the script. If you are using anacoda you can use the conda_environment.yaml to do that automatically. Simply run:
  conda env create -f environment.yaml

Use the ml-cli script to train or test your machine learning problem. Predictions using the C wrappers is not implemented yet.
An example of how to use the script is shown below:

  ./ml-cli.py train --training-dir ../data/6/training/ --testing-dir ../data/6/testing/ --out-model ./et --model-type et

This will train an ExtraTrees model (et) using the svm files located in ../data/6/training/ and evaluate it using the 
testing set in ../data/6/testing/. The generated model will be saved in ./et.
