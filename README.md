# NIH Seizure Prediction
Code and documentation for my solution (51th place) for the Kaggle Melbourne University AES/MathWorks/NIH Seizure Prediction challenge : https://www.kaggle.com/solomonk

*A 2016 Kaggle competition.*

https://www.kaggle.com/c/melbourne-university-seizure-prediction

**Authors**:
* [Shlomo Kashani](http://www.deep-ml.com)

**Contents** :

- [Overview of the my solution](#overview-of-the-my-solution)
- [Models](#models)
- [Cross-validation](#cv-method)
- [Reproduce the solution](#reproduce-the-solution)

**Licence** : BSD 3-clause. see Licence.txt

## Overview of my solution

### Logistic Regression
### Bayesian Logistic Regression
### XGBOOST


## Models

## Logistic Regression 

**Features :** 

Table 1.

| Model name | Public Score |  Private Score |
| --- | --- | --- |
| XXX | 0.77081 | 0.74481 |


**Preprocessing :** 


**Features** :

- Feature 


## CV method

Building a representative CV was one of the most challenging tasks in this competition, especially after the leak. We tried many CV approaches and most of them gave us too optimistic AUC scores.


## References

> [3] Temko, A., et al., Performance assessment for EEG-based neonatal seizure detectors, Clin. Neurophysiol. 122 (2011): 474-82.

## Reproduce the solution


# Install
Using Jupyter docker image to not deal with installing dependencies on your machine.
Just install [docker](https://www.docker.com/) and [docker-compose](https://docs.docker.com/compose/), then run `docker-compose up` and you're all set :)

The notebook will be exposed on your local machine at port `8888`.


The code corresponding to each group of model is available in separate folders.
The solution can be reproduced in three steps :

* 1 : place the data in the `data` folder
* 2 : xxx
* 3 : xxx


