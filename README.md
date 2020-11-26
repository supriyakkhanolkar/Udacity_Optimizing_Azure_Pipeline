# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
This dataset contains data about customers who are contacted to market banking products. We seek to predict if the customer will buy the product.
We use 2 approaches to arrive at a solution. The first approach uses Hyperdrive to get best values of hyperparameters for Logistic Regression algorithm. The second approach uses AutoML to get the best model for the same dataset.

The best performing model was Scikit-learn LogisticRegression using Hyperdrive with accuracy of 0.9159462939120838

## Scikit-learn Pipeline
First we create a training script train.pyipynb to take care of following:
*	connect to a web based database to create a tabular dataset
*	clean the dataset
*	split the dataset into test and train data
*	Call scikit-learn LogisticRegression using given values of Regularization Strength and Max Iterations
*	Create model file

Then we create a notebook udacity-project.ipynb to take care of following:
*	Create compute cluster
*	Use HyperDrive to tune hyperparameters using hyper drive config that supplies training script along with other parameters
*	Find out the best model created by HyperDrive and register it

We chose Random Parameter Sampling for following reaasons:
*	It supports discrete as well as continuous hyperparameters
*	It supports early termination of low performance runs


We chose Median Stopping Policy for early stopping because it is a conservative policy that provides savings without terminating promising jobs.


## AutoML
We use the same notebook udacity-project.ipynb to take care of following:
*	Create a tabular dataset using the same database as used by earlier experiment
*	clean the dataset
*	split the dataset into test and train data
*	Create AutoML experiment by providing automl config parameters and submit it
*	We find the optimized model prepared by AutoML and register it

We delete the compute cluster in the end.

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**
Using HyperDrive
----------------
We got following results for best metric using HyperDrive :
Accuracy = 0.9159462939120838
For a comination with 
Regularization Strength = 2.078306649243789, 
Max Iterations = 82
We observed that out of 20 runs of HyperDrive, 17 runs got accuracy close to 0.915 for varying values of Regularization Strength and Max Iterations.

Using Automated ML
------------------
We got following results for best metric using AutoML :
Accuracy = 0.91439
Algorithm : VotingEnsemble

As we can see that the difference between both approaches is less than 1%. Approach using Hyperdrive is marginally better than approach using AutoML in this case.

## Future work
When using Hyperdrive we can do following improvements:
* Do an initial search with random sampling and then refine the search space to improve results
* Use Bayesian sampling

Also, we can make use of model explanations produced by AutoML run to find out top K features that affect the result.

