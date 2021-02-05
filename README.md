# Geo-locationOfGermanTweets (2nd Place Solution)

This competition was part of the Practical Machine Learning Course from the Artificial Intelligence Masters Program

https://www.kaggle.com/c/pml-2020-unibuc/overview

In this project you can find the next concepts:
- Text cleaning and Lemmatization
- Wrappers over translator APIs
- Word and Character N-Grams
- TF-IDF Features
- Stacked Embeddings
- Adversarial Validation
- Feauture selection based custom metric
- Bagging and Gradient Boosting
- Multi-Layer Stacking Architectures
- Bayesian Optimization
- Weighted Voting Systems

For a better understanding of the pipeline please check the documentation 

# Dataset and Labels Distribution
This is a double regression problem for latitude and longitude coordinates based on nearly 30 thousand german raw tweets 

Latitude             |  Longitude
:-------------------------:|:-------------------------:
![](https://github.com/AdrianIordache/Geo-locationOfGermanTweets/blob/master/images/latitude.png)  |  ![](https://github.com/AdrianIordache/Geo-locationOfGermanTweets/blob/master/images/longitude.png)

# Solution Overview: Multi-Layer Stacking Architecture

This architecture is based on three layers stacking models, based on various types of input
- Level One Models - High Variance Layer
- Level Two Models - Specialization Layer
- Level Three Models - Final Voting Layer

![Cover Image | 1000x800](https://github.com/AdrianIordache/Geo-locationOfGermanTweets/blob/master/images/solution.png)

# Steps for training and improving
- Find new ways to represent the raw data to add variance to Layer One
- Model Selection and Analysis (LGBM, NN, ..) based on custom metric (Weighted Average between MAE Error and Models Correlation)
- Train Selected Models for various hyperparameters 
- Stack OOF Predictions for Layer One
- Retrain Second and Third Level based on previous layer predictions
