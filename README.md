# CSE158-Assignment-1
Assignment 1 for CSE 158/258 - Web Mining and Recommender Systems taught by Prof. Julian McAuley, Fall '23.

[Assignment Writeup](https://cseweb.ucsd.edu/classes/fa23/cse258-a/files/assignment1.pdf)

## Play Prediction
Utilized [Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/pdf/1205.2618.pdf) for (user, game) pairs implemented through a 'vanilla' Tensorflow construction. Handled cold starts of users by calculating the popularity counts (times a game had been played in the training data) of all the games trained, and predicting as true if it was greater than the median popularity. In the case of a game cold start, model defaults to a prediction of 0. ***Play Prediction BPR model achieved an average prediction accuracy of 0.7612, which ranked 4th out of 600 students.***

## Time Played Prediction
Trained a Latent Factor Model, designed to work for the relatively sparse datasets such as this. Implemented the [Surprise SVD++](https://surprise.readthedocs.io/en/stable/matrix_factorization.html) Algorithm, and fine-tuned hyperparamaters using Bayesian Optimization methods. Implemented early-stopping on the validation set to avoid overfitting, and ***achieved an average MSE of 3.0345, which ranked in the top 10% of the class***

## To Train Model Locally:
Download `train.json.gz` from the writeup link, and run the assignment1.py code to train models for both prediction tasks

## Installation/Dependencies

Implicit and Surprise can be installed from pypi with
```
pip install implicit
pip install surprise
```






