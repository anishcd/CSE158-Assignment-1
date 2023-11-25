#!/usr/bin/env python
# coding: utf-8

# In[20]:


get_ipython().system('pip install implicit')
get_ipython().system('pip install surprise')
get_ipython().system('pip install tensorflow')
get_ipython().system('pip install scikit-optimize')
get_ipython().system('pip install scipy')
get_ipython().system('pip install pandas')
import gzip
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
import numpy
import string
import random
import string
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from implicit import bpr
from surprise import SVD, Reader, Dataset
from surprise.model_selection import train_test_split
import tensorflow as tf
import numpy as np


# In[21]:


def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)

def readJSON(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        d = eval(l)
        u = d['userID']
        g = d['gameID']
        yield u,g,d


# In[22]:


allHours = []
for l in readJSON("train.json.gz"):
    allHours.append(l)

hoursTrain = allHours[:165000]
hoursValid = allHours[165000:]

print(len(allHours))


# In[23]:


usersPerItem = defaultdict(set) # Maps an item to the users who rated it
itemsPerUser = defaultdict(set) # Maps a user to the items that they rated

reviewsPerUser = defaultdict(list) # Maps a user to their reviews

for l in allHours:
    user,item,review = l[0], l[1], l[2]
    usersPerItem[item].add(user)
    itemsPerUser[user].add(item)
    reviewsPerUser[user].append(review)

games = set()
for l in allHours:
    games.add(l[1])
gamesList = list(games)


# ## Would Play BPR

# In[ ]:


augValidSet = []
added = set()

for l in hoursValid:
    user, game, review = l[0], l[1], l[2]
    augValidSet.append((user, game, 1))
    added.add((user, game, 1))
    
    randGame = random.sample(gamesList, 1)[0]
    while randGame in itemsPerUser[user] or (user, randGame, 0) in added:
        randGame = random.sample(gamesList, 1)[0]
    
    augValidSet.append((user, randGame, 0))
    added.add((user, randGame, 0))

print(len(hoursValid))
print(len(augValidSet))
print(augValidSet[:3])


# In[ ]:


userIDs,gameIDs = {},{}
interactions = []
userIDlookup, gameIDlookup = {}, {}
games = list(games)

for u, g, r in allHours:
    if not u in userIDs: 
        userIDs[u] = len(userIDs)
        userIDlookup[len(userIDlookup)] = u
    if not g in gameIDs: 
        gameIDs[g] = len(gameIDs)
        gameIDlookup[len(gameIDlookup)] = g

nUsers,nItems = len(userIDs),len(gameIDs)

for u, g, _ in hoursTrain:
    interactions.append((u, g, 1))


# In[ ]:


class BPRbatch(tf.keras.Model):
    def __init__(self, K, lamb):
        super(BPRbatch, self).__init__()
        # Initialize variables
        self.betaI = tf.Variable(tf.random.normal([len(gameIDs)],stddev=0.001))
        self.gammaU = tf.Variable(tf.random.normal([len(userIDs),K],stddev=0.001))
        self.gammaI = tf.Variable(tf.random.normal([len(gameIDs),K],stddev=0.001))
        # Regularization coefficient
        self.lamb = lamb

    # Prediction for a single instance
    def predict(self, u, i):
        p = self.betaI[i] + tf.tensordot(self.gammaU[u], self.gammaI[i], 1)
        return p

    # Regularizer
    def reg(self):
        return self.lamb * (tf.nn.l2_loss(self.betaI) +\
                            tf.nn.l2_loss(self.gammaU) +\
                            tf.nn.l2_loss(self.gammaI))
    
    def score(self, sampleU, sampleI):
        u = tf.convert_to_tensor(sampleU, dtype=tf.int32)
        i = tf.convert_to_tensor(sampleI, dtype=tf.int32)
        beta_i = tf.nn.embedding_lookup(self.betaI, i)
        gamma_u = tf.nn.embedding_lookup(self.gammaU, u)
        gamma_i = tf.nn.embedding_lookup(self.gammaI, i)
        x_ui = beta_i + tf.reduce_sum(tf.multiply(gamma_u, gamma_i), 1)
        return x_ui

    def call(self, sampleU, sampleI, sampleJ):
        x_ui = self.score(sampleU, sampleI)
        x_uj = self.score(sampleU, sampleJ)
        return -tf.reduce_mean(tf.math.log(tf.math.sigmoid(x_ui - x_uj)))


# In[ ]:


optimizer = tf.keras.optimizers.Adam(0.1)
modelBPR = BPRbatch(3, 0.00001)

def trainingStepBPR(model, interactions):
    Nsamples = 50000
    with tf.GradientTape() as tape:
        sampleU, sampleI, sampleJ = [], [], []
        for _ in range(Nsamples):
            u,i,_ = random.choice(interactions) # positive sample
            j = random.choice(games) # negative sample
            while j in itemsPerUser[u]:
                j = random.choice(games)
            sampleU.append(userIDs[u])
            sampleI.append(gameIDs[i])
            sampleJ.append(gameIDs[j])

        loss = model(sampleU,sampleI,sampleJ)
        loss += model.reg()
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients((grad, var) for
                              (grad, var) in zip(gradients, model.trainable_variables)
                              if grad is not None)
    return loss.numpy()


# In[ ]:


for i in range(100):
    obj = trainingStepBPR(modelBPR, interactions)
    if (i % 10 == 9): print("iteration " + str(i+1) + ", objective = " + str(obj))


# In[ ]:


pairsPlayed = {}
for l in open("pairs_Played.csv"):
    if l.startswith("userID"):
        continue
    u,g = l.strip().split(',')
    if (u,g) not in pairsPlayed:
        pairsPlayed[(u,g)] = 0

predictions = []

# Initialize the list of predictions
predictions = []

gameCount = defaultdict(int)
total_played = 0

for user,game,_ in allHours:
  gameCount[game] += 1
  total_played += 1

# Calculate the median of the game counts
median_game_count = np.median(list(gameCount.values()))

user_games_test = defaultdict(list)
# For each user-game pair in the validation set
for user, game in pairsPlayed:
    if user in userIDs and game in gameIDs:
        score = modelBPR.predict(userIDs[user], gameIDs[game]).numpy()
        user_games_test[user].append((game, score))

predSet = {}
for user in user_games_test:
    user_games_test[user] = sorted(user_games_test[user], key=lambda x: x[1], reverse=True)

for user, game in pairsPlayed:
    if user not in userIDs and game in gameIDs:
        # Use dataset statistics or item features to make a prediction
        # For example, if the game is more popular than the median, predict 1, otherwise predict 0
        predSet[(user, game)] = 1 if gameCount[game] > median_game_count else 0
        continue
    elif game not in gameIDs or (user not in userIDs and game not in gameIDs):
        predSet[(user, game)] = 0
        continue
    games_t = user_games_test[user]
    if len(games_t) > 0:
        half = len(games_t) // 2
        # Add the first half of games_t as (user, game): 1
        for game, _ in games_t[:half]:
            predSet[(user, game)] = 1
        # Add the second half of games_t as (user, game): 0
        for game, _ in games_t[half:]:
            predSet[(user, game)] = 0
        
print(len(predSet))
print(predSet[("u25070741", "g03916495")])



# In[ ]:


print(len(userIDs))


# In[ ]:


predictions = open("predictions_Played.csv", 'w')
for l in open("pairs_Played.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,g = l.strip().split(',')
    
    pred = predSet[(u,g)]
    
    _ = predictions.write(u + ',' + g + ',' + str(pred) + '\n')

predictions.close()


# ## Time Played Prediction

# In[5]:


userIDs = {}
gameIDs = {}
interactions = []

for u, g, r in allHours:
    if not u in userIDs: userIDs[u] = len(userIDs)
    if not g in gameIDs: gameIDs[g] = len(gameIDs)
    h = r['hours_transformed']
    interactions.append((u,g,h))

random.shuffle(interactions)
len(interactions)


# In[24]:


nTrain = int(len(interactions) * 0.9)
nTest = len(interactions) - nTrain
interactionsTrain = interactions[:nTrain]
interactionsTest = interactions[nTrain:]

gamesPerUser = defaultdict(list)
usersPerGame = defaultdict(list)
for u,g,r in interactionsTrain:
    gamesPerUser[u].append(g)
    usersPerGame[g].append(u)



trainHours = [r[2]['hours_transformed'] for r in hoursTrain]
globalAverage = sum(trainHours) * 1.0 / len(trainHours)

hoursPerUser = defaultdict(list)
hoursPerItem = defaultdict(list)

hours_trans_list = []

for user, item, data in hoursTrain:
    hours = data['hours_transformed']
    hoursPerUser[user].append((item, hours))
    hoursPerItem[item].append((user, hours))
    hours_trans_list.append(hours)

betaU = {}
for u in hoursPerUser:
    betaU[u] = 0
betaI = {}
for g in hoursPerItem:
    betaI[g] = 0

mu = sum([r for _,_,r in interactionsTrain]) / len(interactionsTrain)
print(mu)

trainHours = np.array(hours_trans_list)

# Calculate the median
ma = np.float32(np.median(hours_trans_list))
print(ma)


# In[25]:


optimizer = tf.keras.optimizers.legacy.Adam(0.1)
actual = [r for _, _, r in interactionsTest]


# In[26]:


class LatentFactorModel(tf.keras.Model):
    def __init__(self, alp, K, lamb_beta, lamb_gamma, betaU_init=None, betaI_init=None):
        super(LatentFactorModel, self).__init__()
        # Initialize to average
        self.alpha = tf.Variable(alp)
        # Initialize to small random values or use provided initial values
        if betaU_init is not None and isinstance(betaU_init, dict):
            betaU_values = [betaU_init.get(uid, tf.random.normal([], stddev=0.001)) for uid in userIDs]
            self.betaU = tf.Variable(betaU_values)
        else:
            self.betaU = tf.Variable(tf.random.normal([len(userIDs)], stddev=0.001))
        
        # Initialize betaI
        if betaI_init is not None and isinstance(betaI_init, dict):
            betaI_values = [betaI_init.get(iid, tf.random.normal([], stddev=0.001)) for iid in gameIDs]
            self.betaI = tf.Variable(betaI_values)
        else:
            self.betaI = tf.Variable(tf.random.normal([len(gameIDs)], stddev=0.001))
        self.gammaU = tf.Variable(tf.random.normal([len(userIDs),K],stddev=0.001))
        self.gammaI = tf.Variable(tf.random.normal([len(gameIDs),K],stddev=0.001))
        self.lamb_beta = lamb_beta
        self.lamb_gamma = lamb_gamma

    # Prediction for a single instance (useful for evaluation)
    def predict(self, u, i):
        p = self.alpha + self.betaU[u] + self.betaI[i] +\
            tf.tensordot(self.gammaU[u], self.gammaI[i], 1)
        return p

    # Regularizer
    def reg(self):
        return self.lamb_beta * (tf.reduce_sum(self.betaU**2) + tf.reduce_sum(self.betaI**2)) +\
               self.lamb_gamma * (tf.reduce_sum(self.gammaU**2) + tf.reduce_sum(self.gammaI**2))
    
    # Prediction for a sample of instances
    def predictSample(self, sampleU, sampleI):
        u = tf.convert_to_tensor(sampleU, dtype=tf.int32)
        i = tf.convert_to_tensor(sampleI, dtype=tf.int32)
        beta_u = tf.nn.embedding_lookup(self.betaU, u)
        beta_i = tf.nn.embedding_lookup(self.betaI, i)
        gamma_u = tf.nn.embedding_lookup(self.gammaU, u)
        gamma_i = tf.nn.embedding_lookup(self.gammaI, i)
        pred = self.alpha + beta_u + beta_i +\
               tf.reduce_sum(tf.multiply(gamma_u, gamma_i), 1)
        return pred
    
    # Loss
    def call(self, sampleU, sampleI, sampleR):
        pred = self.predictSample(sampleU, sampleI)
        r = tf.convert_to_tensor(sampleR, dtype=tf.float32)
        return tf.nn.l2_loss(pred - r) / len(sampleR)
    
def trainingStep(model, interactions):
    Nsamples = 50000
    with tf.GradientTape() as tape:
        sampleU, sampleI, sampleR = [], [], []
        for _ in range(Nsamples):
            u,i,h = random.choice(interactions)
            sampleU.append(userIDs[u])
            sampleI.append(gameIDs[i])
            sampleR.append(h)

        loss = model(sampleU,sampleI,sampleR)
        loss += model.reg()
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients((grad, var) for
                              (grad, var) in zip(gradients, model.trainable_variables)
                              if grad is not None)
    return loss.numpy()


# In[27]:


def initialize_parameters(alpha_init=None, betaU_init=None, betaI_init=None):
    global alpha, betaU, betaI
    alpha = alpha_init if alpha_init is not None else ma
    if betaU_init is not None:
        betaU = betaU_init
    else:
        betaU = betaU
    if betaI_init is not None:
        betaI = betaI_init
    else:
        betaI = betaI
def iterate(lamb, alpha_init=None, betaU_init=None, betaI_init=None):
    global alpha, betaU, betaI
    alpha = alpha_init if alpha_init is not None else alpha
    betaU = betaU_init if betaU_init is not None else betaU
    betaI = betaI_init if betaI_init is not None else betaI

    iteration = 0
    while True:
        alpha_old = alpha
        betaU_old = betaU.copy()
        betaI_old = betaI.copy()

        alpha_sum = 0
        for user, item, hours in interactionsTrain:
            alpha_sum += hours - betaU[user] - betaI[item]

        for user in betaU:
            betaU[user] = sum([hours - alpha_old - betaI_old[game] for game, hours in hoursPerUser[user]]) / (lamb + len(hoursPerUser[user]))

        for item in betaI:
            betaI[item] = sum([hours - alpha_old - betaU[user] for user, hours in hoursPerItem[item]]) / (lamb + len(hoursPerItem[item]))

        alpha = alpha_sum / len(interactionsTrain)

        print(f"Iteration {iteration}:")
        print(f"Alpha difference: {abs(alpha - alpha_old)}")
        print(f"Max betaU difference: {max(abs(betaU[user] - betaU_old[user]) for user in betaU)}")
        print(f"Max betaI difference: {max(abs(betaI[item] - betaI_old[item]) for item in betaI)}")

        iteration += 1

        if abs(alpha - alpha_old) < 5e-3 and all(abs(betaU[user] - betaU_old[user]) < 5e-3 for user in betaU) and all(abs(betaI[item] - betaI_old[item]) < 5e-3 for item in betaI):
            break


# In[16]:


initialize_parameters()
iterate(5)
validMSE = sum([(h - alpha - betaU[u] - betaI[i])**2 for u, i, h in interactionsTest]) / len(interactionsTest)
print("MSE on the validation set: ", validMSE)


# In[19]:


iterate(3, alpha, betaU, betaI)
validMSE = sum([(h - alpha - betaU[u] - betaI[i])**2 for u, i, h in interactionsTest]) / len(interactionsTest)
print("MSE on the validation set: ", validMSE)


# In[132]:


betaU_in = betaU.copy()
betaI_in = betaI.copy()
alpha_in = alpha


# In[133]:


betaU_in


# In[135]:


betaI_in


# In[136]:


alpha_in = np.float32(alpha_in)

alpha_in


# In[142]:


delta = -.002  # Threshold for worsening MSE
n = 5  # Look-back window for MSE values
patience = 3  # Number of iterations to wait after early stopping condition is met
mse_values = [float('inf')] * n  # Initialize the list with infinity
patience_counter = 0  # Counter for patience after early stopping condition is met

# ma, 1, 0.000012, 0.00041

modelLFM1 = LatentFactorModel(ma, 1, 0.00002, 0.000503, betaU_in, betaI_in)

for i in range(200):
    trainingStep(modelLFM1, interactionsTrain)
    if (i % 10 == 9):
        predictions = [modelLFM1.predict(userIDs[u], gameIDs[i]).numpy() for u, i, _ in interactionsTest]
        mse = mean_squared_error(actual, predictions)
        print("iteration " + str(i+1) + ", MSE = " + str(mse))
        
        # Update the list of MSE values and check worsening condition
        mse_values.pop(0)
        mse_values.append(mse)

        # Check if the current MSE is worse than the best MSE in the look-back window by a certain threshold
        if i > 0 and all(mse > min(mse_values) for past_mse in mse_values):
            patience_counter += 1
            print(f"MSE worsening detected at iteration {i+1}. Patience counter: {patience_counter}")
            if patience_counter >= patience:
                print("Early stopping, MSE no longer improving")
                break
        else:
            patience_counter = 0  # Reset patience counter if MSE improves


# In[36]:


predictions = [modelLFM1.predict(userIDs[u], gameIDs[i]).numpy() for u, i, _ in interactionsTest]

mse = mean_squared_error(actual, predictions)

print("MSE: ", mse)


# In[ ]:


# u11400327,g33702735
modelLFM1.predict(userIDs['u11400327'], gameIDs['g33702735']).numpy()


# In[28]:


from surprise import SVD, Reader, Dataset
from surprise.model_selection import train_test_split
get_ipython().run_line_magic('pip', 'install pandas')
import pandas as pd


# In[29]:


interactions


# In[32]:


# 0.000012, 0.00041
model = SVD(n_factors=1, reg_bu = 0.000012, reg_bi = 0.000012, reg_pu = 0.00041, reg_qi = 0.00041)

min_hours_transformed = 10000000
max_hours_transformed = 0
for u, g, h in allHours:
    hours = h['hours_transformed']
    min_hours_transformed = min(min_hours_transformed, hours)
    max_hours_transformed = max(max_hours_transformed, hours)

reader = Reader(rating_scale=(min_hours_transformed, max_hours_transformed))

# Convert the list to a DataFrame
interactions_df = pd.DataFrame(interactions, columns=['userID', 'gameID', 'hours_transformed'])


# In[33]:


print(predictions[0].est)

sse = 0
for p in predictions:
    sse += (p.r_ui - p.est)**2

print(sse / len(predictions))


# In[35]:


from skopt import BayesSearchCV
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from surprise import accuracy
from surprise.model_selection import PredefinedKFold
from surprise import Dataset, Reader, SVD
get_ipython().run_line_magic('pip', 'install optuna')
import optuna
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import cross_validate, KFold

min_hours_transformed = 10000000
max_hours_transformed = 0
for u, g, h in allHours:
    hours = h['hours_transformed']
    min_hours_transformed = min(min_hours_transformed, hours)
    max_hours_transformed = max(max_hours_transformed, hours)

def objective(trial):
    n_factors = trial.suggest_int('n_factors', 1, 10)
    n_epochs = trial.suggest_int('n_epochs', 1, 50)
    lr_all = trial.suggest_float('lr_all', 0.001, 0.1)
    reg_bu = trial.suggest_float('reg_bu', 0.000005, 0.001)
    reg_bi = trial.suggest_float('reg_bi', 0.000005, 0.001)
    reg_qi = trial.suggest_float('reg_qi', 0.00001, 0.01)
    reg_pu = trial.suggest_float('reg_pu', 0.00001, 0.01)

    svd = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_bu=reg_bu, reg_bi = reg_bi, reg_qi = reg_qi, reg_pu=reg_pu, biased=True)
    
    reader = Reader(rating_scale=(min_hours_transformed, max_hours_transformed))
    data = Dataset.load_from_df(interactions_df[['userID', 'gameID', 'hours_transformed']], reader)
    
    kf = KFold(n_splits=3)
    mse = []
    for trainset, testset in kf.split(data):
        svd.fit(trainset)
        predictions = svd.test(testset)
        mse_score = accuracy.mse(predictions, verbose=False)
        mse.append(mse_score)
    avg_mse = sum(mse) / len(mse)
    print(f"Average MSE for this trial: {avg_mse}")
    return avg_mse

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=200)

print(study.best_params)


# In[37]:


predictions = open("predictions_Hours.csv", 'w')
for l in open("pairs_Hours.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,g = l.strip().split(',')
    
    pred = modelLFM1.predict(userIDs[u], gameIDs[g]).numpy()
    
    _ = predictions.write(u + ',' + g + ',' + str(pred) + '\n')

predictions.close()


# In[12]:


predictions = open("predictions_Hours.csv", 'w')
for l in open("pairs_Hours.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,g = l.strip().split(',')
    
    bu = betaU.get(u, 0)
    bi = betaI.get(g, 0)
    
    _ = predictions.write(u + ',' + g + ',' + str(alpha + bu + bi) + '\n')

predictions.close()

