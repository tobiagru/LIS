#matplotlib inline
from __future__ import print_function
# Provides Matlab-style matrix operations
import numpy as np
# Provides Matlab-style plotting
import matplotlib.pylab as plt
# Contains linear models, e.g., linear regression, ridge regression, LASSO, etc.
import sklearn.linear_model as sklin
# Allows us to create custom scoring functions
import sklearn.metrics as skmet
# Provides train-test split, cross-validation, etc.
import sklearn.cross_validation as skcv
# Provides grid search functionality
import sklearn.grid_search as skgs
# The dataset we will use
from sklearn.datasets import load_boston
# For data normalization
import sklearn.preprocessing as skpr
#------------------------------------------------------------------------------------------
filez = open("train.csv")
filez.readline()
data_train = np.loadtxt(filez,delimiter=",")
print('Shape of X:', data_train.shape)
X = data_train[:,1:15]
Y = data_train[:,16]
print('Shape of X:', X.shape)
print('Shape of Y:', Y.shape)

filezz = open("test.csv")
filezz.readline()
data_test = np.loadtxt(filezz,delimiter=",")
X_k = data_test[:,1:15]

print('Shape of X_k:', X_k.shape)
# We can also (optionally) normalize the data
#X = skpr.scale(X)
#------------------------------------------------------------------------------------------
Xtrain, Xtest, Ytrain, Ytest = skcv.train_test_split(X, Y, train_size=0.75)
print('Shape of Xtrain:', Xtrain.shape)
print('Shape of Ytrain:', Ytrain.shape)
print('Shape of Xtest:', Xtest.shape)
print('Shape of Ytest:', Ytest.shape)
#------------------------------------------------------------------------------------------
XYtrain = np.vstack((Xtrain.T, np.atleast_2d(Ytrain)))
correlations = np.corrcoef(XYtrain)[-1, :]
print( 'correlations', len(correlations))
#for correlation in correlations:
#    print('{1:+.4f}'.format(correlation))
#------------------------------------------------------------------------------------------
regressor = sklin.LinearRegression()
regressor.fit(Xtrain, Ytrain)
print('{0:>10} {1:+.4f}'.format('intercept', regressor.intercept_))
#for coef in regressor.coef_:
#    print('{1:+.4f}'.format(coef))
#------------------------------------------------------------------------------------------
def score(gtruth, pred):
    diff = gtruth - pred
    return np.sqrt(np.mean(np.square(diff)))
#------------------------------------------------------------------------------------------
Ypred = regressor.predict(Xtest)
print('score =', score(Ytest, Ypred))
#------------------------------------------------------------------------------------------
scorefun = skmet.make_scorer(score)
scores = skcv.cross_val_score(regressor, X, Y, scoring=scorefun, cv=5)
print('C-V score =', np.mean(scores), '+/-', np.std(scores))
#------------------------------------------------------------------------------------------
regressor_ridge = sklin.Ridge()
param_grid = {'alpha': np.linspace(0, 100, 10)}
neg_scorefun = skmet.make_scorer(lambda x, y: -score(x, y))  # Note the negative sign.
grid_search = skgs.GridSearchCV(regressor_ridge, param_grid, scoring=neg_scorefun, cv=5)
grid_search.fit(Xtrain, Ytrain)
#------------------------------------------------------------------------------------------
best = grid_search.best_estimator_
print(best)
print('best score =', -grid_search.best_score_)
#------------------------------------------------------------------------------------------
Ypred = best.predict(X_k)
np.savetxt('handin_py_tst.txt', Ypred)
#------------------------------------------------------------------------------------------



n_points = 100
points = np.random.randn(n_points, 2) + 10
x_true = np.mean(points, axis=0).ravel()
print('x_true', x_true)
#------------------------------------------------------------------------------------------
x = np.random.randn(2)
step_size = 0.1
errors = []
for iteration in range(100):
    point = np.mean(points, axis=0).ravel()
    # point = points[np.random.randint(n_points), :].ravel()
    x -= step_size * (x - point)
    errors.append(np.linalg.norm(x - x_true))
plt.plot(errors, 'o--');
#------------------------------------------------------------------------------------------
dim = 10
step_size = 0.1
X = np.random.randn(n_points, dim)
w_truth = np.random.randn(dim)
y = X.dot(w_truth).ravel() + 0.001 * np.random.randn(n_points)
errors = []
w = np.random.randn(dim)
for iteration in range(100):
    idx = np.random.randint(n_points)
    point = X[idx, :].ravel()
    w -= step_size * (w.dot(point) - y[idx]) * point
    errors.append(np.mean(np.square(X.dot(w)-y)))
plt.plot(errors, 'ro--');