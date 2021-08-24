import numpy as np                                    
from sklearn import datasets                                                        
from sklearn.linear_model import LinearRegression     
import matplotlib.pyplot as plt
boston = datasets.load_boston()          
X = boston.data                          
Y = boston.target  
M, N = X.shape

# standardization :
std_scaler = preprocessing.StandardScaler()
standardized_X = std_scaler.fit_transform(X)
allOnes = np.ones((len(standardized_X), 1))
standardized_X2 = np.hstack([allOnes, standardized_X])
beta, cost_array = GradientDescent(standardized_X, Y, 0.1, 250, True)
print("R^2:" ,score(standardized_X2,Y,beta))
print("Adjusted R^2 :" , adjScore(standardized_X2,Y,beta,M,N))


#print(score(standardized_X2,Y,beta))
beta = normalEqn(standardized_X2,Y)

print("using normal equation: ")
print()
print("Cost:" , cost(standardized_X2,Y,beta))

print("R^2:" ,score(standardized_X2,Y,beta))

print("Adjusted R^2 :" , adjScore(standardized_X2,Y,beta,M,N))

#cost function calculator
def cost(X, Y, beta):
  return ((Y - (X @ beta))**2).mean()
# predictor function
def predict(X, beta):
  return X @ beta

# A function that finds the R^2 Statistic).
def score(X, Y, beta):
  Y_predicted = predict(X, beta)
  u = ((Y - Y_predicted)**2).sum()
  v = ((Y - Y.mean())**2).sum()
  return 1 - (u/v)

def adjScore(X, Y, beta,M,N):
  return 1-(1-score(X,Y,beta)*(M-1)/(M-N-1))

# the Batch Gradient Descent function.
def GradientDescent(X, Y, alpha, num_iterations, print_cost = False):
  M, N = X.shape
  X2 = X.copy()
  allOnes = np.ones((len(Y), 1))               
  X2 = np.hstack([allOnes, X2]) # Concatenating the allOnes column to X2(for the intercept value).
  #np.random.seed(0)
  beta = np.random.uniform(-10.0, 10.0, N + 1)
  cost_array = []

  for x in range(num_iterations):
    cost_ = cost(X2, Y, beta)
    cost_array.append(cost_)
    if print_cost:
      print("Iteration :", x + 1, '\t', "Cost : " + '%.4f'%cost_)
    slope_array = np.zeros(N + 1)
    for i in range(M):
      f_xi = (beta * X2[i]).sum()
      y_i = Y[i]
      for j in range(N + 1):
        slope_array[j] += (-2/M) * (y_i - f_xi) * X2[i,j]

    beta -= (alpha * slope_array)

  return beta, cost_array

# the normal equation function
def normalEqn(X, y):  
    beta = np.linalg.inv((X.T).dot(X)).dot(X.T).dot(y) 
    return beta # returns array of predictors  

  

