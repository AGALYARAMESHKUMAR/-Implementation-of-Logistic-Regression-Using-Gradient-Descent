# Implementation-of-Logistic-Regression-Using-Gradient-Descent

# AIM:

To write a program to implement the the Logistic Regression Using Gradient Descent.
Equipments Required:

    Hardware – PCs
    Anaconda – Python 3.7 Installation / Jupyter notebook

# Algorithm

    Use the standard libraries in python for finding linear regression.
    Set variables for assigning dataset values.
    Import linear regression from sklearn.
    Predict the values of array.
    Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
    Obtain the graph.

# Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Agalya
RegisterNumber:  212222040003
*/
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data = np.loadtxt("/content/ex2data1.txt", delimiter=',')
X = data[:,[0,1]]
y = data[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y == 1][:,0],X[y == 1][:,1], label="Admitted")
plt.scatter(X[y == 0][:,0],X[y == 0][:,1],label ="Not admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
  return 1/(1+np.exp(-z))

plt.plot()
X_plot = np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costfunction(theta,X,y):
  h = sigmoid(np.dot(X,theta))
  j = -(np.dot(y,np.log(h)) + np.dot(1-y,np.log(1-h))) / X.shape[0]
  grad = np.dot(X.T, h-y)/ X.shape[0]
  return j,grad

X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta =np.array([0,0,0])
j,grad = costfunction(theta,X_train,y)
print(j)
print(grad)

X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta = np.array([-24,0.2,0.2])
j,grad = costfunction(theta,X_train,y)
print(j)
print(grad)

def cost(theta,X,y):
  h = sigmoid(np.dot(X,theta))
  j = -(np.dot(y,np.log(h)) + np.dot(1-y,np.log(1-h)))/X.shape[0]
  return j

def gradient(theta,X,y):
  h = sigmoid(np.dot(X,theta))
  grad = np.dot(X.T,h-y)/X.shape[0]
  return grad

X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta = np.array([0,0,0])
res = optimize.minimize(fun=cost, x0=theta, args=(X_train,y),method='Newton-CG',jac=gradient)

print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
  x_min, x_max = X[:,0].min()-1, X[:,0].max() +1
  y_min, y_max = X[:,1].min()-1, X[:,1].max() +1
  xx,yy =np.meshgrid(np.arange(x_min, x_max,0.1),np.arange(y_min,y_max, 0.1))
  X_plot = np.c_[xx.ravel(), yy.ravel()]
  X_plot = np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
  y_plot = np.dot(X_plot,theta).reshape(xx.shape)
  plt.figure()
  plt.scatter(X[y == 1][:,0],X[y== 1][:,1],label="Admitted")
  plt.scatter(X[y== 0][:,0],X[y ==0][:,1],label="Not admitted")
  plt.contour(xx,yy,y_plot,levels =[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()

plotDecisionBoundary(res.x,X,y)

prob = sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
  X_train = np.hstack((np.ones((X.shape[0],1)),X))
  prob=sigmoid(np.dot(X_train,theta))
  return (prob >=0.5).astype(int)

np.mean(predict(res.x,X)==y)
```

# Output:
# Array Value of x
![image](https://github.com/AGALYARAMESHKUMAR/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119394395/77a2e421-8b3f-492c-8b2a-51ded8aca180)

# Array Value of y
![image](https://github.com/AGALYARAMESHKUMAR/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119394395/1e1b9936-2c8d-4447-9ef1-e4c086847f70)

# Exam 1 - score graph
![image](https://github.com/AGALYARAMESHKUMAR/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119394395/c3ee6c48-7250-4bab-bc60-70f1f59f36cb)

# Sigmoid function graph
![image](https://github.com/AGALYARAMESHKUMAR/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119394395/9932b9ba-15d1-4295-81c1-45478c3e49b0)

# X_train_grad value
![image](https://github.com/AGALYARAMESHKUMAR/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119394395/cc92ffc3-6077-429d-92ee-e5fbbbf93dcd)

# Y_train_grad value
![image](https://github.com/AGALYARAMESHKUMAR/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119394395/2be2efc5-2fef-4797-afef-823d0ffaa9d6)

# Print res.x
![image](https://github.com/AGALYARAMESHKUMAR/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119394395/e0377475-9312-45e4-9714-6e465baec4ab)

# Decision boundary - graph for exam score
![image](https://github.com/AGALYARAMESHKUMAR/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119394395/a0927c58-3711-49d0-9bc4-68a5672edf13)

# Probablity value
![image](https://github.com/AGALYARAMESHKUMAR/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119394395/13d0b304-39df-4272-927a-3d6434cc9588)

# Prediction value of mean
![image](https://github.com/AGALYARAMESHKUMAR/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119394395/d1a20e9f-19c5-463f-b8af-b5793e9ea1c9)

# Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
