

##############################################################################################################



# Principle Component Analysis
import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
iris = load_iris()
x = iris.data
y = iris.target
pca = PCA(n_components=2)
X_pca = pca.fit_transform(x)
print(pca.explained_variance_ratio_)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()



##############################################################################################################



# single layer perceptron
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train/255
x_test = x_test/255
x_train_flatten = x_train.reshape(len(x_train), 28*28)
x_test_flatten = x_test.reshape(len(x_test), 28*28)
model = keras.Sequential([
	keras.layers.Dense(10, input_shape=(784,),
					activation='sigmoid')
])
model.compile(
	optimizer='adam',
	loss='sparse_categorical_crossentropy',
	metrics=['accuracy'])

model.fit(x_train_flatten, y_train, epochs=5)
model.evaluate(x_test_flatten, y_test)



##############################################################################################################




# SVM
import numpy as nm  
import matplotlib.pyplot as mtp  
import pandas as pd  
from sklearn.svm import SVC   
from sklearn.preprocessing import StandardScaler    
from sklearn.model_selection import train_test_split  
from sklearn.metrics import confusion_matrix  
data_set= pd.read_csv('user_data.csv')  
x= data_set.iloc[:, [2,3]].values  
y= data_set.iloc[:, 4].values  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=0)  
st_x= StandardScaler()    
x_train= st_x.fit_transform(x_train)    
x_test= st_x.transform(x_test)  
classifier = SVC(kernel='linear', random_state=0)  
classifier.fit(x_train, y_train)  
y_pred= classifier.predict(x_test)





##############################################################################################################




# Error Backpropogation 
import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)

input_size = 2
hidden_size = 3
output_size = 1
w1 = np.random.randn(input_size, hidden_size)
w2 = np.random.randn(hidden_size, output_size)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
learning_rate = 0.1
num_iterations = 10000
for i in range(num_iterations):
    z1 = np.dot(X, w1)
    a1 = sigmoid(z1)
    z2 = np.dot(a1, w2)
    y_pred = sigmoid(z2)
    
    error = y - y_pred
    delta2 = error * sigmoid_derivative(y_pred)
    delta1 = np.dot(delta2, w2.T) * sigmoid_derivative(a1)
  
    w2 += learning_rate * np.dot(a1.T, delta2)
    w1 += learning_rate * np.dot(X.T, delta1)

print("Final predictions:")
print(y_pred)




##############################################################################################################




# Hebbian Learning
def hebbian_learning(samples):
     print(f'{"INPUT":^8} {"TARGET":^16}{"WEIGHT CHANGES":^15}{"WEIGHTS":^25}')
     w1, w2, b = 0, 0, 0
     print(' ' * 45, f'({w1:2}, {w2:2}, {b:2})')
     for x1, x2, y in samples:
         w1 = w1 + x1 * y
         w2 = w2 + x2 * y
         b = b + y
         print(f'({x1:2}, {x2:2}) {y:2} ({x1:2}, {x2:2}, {y:2}) ({w1:2}, {w2:2}, {b:2})')

AND_samples = {
    'binary_input_binary_output': [
        [1, 1, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ],
    'binary_input_bipolar_output': [
        [1, 1, 1],
        [1, 0, -1],
        [0, 1, -1],
        [0, 0, -1]
    ],
    'bipolar_input_bipolar_output': [
        [ 1, 1, 1],
        [ 1, -1, -1],
        [-1, 1, -1],
        [-1, -1, -1]
    ]
}

print('AND with Binary Input and Binary Output')
hebbian_learning(AND_samples['binary_input_binary_output'])





##############################################################################################################




# Linear Regression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
np.random.seed(0)
X = np.random.randn(100, 1)

y = 2*X + np.random.randn(100, 1)
model = LinearRegression()

model.fit(X, y)
plt.scatter(X, y, color='red')
X_test = np.linspace(-3, 3, 100).reshape(-1, 1)
y_pred = model.predict(X_test)
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.xlabel('Input')
plt.ylabel('Output')
plt.show()




##############################################################################################################




# Logistic Regression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
np.random.seed(0)
X = np.random.randn(100, 1)

y = (X > 0).astype(int).ravel()
model = LogisticRegression()


model.fit(X, y)
plt.scatter(X, y, color='red')
X_test = np.linspace(-3, 3, 100).reshape(-1, 1)
y_pred = model.predict(X_test)
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.xlabel('Input')
plt.ylabel('Class')
plt.legend(['Logistic Regression', 'Data'])
plt.show()












##################################################################################################################################







# MP
from sklearn.neural_network import MLPRegressor
import numpy as np
X_train = np.array([[0,0], [0,1], [1, 0], [1, 1] ])
y_train = np.array([0,1,1,0])
clf = MLPRegressor(hidden_layer_sizes=(2, ), activation='logistic', solver='sgd')
clf.fit(X_train, y_train)
X_test = np.array([[0,0], [0,1], [1, 0], [1, 1] ])
y_pred = clf.predict(X_test)
print(y_pred)



##############################################################################################################



from sklearn.linear_model import Perceptron
X = [[0,0 ], [0, 1], [1, 0], [1, 1]] 
y = [0, 0, 1, 0]
per = Perceptron() 
per.fit(X, y)
w = per.coef_
y1 = X[0][0]*w[0][0]+X[0][1]*w[0][1] 
y2 = X[1][0]*w[0][0]+X[1][1]*w[0][1] 
y3 = X[2][0]*w[0][0]+X[2][1]*w[0][1] 
y4 = X[3][0]*w[0][0]+X[3][1]*w[0][1]
def act(x1, x2):
    yin = x1*w[0][0]+x2*w[0][1]
    if yin >=2:
        return 1
    else:
        return 0
print("give x1 and x2 values for prediction: ")
x1 = 0
x2 = 1
a = act (x1, x2)
print ("Activation output: ",a)




##############################################################################################################




from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# create sample data
X_train = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9],[7, 8,3]])
y_train = np.array([0, 1, 1])

X_test = np.array([[2, 3, 4], [5, 6, 7]])
y_test = np.array([0, 1])

# create a PCA object with 2 principal components
pca = PCA(n_components=2)

# fit and transform the training data
X_train_pca = pca.fit_transform(X_train)

# transform the testing data using the same PCA object
X_test_pca = pca.transform(X_test)

# plot the transformed training data
plt.subplot(1, 2, 1)
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Training Data')

# plot the transformed testing data
plt.subplot(1, 2, 2)
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Testing Data')

plt.tight_layout()
plt.show()

