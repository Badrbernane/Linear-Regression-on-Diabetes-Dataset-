from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('C:\sokar.csv')

x = data[['Glucose', 'BloodPressure', 'BMI', 'Age']].values 
y = data['Outcome'].values.reshape(-1, 1)  
#print(x.shape)  # Pour vérifier les dimensions de x
#print(y.shape)  # Pour vérifier les dimensions de y



"""la premiere etape c'est la dataset sous sa forme matricille"""
"""la matrice X"""
X = np.hstack((x, np.ones((x.shape[0], 1)))) 
#print(X)
"""maintenant theta qui contient  les paramétres (a et b pour un model simple) """
theta = np.random.randn(X.shape[1], 1)
#print(theta.shape)
#print(theta)
#plt.scatter(x[:,0], y) # afficher les résultats. x_1 en abscisse et y en ordonnée
#plt.title('badr')
#plt.show()

"""fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[:,0], x[:,1], y, c='b', marker='o') # affiche en 3D la variable x_1, x_2, et la target y
# affiche les noms des axes
ax.set_xlabel('x_1')
ax.set_ylabel('x_2')
ax.set_zlabel('y')
plt.show()"""

"""le modéle  F = X.0 """
def model(X, theta):
    return X.dot(theta)
#print(model(X, theta)) #pour tester est ce que tout sa marche bien
#plt.scatter(x, y)
#plt.plot(x, model(X, theta))
#plt.show()

"""La fonction cout"""
def fonction_cout(X, theta, y):
    m = len(y)
    return 1/(2*m) * np.sum((model(X, theta) - y)**2)
print(fonction_cout(X, theta, y))

"""gradient & Descente de gradient"""
# awalan le gradient
def gradient(X, theta, y):
    m = len(y)
    return 1/(2*m) * X.T.dot(model(X, theta) - y)

# maintenant la descente du gradient
def descente_du_gradient(X, theta, y, alpha, n_iter):
    cost_history = np.zeros(n_iter) 
    for i in range(0, n_iter):
        theta = theta - alpha * gradient(X, theta, y)
        cost_history[i] = fonction_cout(X, theta, y) 
    return theta, cost_history

"""  entrainemant """
theta_final, cost_history = descente_du_gradient(X, theta, y, alpha=0.0001, n_iter=1000)
#print(theta_final)
prediction = model(X, theta_final)
"""plt.scatter(x[:,2], y)
plt.plot(x[:,2], prediction, color='red')
plt.show()"""

"""fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[:,0], x[:,1], y, c='b', marker='o') # affiche en 3D la variable x_1, x_2, et la target y
ax.scatter(x[:,0], x[:,1], prediction, c='r', marker='o')
# affiche les noms des axes
ax.set_xlabel('x_1')
ax.set_ylabel('x_2')
ax.set_zlabel('y')
plt.show()"""

""" La courbe d'apprentissage """
#plt.plot(range(len(cost_history)), cost_history)
#plt.show()



"""coeficient de determination"""
def coef_determination(y, pred):
    u = ((y - pred)**2).sum()
    v = ((y - y.mean())**2).sum()
    return 1 - u/v
#coef_determination(y, prediction)
print(coef_determination(y, prediction))
