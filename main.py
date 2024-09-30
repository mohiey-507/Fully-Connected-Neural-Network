
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from deep_neural_network_class import Deep_Neural_Network

def classification_task():
    # Load dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # Normalize the feature data (Standardization: mean=0, std=1)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  
    X_test = scaler.transform(X_test)        

    # Create and train the neural network    
    nn = Deep_Neural_Network([30, 8, 2, 1]) # number of features = 30, Two hidden layers with 8 and 2 neurons, output layer = 1
    nn.compile(optimizer='adam', activation='relu', loss='BinaryCrossentropy')
    nn.fit(X_train, y_train, learning_rate=0.01, lambda_reg=0.05, decay_rate=0.5, batch_size=256, verbose=False)
    nn.evaluate(X_test, y_test, 'Testset')

    print('-'*60)

    # Compare with LogisticRegression
    lg = LogisticRegression()
    lg.fit(X_train, y_train)
    print(f"LogisticRegression - Train accuracy: {lg.score(X_train, y_train):.2%}")
    print(f"LogisticRegression - Test accuracy: {lg.score(X_test, y_test):.2%}")

    print('-'*60)

    # Compare with MLPClassifier
    nnc = MLPClassifier(solver='adam', learning_rate='adaptive', learning_rate_init=0.01)
    nnc.fit(X_train, y_train)
    print(f"MLPClassifier - Train accuracy: {nnc.score(X_train, y_train):.2%}")
    print(f"MLPClassifier - Test accuracy: {nnc.score(X_test, y_test):.2%}")

# -------------------------------------------------------------------------------------------------------------------------------- #   

def regression_task():
    # create a regression dataset
    X = np.linspace(0, np.pi, 1500).reshape(-1, 2)
    y = np.sin(2*np.pi*X[:, 0]) + np.cos(2*np.pi*X[:, 1])

    # split the data into training and testing sets 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # normalize the data using StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # create and train the neural network
    nnr = MLPRegressor(solver='adam', max_iter=1000, random_state=12)
    nnr.fit(X_train, y_train)

    # evaluate the model on the testing set
    y_pred = nnr.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MSE: {mse:.2f}, R2: {r2:.2f}")

    # print sample predictions and actual values
    print(f"Predictions: {y_pred[:5]}")
    print(f"Actual values: {y_test[:5]}")

    print('-'*60)

    # Create and train the neural network    
    nnr = Deep_Neural_Network([2, 32, 8, 1]) 
    nnr.compile(optimizer='adam', activation='relu', loss='MSE')
    nnr.fit(X_train, y_train, epoch=1800, learning_rate=0.002, decay_rate=0.9999, batch_size=128)
    y_pred = nnr.predict(X_test).ravel()

    nnr.evaluate(X_test, y_test, 'Testset')
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred) 
    print(f"MSE: {mse:.2f}, R2: {r2:.2f}")

    # print sample predictions and actual values
    print(f"Predictions: {y_pred[:5]}")
    print(f"Actual values: {y_test[:5]}")

classification_task()

print('-'*80)

regression_task()