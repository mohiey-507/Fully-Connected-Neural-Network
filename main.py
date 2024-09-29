
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from deep_neural_network_class import Deep_Neural_Network

def main():
    # Load dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

    # Normalize the feature data (Standardization: mean=0, std=1)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  
    X_test = scaler.transform(X_test)        

    # Create and train the neural network    
    nn = Deep_Neural_Network([30, 8, 2, 1]) # number of features = 30, Two hidden layers with 8 and 2 neurons, output layer = 1
    nn.compile(optimizer='adam', activation='relu', loss='BinaryCrossentropy')
    nn.fit(X_train, y_train, learning_rate=0.01, lambda_reg=0.05, decay_rate=0.5, batch_size=256, verbose=False, )
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


    print('test fix_reg')

main()
# -------------------------------------------------------------------------------------------------------------------------------- #   
