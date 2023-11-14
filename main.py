import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics  import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

def main():
    # Load data from file
    data_df = load_data("data.csv")

    # Normalize data
    normalized_data_df = normalize_data(data_df)

    # Extract features and target variable
    X = normalized_data_df.drop(["class"], axis=1) # features
    y = normalized_data_df["class"] # targets
    class_labels = y.unique()

    # Create the model
    mlp = MLPClassifier()

    # Use GridSearchCV to find the best configuration
    param_grid = get_parameters_grid()
    grid_search = GridSearchCV(mlp, param_grid, cv=5)
    grid_search.fit(X, y)

    # Get the best configuration
    best_config = grid_search.best_params_
    print(f"Best Model Configuration: {best_config}")

    # Get the best model trained
    best_mlp = grid_search.best_estimator_

    # Test the model using cross validation
    y_pred = cross_val_predict(best_mlp, X, y, cv=5)

    # Model metrics
    accuracy = accuracy_score(y, y_pred)
    print(f"Accuracy: {accuracy}")

    # Confusion matrix
    cm = confusion_matrix(y, y_pred, labels=class_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot()
    plt.show()

# Loads data from a CSV file into a Pandas DataFrame.
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Normalizes features using Min-Max scaling
def normalize_data(data: pd.DataFrame, range=(-1, 1)):
    features = data.drop(["audio_name", "class"], axis=1)
    scaler = MinMaxScaler(range)
    normalized_features = scaler.fit_transform(features)

    # Create a new DataFrame with normalized features and the target variable
    data_normalized = pd.DataFrame(data=normalized_features, columns=features.columns)
    data_normalized["class"] = data["class"]
    return data_normalized

# Define a grid of hyperparameters for the MLP model to get the best one
def get_parameters_grid():
    return {
        'hidden_layer_sizes': [(14), (14, 7), (128), (128, 64)],
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'max_iter': [10000],
        'solver': ['sgd', 'adam'],
        'random_state': [42],
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'learning_rate_init': [0.001, 0.01, 0.1]
    }

if __name__ == "__main__":
    main()