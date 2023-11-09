import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics  import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

def main():
    # Load data from file
    data_df = load_data("data.csv")
    class_labels = data_df["class"].unique()

    # Normalize data
    normalized_data_df = normalize_data(data_df)

    # Get train and test sets
    X_train, X_test, y_train, y_test = split_data(normalized_data_df)

    configs = get_configs()
    print(configs)

    for i, config in enumerate(configs):
        print(f"== Model {i + 1}")
        # Create the model
        mlp = MLPClassifier(**config)

        # Train the model
        mlp.fit(X_train, y_train)

        # Test the model
        y_pred = mlp.predict(X_test)

        # Model metrics
        accuracy = accuracy_score(y_test, y_pred)
        print(f"  Accuracy: {accuracy}")
        cm = confusion_matrix(y_test, y_pred, labels=class_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
        disp.plot()
        plt.show()

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def normalize_data(data: pd.DataFrame, range=(-1, 1)):
    features = data.drop(["audio_name", "class"], axis=1)
    scaler = MinMaxScaler(range)
    normalized_features = scaler.fit_transform(features)
    data_normalized = pd.DataFrame(data=normalized_features, columns=features.columns)
    data_normalized["class"] = data["class"]
    return data_normalized

def split_data(data, test_size=0.33, seed=42):
    X = data.drop(["class"], axis=1)
    y = data["class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    return X_train, X_test, y_train, y_test

def get_configs():
    default = {
        "random_state": 42,
        "max_iter": 1000,
        "solver": "adam"
    }
    # Define a configuração do MLP
    mlp_config_0 = {
        "hidden_layer_sizes": (128, 64),
        "activation": "relu",
        **default
    }

    mlp_config_1 = {
        "hidden_layer_sizes": (128, 64),
        "activation": "tanh",
        **default
    }

    mlp_config_2 = {
        "hidden_layer_sizes": (128, 64, 32),
        "activation": "relu",
        **default
    }

    return [mlp_config_0, mlp_config_1, mlp_config_2]

if __name__ == "__main__":
    main()