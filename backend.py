
import os
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

_devnull = os.open(os.devnull, os.O_WRONLY)
os.dup2(_devnull, 2)
os.close(_devnull)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense
)
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

SEED = 42
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

analysis_table = pd.read_pickle("analysis_table.pkl.gz", compression="gzip")

def plot_classical_measures():
    analysis_table[["volatility", "average_correlation"]].plot(
        subplots=True,
        figsize=(12, 6),
        sharex=True,
        grid=True
    )

    plt.xlabel("Date")
    plt.tight_layout()
    plt.show()

def plot_stress_comparison():

    columns = [
        "H1_raw_stress",
        "H1_normalized_stress",
        "H2_raw_stress",
        "H2_normalized_stress",
        "volatility",
        "average_correlation"
    ]

    scaled = analysis_table[columns].apply(
        lambda x: (x - x.mean()) / x.std()
    )

    comparisons = [
        ("H1_raw_stress", "H1 Raw Stress"),
        ("H1_normalized_stress", "H1 Normalized Stress"),
        ("H2_raw_stress", "H2 Raw Stress"),
        ("H2_normalized_stress", "H2 Normalized Stress")
    ]

    for column, title in comparisons:

        plt.figure(figsize=(10, 4))

        plt.plot(
            analysis_table.index,
            scaled[column],
            label=title
        )

        plt.plot(
            analysis_table.index,
            scaled["volatility"],
            label="Average stock volatility"
        )

        plt.plot(
            analysis_table.index,
            scaled["average_correlation"],
            label="Average correlation"
        )

        plt.axhline(0, linewidth=1)

        plt.xlabel("Date")
        plt.ylabel("Standardized value")
        plt.title(f"{title} and Classical Market Stress Measures")

        plt.legend()
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.show()

def show_correlation_table():
    tda_measures = {
        "H1 raw": "H1_raw_stress",
        "H1 normalized": "H1_normalized_stress",
        "H2 raw": "H2_raw_stress",
        "H2 normalized": "H2_normalized_stress"
    }
    rows = []
    for name, column in tda_measures.items():
        rows.append({
            "TDA measure": name,
            "Pearson with volatility":
                analysis_table[column].corr(
                    analysis_table["volatility"],
                    method="pearson"
                ),
            "Spearman with volatility":
                analysis_table[column].corr(
                    analysis_table["volatility"],
                    method="spearman"
                ),
            "Pearson with average correlation":
                analysis_table[column].corr(
                    analysis_table["average_correlation"],
                    method="pearson"
                ),
            "Spearman with average correlation":
                analysis_table[column].corr(
                    analysis_table["average_correlation"],
                    method="spearman"
                )
        })
    correlation_table = pd.DataFrame(rows)
    display(correlation_table.round(3))

###

def diagram_limits(diagram_column, margin=0.05):
    points = np.vstack(analysis_table[diagram_column])
    min_birth = points[:, 0].min()
    max_birth = points[:, 0].max()
    min_persistence = points[:, 1].min()
    max_persistence = points[:, 1].max()
    birth_range = max_birth - min_birth
    persistence_range = max_persistence - min_persistence
    birth_min = max(0, min_birth - margin * birth_range)
    birth_max = max_birth + margin * birth_range
    persistence_min = max(0, min_persistence - margin * persistence_range)
    persistence_max = max_persistence + margin * persistence_range
    return birth_min, birth_max, persistence_min, persistence_max

H1_raw_limits = diagram_limits("H1")
H2_raw_limits = diagram_limits("H2")

H1_normalized_limits = diagram_limits("H1_normalized")
H2_normalized_limits = diagram_limits("H2_normalized")

def plot_diagram(date, column):
    date = pd.Timestamp(date)
    if column == "H1":
        limits = H1_raw_limits
        xlabel = "Birth"
        ylabel = "Persistence"
    elif column == "H2":
        limits = H2_raw_limits
        xlabel = "Birth"
        ylabel = "Persistence"
    elif column == "H1_normalized":
        limits = H1_normalized_limits
        xlabel = "Normalized birth"
        ylabel = "Normalized persistence"
    elif column == "H2_normalized":
        limits = H2_normalized_limits
        xlabel = "Normalized birth"
        ylabel = "Normalized persistence"
    else:
        raise ValueError("Unknown diagram column")
    diagram = analysis_table.loc[date, column]
    birth_min, birth_max, persistence_min, persistence_max = limits
    plt.figure(figsize=(4, 4))
    plt.scatter(
        diagram[:, 0],
        diagram[:, 1]
    )
    plt.xlim(birth_min, birth_max)
    plt.ylim(persistence_min, persistence_max)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{column} persistence diagram — {date.date()}")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

def plot_heatmap(date, column):
    date = pd.Timestamp(date)
    if column == "H1_raw_heatmap":
        limits = H1_raw_limits
        xlabel = "Birth"
        ylabel = "Persistence"
    elif column == "H2_raw_heatmap":
        limits = H2_raw_limits
        xlabel = "Birth"
        ylabel = "Persistence"
    elif column == "H1_normalized_heatmap":
        limits = H1_normalized_limits
        xlabel = "Normalized birth"
        ylabel = "Normalized persistence"
    elif column == "H2_normalized_heatmap":
        limits = H2_normalized_limits
        xlabel = "Normalized birth"
        ylabel = "Normalized persistence"
    else:
        raise ValueError("Unknown heatmap column")
    heatmap = analysis_table.loc[date, column]
    birth_min, birth_max, persistence_min, persistence_max = limits
    fig, ax = plt.subplots(figsize=(4, 4))
    image = ax.imshow(
        heatmap,
        origin="lower",
        extent=[
            birth_min,
            birth_max,
            persistence_min,
            persistence_max
        ],
        aspect="auto"
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{column} — {date.date()}")
    fig.colorbar(image, ax=ax, label="Intensity")
    plt.tight_layout()
    plt.show()

# ML-workflow

# Heatmap inputs
X_H1_raw = np.stack(
    analysis_table["H1_raw_heatmap"].to_numpy()
)[..., np.newaxis]

X_H1_normalized = np.stack(
    analysis_table["H1_normalized_heatmap"].to_numpy()
)[..., np.newaxis]

X_H2_raw = np.stack(
    analysis_table["H2_raw_heatmap"].to_numpy()
)[..., np.newaxis]

X_H2_normalized = np.stack(
    analysis_table["H2_normalized_heatmap"].to_numpy()
)[..., np.newaxis]


# Targets
y_volatility = analysis_table["volatility"].to_numpy()
y_correlation = analysis_table["average_correlation"].to_numpy()


# Chronological split
n = len(analysis_table)

train_end = int(0.65 * n)
val_end = int(0.85 * n)
gap = 100

# H1 raw
X_H1_raw_train = X_H1_raw[:train_end]
X_H1_raw_val = X_H1_raw[train_end + gap:val_end]
X_H1_raw_test = X_H1_raw[val_end + gap:]


# H1 normalized
X_H1_normalized_train = X_H1_normalized[:train_end]
X_H1_normalized_val = X_H1_normalized[train_end + gap:val_end]
X_H1_normalized_test = X_H1_normalized[val_end + gap:]


# H2 raw
X_H2_raw_train = X_H2_raw[:train_end]
X_H2_raw_val = X_H2_raw[train_end + gap:val_end]
X_H2_raw_test = X_H2_raw[val_end + gap:]


# H2 normalized
X_H2_normalized_train = X_H2_normalized[:train_end]
X_H2_normalized_val = X_H2_normalized[train_end + gap:val_end]
X_H2_normalized_test = X_H2_normalized[val_end + gap:]


# Volatility targets
y_vol_train = y_volatility[:train_end]
y_vol_val = y_volatility[train_end + gap:val_end]
y_vol_test = y_volatility[val_end + gap:]


# Correlation targets
y_corr_train = y_correlation[:train_end]
y_corr_val = y_correlation[train_end + gap:val_end]
y_corr_test = y_correlation[val_end + gap:]

# model architecture

def build_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(8, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),
        Conv2D(16, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    return model


def train_model(model, X_train, y_train, X_val, y_val, epochs=150):
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=30,
        min_delta=1e-5,
        restore_best_weights=True)
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=0
    )
    return history


def plot_test_predictions(model, X_test, y_test, target_name):
    y_pred = model.predict(
        X_test,
        verbose=0
    ).flatten()
    test_dates = analysis_table.index[val_end + gap:]
    plt.figure(figsize=(12, 5))
    plt.plot(
        test_dates,
        y_test,
        label=f"Actual {target_name}"
    )
    plt.plot(
        test_dates,
        y_pred,
        label=f"Predicted {target_name}"
    )
    plt.xlabel("Date")
    plt.ylabel(target_name)
    plt.title(f"CNN Prediction vs. Actual {target_name}")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


def evaluate_test(model, X_test, y_test):

    y_pred = model.predict(X_test, verbose=0).flatten()

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    correlation = np.corrcoef(y_test, y_pred)[0, 1]

    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("R²:", r2)
    print("Correlation:", correlation)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R²": r2,
        "Correlation": correlation
    }

def print_evaluation_table(
    result_H1_raw_volatility,
    result_H1_normalized_volatility,
    result_H2_raw_volatility,
    result_H2_normalized_volatility,
    result_H1_raw_correlation,
    result_H1_normalized_correlation,
    result_H2_raw_correlation,
    result_H2_normalized_correlation
):
    rows = [
        ("H1 raw → Volatility", result_H1_raw_volatility),
        ("H1 normalized → Volatility", result_H1_normalized_volatility),
        ("H2 raw → Volatility", result_H2_raw_volatility),
        ("H2 normalized → Volatility", result_H2_normalized_volatility),
        ("H1 raw → Average correlation", result_H1_raw_correlation),
        ("H1 normalized → Average correlation", result_H1_normalized_correlation),
        ("H2 raw → Average correlation", result_H2_raw_correlation),
        ("H2 normalized → Average correlation", result_H2_normalized_correlation)
    ]
    print(
        f"{'Model':35}"
        f"{'MAE':>10}"
        f"{'MSE':>10}"
        f"{'RMSE':>10}"
        f"{'R²':>10}"
        f"{'Correlation':>14}"
    )
    print("-" * 89)
    for model_name, result in rows:
        print(
            f"{model_name:35}"
            f"{result['MAE']:10.3f}"
            f"{result['MSE']:10.3f}"
            f"{result['RMSE']:10.3f}"
            f"{result['R²']:10.3f}"
            f"{result['Correlation']:14.3f}"
        )

