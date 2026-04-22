from __future__ import annotations

"""Task 2 — Multi-step forecasting strategies (Keras).

Goal
----
Compare forecast drift on horizon H=100 using:
1) One-step model (predicts x[t+1]) rolled out recursively with stride=1.
2) K-step model (K=20, predicts x[t+1:t+K]) rolled out recursively with stride=20.
3) The same K-step model rolled out with stride=1, using only the first predicted step each time.

Students implement the core pipeline:
- make_windows
- time_split
- build_model
- train_model
- recursive_rollout_one_step
- recursive_rollout_k_step_stride_k
- recursive_rollout_k_step_stride_1

Everything else (metrics, evaluation helpers, demo plotting) is provided.

Important
---------
- Use time-based split (NO shuffle) to avoid data leakage.
- The difference between strategies is *inference-time usage*, not only the model.
"""

from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# ----------------------------
# Metrics (provided)
# ----------------------------

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth values. Shape: (N,) or (N, 1).
    y_pred : np.ndarray
        Predicted values. Same shape as y_true.

    Returns
    -------
    float
        MAE.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth values. Shape: (N,) or (N, 1).
    y_pred : np.ndarray
        Predicted values. Same shape as y_true.

    Returns
    -------
    float
        RMSE.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ----------------------------
# Data helpers (students implement)
# ----------------------------

def make_windows(series: np.ndarray, window: int, horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Створюємо вікна та багато крокові цілі."""
    X, y = [], []
    for i in range(len(series) - window - horizon + 1):
        X.append(series[i : i + window])
        y.append(series[i + window : i + window + horizon])
    
    X = np.array(X).reshape(-1, window, 1)
    y = np.array(y)
    
    if horizon == 1:
        y = y.reshape(-1, 1)
        
    return X, y


def time_split(
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Спліт за часом без перемішування."""
    n = len(X)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    if train_end == 0 or val_end <= train_end or val_end >= n:
        if n < 3: raise ValueError("Not enough data to split.")

    return (X[:train_end], y[:train_end]), \
           (X[train_end:val_end], y[train_end:val_end]), \
           (X[val_end:], y[val_end:])


def build_model(
    window: int,
    output_dim: int,
    n_units: int = 64,
    dense_units: int = 32,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """LSTM модель, що підтримує вихідний вектор розміром output_dim."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(window, 1)),
        tf.keras.layers.LSTM(n_units),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(dense_units, activation="relu"),
        tf.keras.layers.Dense(output_dim) 
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"]
    )
    return model


def train_model(
    series: np.ndarray,
    window: int,
    horizon: int,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    epochs: int = 25,
    batch_size: int = 64,
    seed: int = 42,
    verbose: int = 0,
) -> Tuple[tf.keras.Model, np.ndarray, np.ndarray]:
    """Пайплан навчання для заданого горизонту."""
    tf.keras.utils.set_random_seed(seed)
    
    X, y = make_windows(series, window, horizon)
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = time_split(X, y, train_frac, val_frac)
    
    model = build_model(window, output_dim=horizon)
    
    stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=[stop]
    )
    
    return model, X_te, y_te


# ----------------------------
# Model helpers (students implement)
# ----------------------------

def build_model(
    window: int,
    output_dim: int,
    n_units: int = 64,
    dense_units: int = 32,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """Будуємо LSTM модель для вектора виходу розміром output_dim."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(window, 1)),
        tf.keras.layers.LSTM(n_units),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(dense_units, activation="relu"),
        tf.keras.layers.Dense(output_dim)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"]
    )
    return model


def train_model(
    series: np.ndarray,
    window: int,
    horizon: int,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    epochs: int = 25,
    batch_size: int = 64,
    seed: int = 42,
    verbose: int = 0,
) -> Tuple[tf.keras.Model, np.ndarray, np.ndarray]:
    """Пайплайн навчання вікна в спліт в модель в fit."""
    tf.keras.utils.set_random_seed(seed)
    
    X, y = make_windows(series, window, horizon)
    
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = time_split(X, y, train_frac, val_frac)
    
    model = build_model(window, output_dim=horizon)
    
    stop_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True
    )
    
    model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=[stop_callback]
    )
    
    return model, X_te, y_te


# ----------------------------
# Rollout strategies (students implement)
# ----------------------------

def recursive_rollout_one_step(
    model: tf.keras.Model,
    init_window: np.ndarray,
    horizon: int = 100,
) -> np.ndarray:
    """Стратегія 1: Прогнозуємо по 1 кроку, додаємо в кінець, зсуваємо."""
    current_window = init_window.copy()
    forecast = []
    window_size = len(init_window)

    for _ in range(horizon):
        pred = model.predict(current_window.reshape(1, window_size, 1), verbose=0)
        val = pred[0, 0]
        forecast.append(val)
        current_window = np.append(current_window[1:], val)
        
    return np.array(forecast)


def recursive_rollout_k_step_stride_k(
    model: tf.keras.Model,
    init_window: np.ndarray,
    k: int = 20,
    horizon: int = 100,
) -> np.ndarray:
    """Стратегія 2: Прогнозуємо блоками по K значень."""
    current_window = init_window.copy()
    forecast = []
    window_size = len(init_window)

    for _ in range(0, horizon, k):
        pred_block = model.predict(current_window.reshape(1, window_size, 1), verbose=0)
        block = pred_block[0] # Отримуємо вектор довжиною k
        forecast.extend(block)
        # Зсуваємо вікно одразу на k елементів
        current_window = np.append(current_window[k:], block)
        
    return np.array(forecast)[:horizon]


def recursive_rollout_k_step_stride_1(
    model: tf.keras.Model,
    init_window: np.ndarray,
    k: int = 20,
    horizon: int = 100,
) -> np.ndarray:
    """Стратегія 3: Прогнозуємо блоком K, але беремо лише перше значення."""
    current_window = init_window.copy()
    forecast = []
    window_size = len(init_window)

    for _ in range(horizon):
        pred_block = model.predict(current_window.reshape(1, window_size, 1), verbose=0)
        val = pred_block[0, 0]
        forecast.append(val)
        current_window = np.append(current_window[1:], val)
        
    return np.array(forecast)


# ----------------------------
# Evaluation + plotting (provided)
# ----------------------------

def horizon_errors(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MAE/RMSE for a horizon forecast.

    Parameters
    ----------
    y_true : np.ndarray
        True future values, shape (H,).
    y_pred : np.ndarray
        Predicted future values, shape (H,).

    Returns
    -------
    dict
        {"mae": float, "rmse": float}
    """
    return {"mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred)}


def plot_rollouts(y_true: np.ndarray, preds: Dict[str, np.ndarray]) -> None:
    """Plot ground truth and multiple forecast rollouts.

    Parameters
    ----------
    y_true : np.ndarray
        True future values, shape (H,).
    preds : Dict[str, np.ndarray]
        Mapping strategy_name -> predicted future values, each shape (H,).
    """
    plt.figure(figsize=(12, 4))
    plt.plot(y_true, label="true", linewidth=2)
    for name, y_hat in preds.items():
        plt.plot(y_hat, label=name, alpha=0.9)
    plt.grid(True)
    plt.legend()
    plt.title("Multi-step rollout comparison")
    plt.show()


# ----------------------------
# Demo (provided, not used in tests)
# ----------------------------

def _make_series(n: int = 2500, seed: int = 0) -> np.ndarray:
    """Generate a synthetic series (trend + seasonality + noise)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float32)
    x = 0.0009 * t + 2.0 * np.sin(2 * np.pi * t / 50.0) + 0.8 * np.sin(2 * np.pi * t / 16.0)
    x += rng.normal(0, 0.2, size=n).astype(np.float32)
    return x.astype(np.float32)


def demo() -> None:
    """End-to-end demo.

    Trains a one-step model and a K-step model (K=20) on synthetic data,
    then compares three rollout strategies on horizon H=100.

    This function is for student orientation (plots), not for unit tests.
    """
    tf.keras.utils.set_random_seed(123)

    series = _make_series(n=2600, seed=123)
    window = 40
    k = 20
    H = 100

    # Train models (students implement train_model)
    one_model, X_test_1, y_test_1 = train_model(series, window=window, horizon=1, epochs=15, seed=123, verbose=0)
    k_model, X_test_k, y_test_k = train_model(series, window=window, horizon=k, epochs=15, seed=123, verbose=0)

    # Create an initial window and ground-truth future from the end of the series
    init_window = series[-(window + H) : -H]
    y_true = series[-H:]

    pred_1 = recursive_rollout_one_step(one_model, init_window, horizon=H)
    pred_k20 = recursive_rollout_k_step_stride_k(k_model, init_window, k=k, horizon=H)
    pred_k1 = recursive_rollout_k_step_stride_1(k_model, init_window, k=k, horizon=H)

    preds = {
        "one-step (stride=1)": pred_1,
        "K-step=20 (stride=20)": pred_k20,
        "K-step=20 (stride=1, use first)": pred_k1,
    }

    for name, y_hat in preds.items():
        print(name, horizon_errors(y_true, y_hat))

    plot_rollouts(y_true, preds)


if __name__ == "__main__":
    demo()
