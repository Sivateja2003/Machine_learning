"""
ML Dashboard Backend
=====================
Flask API that trains 5 supervised-learning models on startup (in-memory,
no pickled artifacts needed), all from the single stock_prices.csv dataset,
and exposes sample-data / predict / info endpoints for each of them.

Run:
    pip install -r requirements.txt
    python generate_datasets.py   # only needed once, already generated
    python app.py
Server starts on http://localhost:5000
"""
import os
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, mean_absolute_error

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "datasets")

app = Flask(__name__)
CORS(app)

REGISTRY = {}  # populated by each train_* function

FEATURES = ["prev_close", "open", "high", "low", "volume"]
MOVEMENT_CLASSES = {0: "Down", 1: "Up"}


def _load_stock_data():
    return pd.read_csv(os.path.join(DATA_DIR, "stock_prices.csv"))


def _with_movement(df):
    """Derive an Up/Down movement label from next_close vs prev_close."""
    view = df.copy()
    movement = (view["next_close"] > view["prev_close"]).astype(int)
    view["movement"] = movement.map(MOVEMENT_CLASSES)
    return view, movement


def _with_price_change(df):
    """Derive the day-over-day price change used as the regression target."""
    view = df.copy()
    view["price_change"] = (view["next_close"] - view["prev_close"]).round(2)
    return view


# =============================================================================
# 1. LINEAR REGRESSION — Stock Price Change Prediction
# =============================================================================
def train_linear_regression():
    df = _with_price_change(_load_stock_data())
    features = FEATURES
    target = "price_change"

    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df[target], test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))
    mae = mean_absolute_error(y_test, model.predict(X_test))

    REGISTRY["linear-regression"] = {
        "model": model,
        "features": features,
        "target": target,
        "df": df,
        "metrics": {"r2_score": round(r2, 4), "mae": round(mae, 2)},
        "task": "regression",
        "name": "Linear Regression",
        "use_case": "Stock Price Change Prediction",
        "explanation": (
            "Linear Regression learns a straight-line relationship between the input "
            "features (previous close, open, high, low prices and trading volume) and "
            "the target: the size of tomorrow's price move in dollars (next close minus "
            "previous close). It finds the weights that minimize the sum of squared "
            "errors between predicted and actual price changes, producing an equation of "
            "the form y = w1*x1 + w2*x2 + ... + b. Since the underlying price move here "
            "is close to a linear combination of these inputs, Linear Regression "
            "captures it very well."
        ),
    }


# =============================================================================
# 2. LOGISTIC REGRESSION — Stock Price Movement Prediction
# =============================================================================
def train_logistic_regression():
    df, movement = _with_movement(_load_stock_data())
    features = FEATURES

    X_train, X_test, y_train, y_test = train_test_split(
        df[features], movement, test_size=0.2, random_state=42, stratify=movement
    )
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))

    REGISTRY["logistic-regression"] = {
        "model": model,
        "features": features,
        "target": "movement",
        "df": df,
        "metrics": {"accuracy": round(acc, 4)},
        "task": "classification",
        "classes": MOVEMENT_CLASSES,
        "name": "Logistic Regression",
        "use_case": "Stock Price Movement Prediction",
        "explanation": (
            "Logistic Regression estimates the probability that tomorrow's close will "
            "be higher than today's by applying the sigmoid function to a weighted sum "
            "of the previous close, open, high, low prices and trading volume. Unlike "
            "Linear Regression, its output is squashed between 0 and 1 and interpreted "
            "as a probability: above 0.5 the model predicts 'Up', otherwise 'Down'."
        ),
    }


# =============================================================================
# 3. DECISION TREE — Stock Price Movement Prediction
# =============================================================================
def train_decision_tree():
    df, movement = _with_movement(_load_stock_data())
    features = FEATURES

    X_train, X_test, y_train, y_test = train_test_split(
        df[features], movement, test_size=0.2, random_state=42, stratify=movement
    )
    model = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))

    REGISTRY["decision-tree"] = {
        "model": model,
        "features": features,
        "target": "movement",
        "df": df,
        "metrics": {"accuracy": round(acc, 4)},
        "task": "classification",
        "classes": MOVEMENT_CLASSES,
        "name": "Decision Tree",
        "use_case": "Stock Price Movement Prediction",
        "explanation": (
            "A Decision Tree splits the data step by step using the feature that best "
            "separates the classes at each node (measured here with information gain / "
            "entropy). Using the previous close, open, high, low prices and trading "
            "volume, the tree asks a sequence of threshold questions (e.g. 'is volume "
            "above X?') until it reaches a leaf that predicts whether the next close "
            "will move Up or Down. It is easy to interpret because the decision path "
            "can be read as a set of simple rules."
        ),
    }


# =============================================================================
# 4. RANDOM FOREST — Stock Price Movement Prediction
# =============================================================================
def train_random_forest():
    df, movement = _with_movement(_load_stock_data())
    features = FEATURES

    X_train, X_test, y_train, y_test = train_test_split(
        df[features], movement, test_size=0.2, random_state=42, stratify=movement
    )
    model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))

    REGISTRY["random-forest"] = {
        "model": model,
        "features": features,
        "target": "movement",
        "df": df,
        "metrics": {"accuracy": round(acc, 4)},
        "task": "classification",
        "classes": MOVEMENT_CLASSES,
        "name": "Random Forest",
        "use_case": "Stock Price Movement Prediction",
        "explanation": (
            "Random Forest builds many independent Decision Trees, each trained on a "
            "random subset of the data (bagging) and a random subset of features at "
            "every split. To predict whether the next close will move Up or Down from "
            "today's OHLC prices and volume, every tree in the forest 'votes' for a "
            "direction, and the forest returns the majority vote as the prediction and "
            "the vote share as a confidence score. Averaging many trees reduces "
            "overfitting compared to a single tree."
        ),
    }


# =============================================================================
# 5. GRADIENT BOOSTING — Stock Price Change Prediction
# =============================================================================
def train_gradient_boosting():
    df = _with_price_change(_load_stock_data())
    features = FEATURES
    target = "price_change"

    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df[target], test_size=0.2, random_state=42
    )
    model = GradientBoostingRegressor(
        n_estimators=150, learning_rate=0.1, max_depth=3,
        subsample=0.8, min_samples_leaf=3, random_state=42,
    )
    model.fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))
    mae = mean_absolute_error(y_test, model.predict(X_test))

    REGISTRY["gradient-boosting"] = {
        "model": model,
        "features": features,
        "target": target,
        "df": df,
        "metrics": {"r2_score": round(r2, 4), "mae": round(mae, 2)},
        "task": "regression",
        "name": "Gradient Boosting",
        "use_case": "Stock Price Change Prediction",
        "explanation": (
            "Gradient Boosting builds an ensemble of shallow Decision Trees "
            "sequentially: each new tree is trained to correct the errors (residuals) "
            "made by the trees before it, scaled by a learning rate. It predicts the "
            "same target as Linear Regression — the size of tomorrow's price move — "
            "from the previous close, open, high, low prices and trading volume. "
            "Because the true relationship here is close to linear and the training set "
            "is modest in size, boosting's extra flexibility doesn't pay off the way it "
            "would on a larger, more non-linear dataset: it lands behind Linear "
            "Regression's R², a useful reminder that a simpler model can beat a more "
            "powerful one when the data doesn't call for the extra complexity."
        ),
    }


def train_all_models():
    train_linear_regression()
    train_logistic_regression()
    train_decision_tree()
    train_random_forest()
    train_gradient_boosting()
    print(f"Trained {len(REGISTRY)} models: {list(REGISTRY.keys())}")


# =============================================================================
# Helpers
# =============================================================================
def confidence_from_proba(proba_row, classes):
    idx = int(np.argmax(proba_row))
    return classes[idx], float(round(proba_row[idx] * 100, 2)), {
        classes[i]: float(round(p * 100, 2)) for i, p in enumerate(proba_row)
    }


def clean_records(df, n=15):
    """Return first n rows as JSON-safe list of dicts."""
    return df.head(n).to_dict(orient="records")


# =============================================================================
# API ROUTES
# =============================================================================
@app.route("/api/models", methods=["GET"])
def list_models():
    out = []
    for key, entry in REGISTRY.items():
        out.append(
            {
                "id": key,
                "name": entry["name"],
                "use_case": entry["use_case"],
                "task": entry["task"],
                "features": entry["features"],
                "metrics": entry["metrics"],
            }
        )
    return jsonify(out)


@app.route("/api/<model_id>/sample", methods=["GET"])
def get_sample(model_id):
    entry = REGISTRY.get(model_id)
    if not entry:
        return jsonify({"error": "Model not found"}), 404
    return jsonify(
        {
            "columns": list(entry["df"].columns),
            "rows": clean_records(entry["df"]),
            "total_rows": len(entry["df"]),
        }
    )


@app.route("/api/<model_id>/info", methods=["GET"])
def get_info(model_id):
    entry = REGISTRY.get(model_id)
    if not entry:
        return jsonify({"error": "Model not found"}), 404
    return jsonify(
        {
            "id": model_id,
            "name": entry["name"],
            "use_case": entry["use_case"],
            "task": entry["task"],
            "features": entry["features"],
            "target": entry.get("target"),
            "metrics": entry["metrics"],
            "explanation": entry["explanation"],
            "classes": entry.get("classes"),
        }
    )


@app.route("/api/<model_id>/predict", methods=["POST"])
def predict(model_id):
    entry = REGISTRY.get(model_id)
    if not entry:
        return jsonify({"error": "Model not found"}), 404

    payload = request.get_json(force=True, silent=True) or {}
    features = entry["features"]

    missing = [f for f in features if f not in payload]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    try:
        X = pd.DataFrame([[float(payload[f]) for f in features]], columns=features)
        model = entry["model"]

        if entry["task"] == "regression":
            pred = float(model.predict(X)[0])
            response = {
                "prediction": round(pred, 2),
                "task": "regression",
                "metrics": entry["metrics"],
            }
        else:
            classes = entry["classes"]
            proba = model.predict_proba(X)[0]
            label, confidence, full_proba = confidence_from_proba(proba, classes)
            response = {
                "prediction": label,
                "confidence": confidence,
                "probabilities": full_proba,
                "task": "classification",
                "metrics": entry["metrics"],
            }

        return jsonify(response)

    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "models_loaded": list(REGISTRY.keys())})


train_all_models()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
