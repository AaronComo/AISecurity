from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from fairlearn.adversarial import AdversarialFairnessClassifier
from fairlearn.metrics import MetricFrame
from fairlearn.metrics import (
    selection_rate,
    demographic_parity_difference,
    count)
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from numpy import mean
from numpy import number
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .models import PredictorModel

schedulers = []
step = 1
X_prep_test, Y_test, pos_label, Z_test = 0, 0, 0, 0


def raw(X, y_true, sex, path):
    classifier = DecisionTreeClassifier(min_samples_leaf=10, max_depth=4)
    classifier.fit(X, y_true)

    y_pred = classifier.predict(X)
    gm = MetricFrame(metrics=accuracy_score, y_true=y_true, y_pred=y_pred, sensitive_features=sex)

    sr = MetricFrame(metrics=selection_rate, y_true=y_true, y_pred=y_pred, sensitive_features=sex)

    metrics = {
        "Accuracy": accuracy_score,
        "Selection Rate": selection_rate,
        "Salary": count,
    }
    metric_frame = MetricFrame(
        metrics=metrics, y_true=y_true, y_pred=y_pred, sensitive_features=sex
    )
    plot(metric_frame, 'Raw AI Model', savepth=path)


def Demographicparity(X, y_true, sex, path):
    np.random.seed(0)  # set seed for consistent results with ExponentiatedGradient
    constraint = DemographicParity()
    classifier = DecisionTreeClassifier(min_samples_leaf=10, max_depth=4)
    mitigator = ExponentiatedGradient(classifier, constraint)
    mitigator.fit(X, y_true, sensitive_features=sex)

    y_pred_mitigated = mitigator.predict(X)
    sr_mitigated = MetricFrame(metrics={"accuracy": accuracy_score, "selection_rate": selection_rate}, y_true=y_true,
                               y_pred=y_pred_mitigated, sensitive_features=sex)

    metrics = {
        "Accuracy": accuracy_score,
        "Selection Rate": selection_rate,
        "Salary": count,
    }
    metric_frame = MetricFrame(
        metrics=metrics, y_true=y_true, y_pred=y_pred_mitigated, sensitive_features=sex
    )
    plot(metric_frame, 'Model debias based on data preprocessing', savepth=path)


def Adversarialfairness(path):
    global X_prep_test, Y_test, pos_label, Z_test
    X, y = fetch_openml(data_id=1590, as_frame=True, return_X_y=True)
    pos_label = y[0]

    z = X["sex"]

    ct = make_column_transformer(
        (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("normalizer", StandardScaler()),
                ]
            ),
            make_column_selector(dtype_include=number),
        ),
        (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(drop="if_binary", sparse=False)),
                ]
            ),
            make_column_selector(dtype_include="category"),
        ),
    )

    X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(
        X, y, z, test_size=0.2, random_state=12345, stratify=y
    )

    X_prep_train = ct.fit_transform(X_train)  # Only fit on training data!
    X_prep_test = ct.transform(X_test)

    predictor_model = PredictorModel()

    mitigator = AdversarialFairnessClassifier(
        predictor_model=predictor_model,
        adversary_model=[3, "leaky_relu"],
        predictor_optimizer=optimizer_constructor,
        adversary_optimizer=optimizer_constructor,
        epochs=10,
        batch_size=2 ** 7,
        shuffle=True,
        callbacks=callbacks,
        random_state=123,
    )

    mitigator.fit(X_prep_train, Y_train, sensitive_features=Z_train)

    predictions = mitigator.predict(X_prep_test)
    mf = MetricFrame(
        metrics={"accuracy": accuracy_score, "selection_rate": selection_rate},
        y_true=Y_test == pos_label,
        y_pred=predictions == pos_label,
        sensitive_features=Z_test,
    )

    metrics = {
        "Accuracy": accuracy_score,
        "Selection Rate": selection_rate,
        "Salary": count,
    }
    metric_frame = MetricFrame(
        metrics=metrics, y_true=Y_test == pos_label, y_pred=predictions == pos_label, sensitive_features=Z_test
    )
    plot(metric_frame, 'Model debias based on forgetting learning', savepth=path)


def validate(mitigator):
    global X_prep_test, Y_test, pos_label, Z_test
    predictions = mitigator.predict(X_prep_test)
    dp_diff = demographic_parity_difference(
        Y_test == pos_label,
        predictions == pos_label,
        sensitive_features=Z_test,
    )
    accuracy = mean(predictions.values == Y_test.values)
    selection_rate = mean(predictions == pos_label)
    print(
        "DP diff: {:.4f}, accuracy: {:.4f}, selection_rate: {:.4f}".format(
            dp_diff, accuracy, selection_rate
        )
    )
    return dp_diff, accuracy, selection_rate


def optimizer_constructor(model):
    global schedulers
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    schedulers.append(
        torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
    )
    return optimizer


def callbacks(model, *args):
    global step
    global schedulers
    step += 1
    # Update hyperparameters
    model.alpha = 0.3 * sqrt(step // 1)
    for scheduler in schedulers:
        scheduler.step()
    # Validate (and early stopping) every 50 steps
    if step % 50 == 0:
        dp_diff, accuracy, selection_rate = validate(model)
        # Early stopping condition:
        # Good accuracy + low dp_diff + no mode collapse
        if (
                dp_diff < 0.03
                and accuracy > 0.8
                and 0.01 < selection_rate < 0.99
        ):
            return True


def plot(metric_frame, title, savepth):
    metric_frame.by_group.plot.bar(
        subplots=True,
        layout=[1, 3],
        legend=False,
        figsize=[12, 4],
        title=title,
    )
    plt.tight_layout()
    plt.savefig(savepth)


def fairness(raw_path, demo_path, adv_path):
    # Load dataset from openml
    data = fetch_openml(data_id=1590, as_frame=True)
    X = pd.get_dummies(data.data)

    y_true = (data.target == '>50K') * 1
    sex = data.data['sex']

    raw(X, y_true, sex, raw_path)

    Demographicparity(X, y_true, sex, demo_path)

    Adversarialfairness(adv_path)
    return True
