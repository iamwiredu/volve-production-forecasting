"""Forecast error metrics used to compare Arps and ML models."""
import numpy as np


def rmse(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mape(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    return float(np.mean(np.abs((actual - predicted) / actual)) * 100)


def cumulative_error_pct(actual, predicted):
    """Error in total volume over the test window, in percent."""
    return float((np.sum(predicted) - np.sum(actual)) / np.sum(actual) * 100)
