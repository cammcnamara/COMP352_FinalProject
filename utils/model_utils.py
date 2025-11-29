import numpy as np
from typing import List, Tuple
from sklearn.base import clone
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from collections import OrderedDict
import xgboost as xgb
import lightgbm as lgb
from skopt import BayesSearchCV
from skopt.space import Real, Integer

import pandas as pd

def hyperparameter_tune_bayesian(
    X_train: np.ndarray,
    y_train: np.ndarray,
    regressor_type: str = 'xgboost'
) -> OrderedDict:
    """
    Perform hyperparameter tuning for XGBoost or LightGBM using Bayesian search.
    
    Args:
        X_train: Training feature matrix
        y_train: Training target vector
        regressor_type: Type of regressor ('xgboost' or 'lightgbm')
        
    Returns:
        Best hyperparameters found during tuning.
    """
    # Define the parameter search space based on the regressor type
    if regressor_type.lower() == 'xgboost':
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=42
        )
        param_space = {
            'n_estimators': Integer(100, 1500),
            'max_depth': Integer(3, 7),
            'learning_rate': Real(0.01, 0.1, prior='log-uniform')
        }
    elif regressor_type.lower() == 'lightgbm':
        model = lgb.LGBMRegressor(
            objective='regression',
            random_state=42
        )
        param_space = {
            'n_estimators': Integer(100, 1500),
            'max_depth': Integer(3, 7),
            'learning_rate': Real(0.01, 0.1, prior='log-uniform')
        }
    else:
        raise ValueError(f"Unsupported regressor type: {regressor_type}")
    
    # Create the BayesSearchCV object
    opt = BayesSearchCV(
        model,
        param_space,
        n_iter=50,
        cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        random_state=42
    )
    
    # Fit the optimizer
    opt.fit(X_train, y_train)
    
    # Get the best hyperparameters
    return opt.best_params_

def time_series_split_regression(
    data: pd.DataFrame,
    regressor: object,
    date_column: str = "date_sold",
    target_column: str = "SalePrice",
    cols_to_ignore: List[str] = ["Id", "SalePrice_normalized"],
    n_splits: int = 5,
    tune_hyperparameters: bool = False,
) -> Tuple[
    pd.DataFrame,  # result_df
    List[Tuple[float, int]],  # rmse_scores
    List[Tuple[str, str]],  # split_dates
    List[Tuple[int, int]],  # num_records
]:
    """
    Perform time series split on a pandas DataFrame based on a date column and
    train a regression model, calculating RMSE for each split.

    Parameters:
    - data: pandas DataFrame
    - regressor: scikit-learn regressor object
        The regression algorithm to use.
    - date_column: str, default="date_sold"
        The name of the date column in the DataFrame.
    - target_column: str, default="SalePrice"
        The name of the target column in the DataFrame.
    - n_splits: int, default=5
        Number of splits for TimeSeriesSplit.
    - tune_hyperparameters: bool, default=False

    Returns:
    - result_df: pandas DataFrame
        DataFrame containing the Id, actual value, predicted value, fold, and whether it was in the test or train set.
    - rmse_scores: list of floats
        List of RMSE scores for each split.
    - split_dates: list of tuples
        List of (min_date, max_date) tuples for each split.
    - num_records: list of tuples
        List of (train_size, test_size) tuples for each split.
    """

    # Sort the DataFrame based on the date column
    data = data.sort_values(by=date_column)

    # Initialize TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_splits)

    rmse_scores = []
    split_dates = []
    num_records = []
    all_predictions = []

    # Perform the time series split and train regression model for each split
    for fold, (train_index, test_index) in enumerate(tscv.split(data)):
        train_data, test_data = data.iloc[train_index], data.iloc[test_index]

        cols_to_ignore = cols_to_ignore + [target_column, date_column]

        # Bayesian hyperparameter tuning for XGBoost
        if (
            isinstance(regressor, (xgb.XGBRegressor, lgb.LGBMRegressor))
            and tune_hyperparameters
        ):  # Add LGBMRegressor to the isinstance check
            # Determine the regressor_type based on the type of the regressor
            if isinstance(regressor, xgb.XGBRegressor):
                regressor_type = "XGBoost"
            elif isinstance(regressor, lgb.LGBMRegressor):
                regressor_type = "LightGBM"
            else:
                raise ValueError(
                    "Unsupported regressor type. Supported types: XGBRegressor, LGBMRegressor"
                )

            X_train_hyper, y_train_hyper = (
                data.drop(cols_to_ignore, axis=1),
                data[target_column],
            )
            best_params = hyperparameter_tune_bayesian(
                X_train_hyper, y_train_hyper, regressor_type
            )  # Specify 'xgboost' or 'lightgm' as the regressor type
            print(
                f"Best hyperparameters for {regressor_type} Fold {fold}: {best_params}"
            )
            regressor.set_params(**best_params)  # Set the best hyperparameters

        X_train = train_data.drop(cols_to_ignore, axis=1)
        X_test = test_data.drop(cols_to_ignore, axis=1)
        y_train, y_test = train_data[target_column], test_data[target_column]

        # Record the minimum and maximum dates for each split
        min_date, max_date = test_data[date_column].min(), test_data[date_column].max()
        split_dates.append((min_date, max_date))

        # Train regression model
        regressor.fit(
            X_train, np.log1p(y_train)
        )  # Apply log1p transformation to the target variable during training

        # Make predictions
        y_pred_log = regressor.predict(X_test)
        y_pred_train_log = regressor.predict(X_train)

        # Inverse transform predictions to get back the original scale
        y_pred = np.expm1(y_pred_log)
        y_pred_train = np.expm1(y_pred_train_log)

        # Check for NaN or infinity values in y_pred or y_test
        if (
            np.isnan(y_pred).any()
            or np.isinf(y_pred).any()
            or np.isnan(y_test).any()
            or np.isinf(y_test).any()
        ):
            print(
                f"Warning: NaN or infinity values found in predictions or true values. Imputing 0 for problematic values in y_pred for fold {fold}."
            )
            y_pred[np.isnan(y_pred) | np.isinf(y_pred)] = 0
            # Optionally, you can also handle y_test in a similar way if needed
            # y_test[np.isnan(y_test) | np.isinf(y_test)] = 0

        # Calculate RMSE on the original scale
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rmse_scores.append((rmse, fold))

        # Record results for 'Id', 'Actual', 'Predicted', 'Fold', and 'Set' in a list
        fold_predictions = list(
            zip(
                test_data["Id"],
                y_test,
                y_pred,
                [fold] * len(test_data),
                ["test"] * len(test_data),
            )
        )
        fold_predictions += list(
            zip(
                train_data["Id"],
                y_train,
                y_pred_train,
                [fold] * len(train_data),
                ["train"] * len(train_data),
            )
        )
        all_predictions.extend(fold_predictions)

        # Calculate the size of each train-test split
        num_records.append((len(train_data), len(test_data)))

    # Create a DataFrame from the results
    result_df = pd.DataFrame(
        all_predictions, columns=["Id", "Actual", "Predicted", "Fold", "Set"]
    )

    return result_df, rmse_scores, split_dates, num_records

class StackedEnsembleCVRegressor:
    def __init__(
        self, base_models: List[object], meta_model: object, n_folds: int = 5
    ) -> None:
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    # We again fit the data on clones of the original models
    def fit(self, X: np.ndarray, y: np.ndarray):
        X, y = np.array(X), np.array(y)

        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    # Do the predictions of all base models on the test data and use the averaged predictions as
    # meta-features for the final prediction which is done by the meta-model
    def predict(self, X: np.ndarray) -> np.ndarray:
        meta_features = np.column_stack(
            [
                np.column_stack([model.predict(X) for model in base_models]).mean(
                    axis=1
                )
                for base_models in self.base_models_
            ]
        )
        return self.meta_model_.predict(meta_features)