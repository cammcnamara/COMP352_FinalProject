from typing import List, Tuple
import numpy as np

def compute_rmse_std(tuple_list: List[Tuple[float, int]]) -> Tuple[float, float]:
    """
    Compute the mean and standard deviation of RMSE values.
    
    Args:
        tuple_list (List[Tuple[float, int]]): List of tuples where the first element
            is the RMSE value and the second element is typically an index.
    
    Returns:
        Tuple[float, float]: A tuple containing (mean_rmse, standard_deviation_rmse)
    """
    first_elements = [t[0] for t in tuple_list]
    mean = np.mean(first_elements)
    std = np.std(first_elements)
    return mean, std


def print_rmse_and_dates(
    model_rmse: List[Tuple[float, int]],
    model_split_dates: List[Tuple[str, str]],
    num_records: List[Tuple[int, int]],
    model_name: str,
) -> None:
    """
    Print RMSE scores along with date ranges and record counts for model evaluation.
    
    Args:
        model_rmse (List[Tuple[float, int]]): List of tuples containing RMSE values and split indices.
        model_split_dates (List[Tuple[str, str]]): List of tuples with (min_date, max_date) for each split.
        num_records (List[Tuple[int, int]]): List of tuples containing (train_record_count, test_record_count) for each split.
        model_name (str): Name of the model being evaluated.
    
    Returns:
        None: This function prints information to the console but does not return a value.
    """
    # Print RMSE scores and split dates for each split
    for i, (rmse, dates, records) in enumerate(
        zip(model_rmse, model_split_dates, num_records)
    ):
        min_date, max_date = dates
        num_train_records, num_test_records = records

        min_date = min_date.date()
        max_date = max_date.date()

        print(
            f"Split {i + 1}: Min Date: {min_date}, Max Date: {max_date}, RMSE: {rmse[0]}, Train Records: {num_train_records}, Test Records: {num_test_records}"
        )

    rmse_std = compute_rmse_std(model_rmse)
    print(model_name, "RMSE score: {:.4f} ({:.4f})\n".format(rmse_std[0], rmse_std[1]))