import numpy as np

def load_sushi_data():
    """
    Loads the sushi rating data from the given file path.

    Parameters:
        file_path (str): Path to the sushi rating dataset file.

    Returns:
        np.ndarray: A NumPy array containing the sushi rating data.
    """
    file_path = 'datasets/sushi3-2016/sushi3a.5000.10.order'
    try:
        # Load the data, skipping the first row and using columns 1 to 10
        data = np.loadtxt(file_path, skiprows=1, usecols=range(1, 11))
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None