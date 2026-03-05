import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import math
from typing import List, Any, Union

def create_plot_grid(
    data_list: List[Union[List[Any], Any]], 
    nr_cols: int, 
    title: str = "Multi-plot Figure"
) -> Figure:
    """
    Creates a matplotlib figure with a list of plots arranged in a specified number of columns.

    Args:
        data_list (List[Union[List[Any], Any]]): A list where each element contains 
            the data points to be plotted on an individual subplot.
        nr_cols (int): The fixed number of columns for the figure layout.
        title (str, optional): The main title for the entire figure. Defaults to "Multi-plot Figure".

    Returns:
        matplotlib.figure.Figure: The generated figure object containing the subplots.

    Raises:
        ValueError: If data_list is empty or nr_cols is less than 1.
    """
    if not data_list:
        raise ValueError("The data_list cannot be empty.")
    if nr_cols < 1:
        raise ValueError("nr_cols must be an integer greater than or equal to 1.")

    num_plots: int = len(data_list)
    # Calculate rows needed to accommodate all plots given the column constraint
    nr_rows: int = math.ceil(num_plots / nr_cols)

    # Create the figure and axes
    # Size logic: Width scales with columns, Height scales with rows
    fig, axes = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols * 4, nr_rows * 3), squeeze=False)
    fig.suptitle(title, fontsize=16)

    # Flatten the 2D axes array for easy iteration
    axes_flat = axes.flatten()

    for i in range(len(axes_flat)):
        if i < num_plots:
            axes_flat[i].plot(data_list[i])
            axes_flat[i].set_title(f"Plot {i+1}")
        else:
            # Hide axes for empty grid slots (the "remainder" of the grid)
            axes_flat[i].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

if __name__ == "__main__":
    # Example: 7 plots in a 3-column layout
    import numpy as np
    data = [np.random.rand(10) for _ in range(7)]
    figure = create_plot_grid(data, nr_cols=3)
    plt.show()