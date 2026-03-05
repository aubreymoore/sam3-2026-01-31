import unittest
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from am_ptools import create_plot_grid

class TestPlotGridColumnLogic(unittest.TestCase):

    def test_return_type(self) -> None:
        """Verify the function returns a matplotlib Figure object."""
        data = [[1, 2, 3]]
        fig = create_plot_grid(data, nr_cols=1)
        self.assertIsInstance(fig, Figure)
        plt.close(fig)

    def test_row_calculation(self) -> None:
        """Test if 5 plots in 2 columns results in a 3x2 grid (6 total axes)."""
        data = [[0]] * 5
        cols = 2
        # 5 plots / 2 columns = 2.5 -> 3 rows needed
        fig = create_plot_grid(data, nr_cols=cols)
        
        # Check total axes count (rows * cols)
        self.assertEqual(len(fig.get_axes()), 6)
        plt.close(fig)

    def test_single_column_layout(self) -> None:
        """Verify that 3 plots in 1 column creates 3 rows."""
        data = [[0], [0], [0]]
        fig = create_plot_grid(data, nr_cols=1)
        # 1 column * 3 rows = 3 axes
        self.assertEqual(len(fig.get_axes()), 3)
        plt.close(fig)

    def test_invalid_col_error(self) -> None:
        """Ensure ValueError is raised for nr_cols < 1."""
        with self.assertRaises(ValueError):
            create_plot_grid([[1]], nr_cols=0)

if __name__ == "__main__":
    unittest.main()