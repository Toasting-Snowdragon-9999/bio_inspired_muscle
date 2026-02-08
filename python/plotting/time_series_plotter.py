import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Union


class TimeSeriesPlotter:
    """
    A flexible plotting class for time series data.
    
    Usage:
        plotter = TimeSeriesPlotter(time_data)
        plotter.plot({
            'Neuron 1': neuron1_data,
            'Neuron 2': neuron2_data,
            'Neuron 3': neuron3_data
        })
        plotter.show()
    """
    
    def __init__(
        self,
        time: np.ndarray,
        figsize: Tuple[float, float] = (12, 3),
        subplot_height: float = 3.0
    ):
        """
        Initialize the plotter.
        
        Args:
            time: Time array for x-axis
            figsize: Base figure size (width, height_per_subplot)
            subplot_height: Height of each subplot in inches
        """
        self.time = time
        self.base_width = figsize[0]
        self.subplot_height = subplot_height
        self.fig = None
        self.axes = None
        
    def plot(
        self,
        data_dict: Dict[str, np.ndarray],
        ylabel: str = 'Value',
        xlabel: str = 'Time (seconds)',
        title_prefix: str = '',
        colors: Optional[list] = None,
        linewidth: float = 0.8,
        grid: bool = True,
        grid_alpha: float = 0.3
    ):
        """
        Plot each data series on its own subplot.
        
        Args:
            data_dict: Dictionary of {name: data_array} pairs
            ylabel: Y-axis label for all subplots
            xlabel: X-axis label for bottom subplot
            title_prefix: Optional prefix for subplot titles
            colors: List of colors (default uses matplotlib color cycle)
            linewidth: Line width for plots
            grid: Whether to show grid
            grid_alpha: Grid transparency
        """
        n_plots = len(data_dict)
        
        if n_plots == 0:
            raise ValueError("data_dict must contain at least one entry")
        
        # Create figure with calculated height
        fig_height = self.subplot_height * n_plots
        self.fig, self.axes = plt.subplots(
            n_plots, 1,
            figsize=(self.base_width, fig_height),
            sharex=True
        )
        
        # Handle single subplot case (axes is not an array)
        if n_plots == 1:
            self.axes = [self.axes]
        
        # Plot each data series
        for idx, (name, data) in enumerate(data_dict.items()):
            color = colors[idx] if colors else f'C{idx}'
            
            self.axes[idx].plot(
                self.time,
                data,
                label=name,
                linewidth=linewidth,
                color=color
            )
            
            # Set title
            title = f"{title_prefix}{name}" if title_prefix else name
            self.axes[idx].set_title(title)
            
            # Set y-label
            self.axes[idx].set_ylabel(ylabel)
            
            # Add legend
            self.axes[idx].legend()
            
            # Grid
            if grid:
                self.axes[idx].grid(True, alpha=grid_alpha)
        
        # Set x-label only on bottom plot
        self.axes[-1].set_xlabel(xlabel)
        
        plt.tight_layout()
        
        return self.fig, self.axes
    
    def show(self):
        """Display the plot."""
        if self.fig is None:
            raise RuntimeError("Must call plot() before show()")
        plt.show()
    
    def save(
        self,
        filepath: Union[str, Path],
        dpi: int = 150,
        bbox_inches: str = 'tight'
    ):
        """
        Save the plot to file.
        
        Args:
            filepath: Output file path
            dpi: Resolution
            bbox_inches: Bounding box setting
        """
        if self.fig is None:
            raise RuntimeError("Must call plot() before save()")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        self.fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
        print(f"Saved plot to {filepath}")
    
    def close(self):
        """Close the figure."""
        if self.fig is not None:
            plt.close(self.fig)


class MultiSeriesPlotter:
    """
    Plot multiple data series on the same subplot grid.
    
    Usage:
        plotter = MultiSeriesPlotter(time_data, n_subplots=3)
        plotter.add_to_subplot(0, 'Neuron 1', neuron1_data)
        plotter.add_to_subplot(0, 'Neuron 2', neuron2_data)
        plotter.add_to_subplot(1, 'Internal State 1', internal1_data)
        plotter.finalize()
        plotter.show()
    """
    
    def __init__(
        self,
        time: np.ndarray,
        n_subplots: int,
        figsize: Tuple[float, float] = (12, 9),
        subplot_titles: Optional[list] = None
    ):
        """
        Initialize the multi-series plotter.
        
        Args:
            time: Time array for x-axis
            n_subplots: Number of subplots
            figsize: Figure size (width, height)
            subplot_titles: List of titles for each subplot
        """
        self.time = time
        self.n_subplots = n_subplots
        
        self.fig, self.axes = plt.subplots(
            n_subplots, 1,
            figsize=figsize,
            sharex=True
        )
        
        # Handle single subplot case
        if n_subplots == 1:
            self.axes = [self.axes]
        
        # Set titles if provided
        if subplot_titles:
            for idx, title in enumerate(subplot_titles):
                if idx < n_subplots:
                    self.axes[idx].set_title(title)
    
    def add_to_subplot(
        self,
        subplot_idx: int,
        label: str,
        data: np.ndarray,
        color: Optional[str] = None,
        linewidth: float = 0.8,
        **kwargs
    ):
        """
        Add a data series to a specific subplot.
        
        Args:
            subplot_idx: Index of subplot (0-based)
            label: Label for the data series
            data: Data array to plot
            color: Line color
            linewidth: Line width
            **kwargs: Additional arguments passed to plot()
        """
        if subplot_idx >= self.n_subplots:
            raise ValueError(f"subplot_idx {subplot_idx} out of range (max: {self.n_subplots-1})")
        
        self.axes[subplot_idx].plot(
            self.time,
            data,
            label=label,
            color=color,
            linewidth=linewidth,
            **kwargs
        )
    
    def finalize(
        self,
        ylabel: str = 'Value',
        xlabel: str = 'Time (seconds)',
        grid: bool = True,
        grid_alpha: float = 0.3,
        legend_ncol: int = 1
    ):
        """
        Finalize the plot with labels, legends, and grid.
        
        Args:
            ylabel: Y-axis label for all subplots
            xlabel: X-axis label for bottom subplot
            grid: Whether to show grid
            grid_alpha: Grid transparency
            legend_ncol: Number of columns in legend
        """
        for ax in self.axes:
            ax.set_ylabel(ylabel)
            ax.legend(ncol=legend_ncol)
            if grid:
                ax.grid(True, alpha=grid_alpha)
        
        self.axes[-1].set_xlabel(xlabel)
        plt.tight_layout()
    
    def show(self):
        """Display the plot."""
        plt.show()
    
    def save(
        self,
        filepath: Union[str, Path],
        dpi: int = 150,
        bbox_inches: str = 'tight'
    ):
        """Save the plot to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        self.fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
        print(f"Saved plot to {filepath}")
    
    def close(self):
        """Close the figure."""
        plt.close(self.fig)
