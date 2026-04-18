import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class GaitFreqData:
    """Stores COT (or any metric) values for a single gait across multiple frequencies."""
    gait_name: str
    frequency_labels: List[str]
    values: List[float]

    def __post_init__(self):
        if len(self.frequency_labels) != len(self.values):
            raise ValueError(
                f"frequency_labels length ({len(self.frequency_labels)}) "
                f"must match values length ({len(self.values)})"
            )


def build_heatmap_from_gaits(gait_data_list: List[GaitFreqData]) -> "HeatMap":
    """Construct a HeatMap from a list of GaitFreqData entries.
    Rows = gaits, Columns = frequencies.
    """
    # Validate all entries share the same frequency labels
    freq_labels = gait_data_list[0].frequency_labels
    for g in gait_data_list[1:]:
        if g.frequency_labels != freq_labels:
            raise ValueError("All GaitFreqData entries must share the same frequency_labels.")

    matrix = np.array([g.values for g in gait_data_list])
    gait_names = [g.gait_name for g in gait_data_list]
    hm = HeatMap(matrix)
    hm._default_y_labels = gait_names
    hm._default_x_labels = freq_labels
    return hm


class HeatMap:
    def __init__(self, data):
        self.data = data
        # Optional default labels set by build_heatmap_from_gaits()
        self._default_x_labels: List[str] | None = None
        self._default_y_labels: List[str] | None = None

    def reset(self):
        self.data = np.zeros_like(self.data)

    def update(self, new_data):
        self.data = new_data

    def plot(self, x_labels=None, y_labels=None):
        # Fall back to labels stored by build_heatmap_from_gaits() if none supplied
        x_labels = x_labels if x_labels is not None else self._default_x_labels
        y_labels = y_labels if y_labels is not None else self._default_y_labels

        plt.imshow(self.data, cmap='Blues', interpolation='nearest')
        plt.colorbar()
        plt.title('Heat Map')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Gait')
        if x_labels is not None:
            plt.xticks(ticks=range(len(x_labels)), labels=x_labels)
        if y_labels is not None:
            plt.yticks(ticks=range(len(y_labels)), labels=y_labels)
        plt.show()


def test_main():
    freq_labels = ['1.0 Hz', '1.5 Hz', '2.0 Hz']

    gait_data = [
        GaitFreqData('Walk',   freq_labels, [0.30, 0.20, 0.10]),
        GaitFreqData('Trot',   freq_labels, [0.15, 0.30, 0.15]),
        GaitFreqData('Gallop', freq_labels, [0.05, 0.10, 0.40]),
    ]

    heat_map = build_heatmap_from_gaits(gait_data)
    heat_map.plot()

if __name__ == "__main__":
    test_main()