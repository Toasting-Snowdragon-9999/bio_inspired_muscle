"""@brief heat_map.py — heat map."""
import re
import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import List


base_size = 20
plt.rcParams.update({
    'font.size': base_size,        # Default text size
    'axes.titlesize': base_size + 5,   # Title size
    'axes.labelsize': base_size + 4,   # X and Y label size
    'xtick.labelsize': base_size + 2,  # X tick size
    'ytick.labelsize': base_size + 2,  # Y tick size
    'legend.fontsize': base_size + 2   # Legend size
})

@dataclass
class GaitFreqData:
    """@brief GaitFreqData — gait freq data."""
    gait_name: str
    frequency_labels: List[str]
    values: List[float]

    def __post_init__(self):
        """@brief Post-initialisation hook for the dataclass."""
        if len(self.frequency_labels) != len(self.values):
            raise ValueError(
                f"frequency_labels length ({len(self.frequency_labels)}) "
                f"must match values length ({len(self.values)})"
            )


class HeatMap:
    """@brief HeatMap — heat map."""
    def __init__(self, data):
        """
        @brief Construct a HeatMap instance.
        @param data:
        """
        self.data = data
        self._default_x_labels = None
        self._default_y_labels = None

    def plot(self, x_labels=None, y_labels=None):

        """
        @brief Plot.
        @param x_labels:
        @param y_labels:
        """
        x_labels = x_labels if x_labels is not None else self._default_x_labels
        y_labels = y_labels if y_labels is not None else self._default_y_labels

        base_size = plt.rcParams['font.size']

        width = max(
            12,
            self.data.shape[1] * (base_size * 0.35) * 0.2
        )

        height = max(
            6,
            self.data.shape[0] * (base_size * 0.22) * 0.2
        )

        plt.figure(figsize=(width, height))

        plt.imshow(
            self.data,
            cmap='Blues',
            interpolation='nearest',
            vmin=0,
            vmax=2,
            aspect='auto'
        )

        plt.colorbar(label="Mean CoT")

        plt.title('Mean CoT vs Frequency')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Gait')

        if x_labels is not None:
            plt.xticks(
                ticks=range(len(x_labels)),
                labels=x_labels,
                rotation=45
            )

        if y_labels is not None:
            plt.yticks(
                ticks=range(len(y_labels)),
                labels=y_labels
            )

        # Draw mean values in cells
        for i in range(self.data.shape[0]):
            for j in range(self.data.shape[1]):

                value = self.data[i, j]

                plt.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha='center',
                    va='center',
                    color='black'
                )

        plt.tight_layout(pad=2.0)
        plt.show()


def build_heatmap_from_gaits(gait_data_list):

    # Collect ALL unique frequencies
    """
    @brief Build heatmap from gaits.
    @param gait_data_list:
    @return
    """
    all_freqs = sorted(set(
        freq
        for gait in gait_data_list
        for freq in gait.frequency_labels
    ))

    matrix = []

    for gait in gait_data_list:

        # Map frequency -> value
        value_map = dict(zip(gait.frequency_labels, gait.values))

        row = []

        for freq in all_freqs:

            # Missing frequencies become max penalty
            row.append(value_map.get(freq, 2.0))

        matrix.append(row)

    matrix = np.array(matrix)

    gait_names = [g.gait_name for g in gait_data_list]

    hm = HeatMap(matrix)

    hm._default_x_labels = all_freqs
    hm._default_y_labels = gait_names

    return hm


def parse_frequency_file(filepath, gait_name):

    """
    @brief Parse frequency file.
    @param filepath:
    @param gait_name:
    @return
    """
    frequency_data = {}

    current_freq = None

    with open(filepath, "r") as f:
        lines = f.readlines()

    for line in lines:

        # Match:
        # Incrementing frequency to 1.40 Hz
        freq_match = re.search(
            r'Incrementing frequency to ([\d.]+)\s*Hz',
            line
        )

        if freq_match:

            current_freq = float(freq_match.group(1))

            if current_freq not in frequency_data:
                frequency_data[current_freq] = []

            continue

        # Match:
        # 41  CoT:  0.48066626397280193
        cot_match = re.search(r'CoT:\s*([^\s]+)', line)

        if cot_match and current_freq is not None:

            cot_str = cot_match.group(1)

            # Treat None as maximum penalty
            if cot_str == "None":
                cot = 2.0
            else:
                cot = float(cot_str)

                # Clip ALL values into range [0, 2]
                cot = np.clip(cot, 0, 2)

            frequency_data[current_freq].append(cot)

    frequencies = sorted(frequency_data.keys())

    mean_values = []

    for freq in frequencies:

        values = frequency_data[freq]

        if len(values) == 0:
            mean_values.append(2.0)
        else:
            mean_values.append(np.mean(values))

    freq_labels = [f"{f:.1f} Hz" for f in frequencies]

    return GaitFreqData(
        gait_name=gait_name,
        frequency_labels=freq_labels,
        values=mean_values
    )


def main():

    """@brief Main."""
    walk_data = parse_frequency_file(
        "data/frequency_analysis_walk.txt",
        "Walk"
    )

    amble_data = parse_frequency_file(
        "data/frequency_analysis_amble.txt",
        "Amble"
    )

    trot_data = parse_frequency_file(
        "data/frequency_analysis_trot.txt",
        "Trot"
    )

    heatmap = build_heatmap_from_gaits([
        walk_data,
        amble_data,
        trot_data
    ])

    heatmap.plot()


if __name__ == "__main__":
    main()