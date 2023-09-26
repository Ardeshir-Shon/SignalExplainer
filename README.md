# SignalExplainer

SignalExplainer is a specialized tool designed for deep learning models that work with signal-based data. It provides visual insights to explain the decision-making process of the model when classifying signal data.

## How to Use

Usage: python signalExplainer.py <path to TF model> <path to NPZ data file>
Example: python signalExplainer.py model/multi_convlstm.h5 data/dataset.npz


### NPZ File Structure:
- **X**: A numpy array with shape (n_samples, signal, n_channels).
- **y**: A numpy array with shape (n_samples, 1).
- **channel_names**: A numpy array with shape (numberOfChannels, 1).
- **class_names**: A numpy array with shape (numberOfClasses, 1).

### TF Model Structure:
- **model**: A Keras model.

> **Note**: Ensure that `X` is ordered consistently with the channels in the model.

## Citation

If you find this tool useful, please consider citing our work:

```bibtex
@inproceedings{jalayer2023model,
  title={A Model Identification Forensics Approach for Signal-Based Condition Monitoring},
  author={Jalayer, Masoud and Shojaeinasab, Ardeshir and Najjaran, Homayoun},
  booktitle={International Conference on Flexible Automation and Intelligent Manufacturing},
  pages={12--19},
  year={2023},
  organization={Springer}
}
