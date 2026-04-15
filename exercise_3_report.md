# Exercise 3 Report: Motor Current Anomaly Detection
## Overview
This exercise compares an LSTM, a 1D CNN, and a transformer encoder on a synthetic motor current anomaly detection task. The dataset contains 1200 waveforms (128 time steps each) in three classes: healthy, bearing wear, and winding fault.

## Dataset
The synthetic dataset is generated with small load variation, high-frequency ripple for bearing wear, and slight positive-peak asymmetry for winding fault. A saved dataset file `motor_current_data.npz` is produced by the script.

## Visualizations
![Waveform Visualization](waveform_visualization.png)

## Test Results
| Model | Test Accuracy |
|---|---:|
| LSTM | 0.311 |
| 1D CNN | 0.811 |
| Transformer | 0.822 |

## Per-Class Performance
The classification reports below show model performance on healthy, bearing wear, and winding fault examples.

### LSTM

```               precision    recall  f1-score   support

      healthy       0.29      0.76      0.42        51
 bearing_wear       1.00      0.01      0.03        68
winding_fault       0.36      0.26      0.30        61

     accuracy                           0.31       180
    macro avg       0.55      0.35      0.25       180
 weighted avg       0.58      0.31      0.23       180

```
### 1D CNN

```               precision    recall  f1-score   support

      healthy       0.60      1.00      0.75        51
 bearing_wear       1.00      0.51      0.68        68
winding_fault       1.00      0.98      0.99        61

     accuracy                           0.81       180
    macro avg       0.87      0.83      0.81       180
 weighted avg       0.89      0.81      0.81       180

```
### Transformer

```               precision    recall  f1-score   support

      healthy       0.66      0.76      0.71        51
 bearing_wear       0.80      0.71      0.75        68
winding_fault       1.00      1.00      1.00        61

     accuracy                           0.82       180
    macro avg       0.82      0.82      0.82       180
 weighted avg       0.83      0.82      0.82       180

```
## Attention Visualization
The transformer attention heatmaps are saved as `attention_bearing_wear.png` and `attention_winding_fault.png`.

![Attention Bearing Wear](attention_bearing_wear.png)
![Attention Winding Fault](attention_winding_fault.png)

## Discussion
The transformer performs best because self-attention compares all time steps in parallel and can detect both the high-frequency bearing ripple and the global winding asymmetry. The 1D CNN captures local time-domain patterns, while the LSTM struggles with subtle spectral detail because it must accumulate information sequentially.
