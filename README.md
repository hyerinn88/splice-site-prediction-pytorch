# Splice Site Prediction Using Neural Networks

A bioinformatics course project using PyTorch to predict splice sites from human genomic DNA sequences.

## Project Overview

The goal of this project was to develop and optimize a neural network for binary classification of genomic DNA sequences according to the presence or absence of a splice site.

The project used 196-bp human genomic DNA sequences divided into training and validation datasets. A starter PyTorch framework and the datasets were provided as part of the course assignment.

## Project Work

As part of the project, our group:

- Implemented classification metrics including accuracy, precision, recall, and F1-score
- Designed and optimized a convolutional neural network architecture in PyTorch
- Configured the model training routine and hyperparameters
- Tracked training and validation performance across epochs
- Evaluated model performance and investigated potential overfitting

## Technologies

- Python
- PyTorch
- NumPy
- Neural Networks
- Genomic Sequence Analysis
- Linux
- GPU Computing

## Model Evaluation

The model was evaluated using accuracy, precision, recall, and F1-score. Training and validation performance were monitored across epochs to assess model performance and potential overfitting.

The training results are provided in `training_34_.csv`.

## Repository Structure

- `cls_model.py` – neural network architecture
- `funct_metrics.py` – implementation of classification performance metrics
- `main_train.py` – model training and validation workflow
- `training_34_.csv` – training and validation performance results

## Note

This project was completed as a group assignment for a Bioinformatics course. A starter PyTorch framework and genomic datasets were provided as part of the course. The neural network architecture, performance evaluation, training configuration, and analysis were developed as part of the assignment.
