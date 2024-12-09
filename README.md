# MS-AAI 521: Automated Ice Hockey Player Tracking
Douglas Code - Group 8s

This repo contains the implementation for a computer vision model that performs detection, classification, and tracking on players in ice hockey video clips.
The classification module classifies players as either skaters or goalies, and as members of the home or away team.
Source code can be found in the `src` directory, and the LaTeX code for the final paper is in the `final-paper` directory.

Key source files:
* **config.py**: holds configuration settings for the entire project, including file paths and training hyperparameters.
* **main.py**: the point of entry script for training the model, performing inference, and 

## Running Locally

To run the full training, inference, evaluation, and visualization pipeline:

    python main.py

The script takes the following arguments:
* `--skip-preprocessing`: skips preprocessing steps and instead reads the already preprocessed data from disk
* `--skip-training`: skips training the model and proceeds directly to model evaluation and visualization
* `--tune-hyperparameters`: performs hyperparameter tuning, writing the optimized hyperparameters to disk after completion