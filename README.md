# MS-AAI 521: Automated Ice Hockey Player Tracking
Douglas Code - Group 8s

This repo contains the implementation for a computer vision model that performs detection, classification, and tracking on players in ice hockey video clips.
The classification module classifies players as either skaters or goalies, and as members of the home or away team.
Source code can be found in the `src` directory, and the LaTeX code for the final paper is in the `final-paper` directory.

Key source files:
* **config.py**: holds configuration settings for the entire project, including file paths and training hyperparameters.
* **main.py**: the point of entry script for training the model, performing inference, and doing evaluation/visualizations for the trained model
* **AnnotationParser.py**: a class that parses the CVAT annotation data into a format that is ready for consumption by the rest of the pipeline
* **Preprocessor.py**: a class that performs preprocessing tasks including calling the AnnotationParser, preparing data for consumption by the YOLO implemenation, and dataset splitting
* **ModelTrainer.py**: a class for handling training of the YOLO model 
* **ObjectDetector.py**: a class that uses YOLO and ByteTrack to perform object detection and tracking on provided video  
* **HyperparameterTuner.py**: a class that manages hyperparameter tuning with Optuna for both training and inference hyperparameters
* **ModelEvaluator.py**: a class that performs inference on and calculates key evaluation metrics for a provided clip
* **Visualizer.py**: a class for creating customized video visualizations of the model's predictions 

## Running Locally

To run the full training, inference, evaluation, and visualization pipeline:

    python main.py

The script takes the following arguments:
* `--skip-preprocessing`: skips preprocessing steps and instead reads the already preprocessed data from disk
* `--skip-training`: skips training the model and proceeds directly to model evaluation and visualization
* `--tune-hyperparameters`: performs hyperparameter tuning, writing the optimized hyperparameters to disk after completion