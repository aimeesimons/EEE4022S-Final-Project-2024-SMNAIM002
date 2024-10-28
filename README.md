# EEE4022S Final Project 2024 SMNAIM002
## Applications of Artificial Intelligence to Fault Identification in Power Systems
* Welcome to my Github repository
* This repository contains all the data and scripts used to complete this project

This repository contains the following directories and files:
- Data Collection and Preprocessing/ (which pertains to the scripts used to collect and prepare the data for training)
  - DataCollection.ipynb
  - feature_extraction.ipynb
  - Noise.ipynb
- Data/ (which pertains to the label data and linked datasets)
- DigSilent Models/  (which pertains to the scripts used to collect and prepare the data for training)
  - FinalModelEEE4002S_1 -> which is the 21-Bus system
  - TesterModelEEE4022S_1 -> which is the built tester model
- Model Training/ (which pertains to the Python files utilised in training the respective models)
  - X_classify.py
  - X_detect.py
  - X_locate.py
  - X_locate_graphs_EdgeCNN.py
  - X_locate_graphs_NodeCNN.py
  - X_locator_test.py
- Models/ (which pertains to the saved and trained models)
  - LineModels_new/ -> all the saved line models
  - ClassificationModel_mixed_noise_2.joblib -> Classification Model
  - DetectionModel_mixed_noise_2.joblib -> Detection Model
  - ClassificationModel_mixed_noise_1.joblib -> Classification Model initial
  - DetectionModel_mixed_noise_1.joblib -> Detection Model initial
  - graphs_edges_ordinal_encoder1.pkl -> used for encoding and decoding the respective labels for each graph
  - graphs_nodes_ordinal_encoder1.pkl -> used for encoding and decoding the respective labels for each graph
  - label_encoder.pickle -> used for encoding and decoding the labels for the classification model
  - model_edge_2.pth -> Edge Model
  - model_node_2.pth -> Node Model
  - ordinal_encoder.pickle -> used for encoding and decoding the labels for the detection model
- Results/  (which pertains to the results received from the line models)
- Simulation Scripts/ (which pertains to the Python files utilised in DigSilent's embedded Python environment and used to run various simulations)
  - GraphsScript.py -> for collecting the data to create the graphs
  - IntegrationTest.py -> for integrating all fault identification tasks together
  - NoFaultscript.py -> for running EMT simulations with no faults
  - shortcircuitScript.py -> for defining and running a short-circuit event and collecting the data 


