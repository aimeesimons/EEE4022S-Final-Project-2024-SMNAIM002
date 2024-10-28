### The Model Training consists of 6 programs

- X_classify.py -> which trains the classification model
- X_detect.py -> which trains the detection model
- X_locate.py -> which loops through all the transmission lines and trains each model
- X_locate_graphs_EdgesCNN.py -> which trains the edge model
- X_locate_graphs_NodesCNN.py -> which trains the node model
- X_locator_test.py -> which tests the line location models on 15 locations.

All datasets can be found in * Data/ *. A machine with at least 32GB of RAM is required to train the detection and classification models, while a machine with at least 16GB is required to train the Graph Models. 
