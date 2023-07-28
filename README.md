# Binary-AI
Title: Neural Network Binary Classification with Backpropagation

Description:

    This GitHub repository contains a Python script implementing a basic feedforward neural network for binary classification using the backpropagation algorithm. The neural network is trained to classify binary input data into one of two classes (0 or 1). The repository also includes sample data and a JSON file for version information.

Key Features:

    main.py: This is the main Python script that defines the NeuroYurii class, representing the neural network. It includes methods for forward propagation (send_signal), training (train), and resetting/restoring weights (reset_memory and restore_memory).
    version.json: A JSON file that contains information about the version, name, and maintainer of the project.
    data/: A folder containing sample data for the binary classification problem. It includes input data (x) and target data (y).
    plots/: A folder to store the training progress plots generated during training.

Usage:

    Clone the repository to your local computer using git clone https://github.com/your-username/neural-network-binary-classification.git.
    Make sure you have Python (version 3.10 or above) and the required libraries (execute pip install -r requirements.txt) installed.
    Run the main.py script to train the neural network. You can modify hyperparameters such as the number of hidden neurons, learning rate, and epochs in the script.
    After training, the script will generate training progress plots in the plots/ folder.
    
    
Note: 

    The reset_memory and restore_memory methods in the NeuroYurii class are currently not being utilized in the training process.        
    You can further explore and enhance these methods if needed.

Feel free to use, modify, and contribute to this project. If you find any issues or have suggestions, please open an issue or create a pull request.


![alt text](https://github.com/NoNFake/Binary-AI/blob/master/plots/17%3A48training_plot.png)





