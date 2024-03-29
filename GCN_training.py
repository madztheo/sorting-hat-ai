"""Script for executing model training on local data."""
from pathlib import Path
from typing import Dict, List

import json
import pickle

import numpy as np
from feltlabs.config import OceanConfig, TrainingConfig, parse_training_args
from numpy.typing import NDArray

import networkx as nx
from sklearn.preprocessing import OneHotEncoder
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def get_datasets(config: OceanConfig) -> Dict[str, List[Path]]:
    """Get all dataset paths provided in Ocean's compute job environment.

    Args:
        config: FELT config object containing input folder path

    Returns:
        dictionary mapping dataset DID to list of file paths
    """
    datasets = {}
    for path in config.input_folder.iterdir():
        if path.is_dir():
            # path.name corresponds to DID of dataset
            datasets[path.name] = [p for p in path.glob("**/*") if p.is_file()]
    return datasets


def load_data(config: OceanConfig) -> NDArray:
    """Load data and return them as single numpy array.

    Args:
        config: FELT config object containing input folder path

    Returns:
        numpy array containing data from all files
    """
    datasets = get_datasets(config)

    # Here we assume that datasets are CSV format:
    data = []
    for did, files in datasets.items():
        for file in files:
            data.append(np.genfromtxt(file, delimiter=',', dtype=str, skip_header=True))

    # We concatenate all numpy arrays into one
    data = np.concatenate(data, axis=0)
    return data


def main(config: TrainingConfig) -> None:
    """Main function executing the data loading, training and saving outputs

    Args:
        config: training config object provided by FELT containing all paths
    """
    # Load data as numpy array
    data = load_data(config)

    # Create model and train it
    trained_weights = train_model(data)

    # We convert trained_value (NDArray type) to bytes
    # It can be later loaded using np.frombuffer(model_bytes)
    model_bytes = json.dumps({
        "W1": pickle.dumps(trained_weights[0]).decode('latin-1'), 
        "W2": pickle.dumps(trained_weights[1]).decode('latin-1')
        })

    # Save models into output folder. You have to name output file as "model"
    with open(config.output_folder / "model", "w+") as f:
        f.write(model_bytes)

    print("Training finished.")


def train_model(data: NDArray) -> List[NDArray]:
    """Main function training the model

    Args:
        data: data used to train the model
        
    Returns:
        a list of two numpy arrays representing the input-to-hidden and 
        hidden-to-output weights
    """
    # Preprocess data to turn into a graph
    G, original_node, neighbours_lvl1 = get_graph(data)
    
    # Order the data
    order = sorted(list(G.nodes()))

    # Create an adjacency matrix from the data
    A = nx.to_numpy_array(G, nodelist= order) 
    A = np.matrix(A)

    # Create an identity matrix from the data
    I = np.eye(*A.shape)

    # Add a self-loop to each node
    A_tilde = A.copy() + I

    # Create the diagonal node-degree matrix of A_tilde
    D_hat = np.array(np.sum(A_tilde, axis=0))[0]

    # Apply symmetric normalization
    D_inv = np.matrix(np.diag(D_hat**(-0.5)))
    A_hat = D_inv * A_tilde * D_inv

    # define feature matrix 
    X = I

    # Semi-supervised Node Classification
    A_Hat = tf.constant(A_hat, 'float')

    # ---------------------------------------------------------------- 
    X_train = I
    node_labels = [2 if v == original_node else 
                   2 if v in neighbours_lvl1 else 1 for v in G] #TODO: review?
    y_train = np.array(node_labels)
    Y_train = OneHotEncoder(categories='auto').fit_transform(y_train.reshape(-1, 1))
    Y_train = Y_train.toarray()

    X = tf.placeholder('float', [X_train.shape[0], X_train.shape[1]],name = 'X')  
    Y = tf.placeholder('float', [Y_train.shape[0], Y_train.shape[1]],name = 'Y') 
    mask = tf.placeholder('float', [X_train.shape[0]],name = 'Mask') 
    # ---------------------------------------------------------------- 
    # Trainable parameters
    n_input = G.number_of_nodes()
    n_hidden = 4
    n_classes = 2

    params = {
        # Input to hidden weights
        'W_1': tf.Variable(tf.random_normal([n_input, n_hidden], mean= 0, stddev= 1), name = 'W_1'),
        # Hidden to output weights
        'W_2': tf.Variable(tf.random_normal([n_hidden, n_classes], mean= 0, stddev= 1), name = 'W_2')
    }
    # ---------------------------------------------------------------- 

    def GCN_supervised(X, params):
        layer_1 = tf.matmul(tf.matmul(A_Hat, X), params['W_1'])
        layer_1 = tf.nn.relu(layer_1, name = 'layer_1')  
        output = tf.matmul(tf.matmul(A_Hat, layer_1), params['W_2'], name = 'output_layer') 
        return output

    Z = GCN_supervised(X, params)
    Y_prob = tf.nn.softmax(Z, name = 'Y_prob')
    mask_tr = np.zeros(G.number_of_nodes())
    mask_tr[[0, G.number_of_nodes() - 1]] = 1
    mask_tr = np.array(mask_tr, dtype=bool)
    fro = tf.matmul(Z, tf.transpose(Z)) - A_Hat
    reg = tf.sqrt(tf.reduce_sum(tf.square(fro)))
    lambdaa = 0.02
    cost = masked_softmax_cross_entropy(preds = Y_prob, labels = Y, mask = mask_tr) + lambdaa * reg
    # ---------------------------------------------------------------- 
    num_epochs = 600
    learning_rate = 0.02
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) 
    accuracy = masked_accuracy(preds = tf.round(Y_prob), labels = Y, mask = mask_tr)
    # ----------------------------------------------------------------
    # Global variables initializer 
    init = tf.global_variables_initializer() 
    # Starting the Tensorflow Session 
    weights_arrays_list = []
    with tf.Session() as sess:       
        # Initializing the variables 
        init.run()  
        cost_history = []
        accuracy_history = [] 
        embeddings = []
        for epoch in range(num_epochs):    
            # Running the optimizer 
            sess.run(optimizer, feed_dict={X: X_train, Y: Y_train, mask: mask_tr}) 
            # Calculating & storing cost, accuracy and embeddings on current epoch 
            c, acc, z = sess.run([cost, accuracy, Z], feed_dict = {X: X_train, Y: Y_train, mask: mask_tr}) 
            cost_history.append(c) 
            accuracy_history.append(acc)
            embeddings.append(z)
    # ----------------------------------------------------------------
    # Store weights arrays
        var = [v for v in tf.trainable_variables() if v.name == "W_1:0"][0]
        W_1 = sess.run(var)
        weights_arrays_list.append(W_1)
        var = [v for v in tf.trainable_variables() if v.name == "W_2:0"][0]
        W_2 = sess.run(var)
        weights_arrays_list.append(W_2)
    tf.reset_default_graph()
    return weights_arrays_list


def get_graph(data: NDArray):
    W = nx.Graph()
    nodes_map = {}
    neighbours_l1 = []
    current_index = 0
    #Save original node
    original_node = data[0][0]
    # Add edges to the graph
    for row in data:
        source = row[0]
        target = row[1]
        if not source in nodes_map:
            nodes_map[source] = current_index
            current_index += 1
        if not target in nodes_map:
            nodes_map[target] = current_index
            current_index += 1
        if source == original_node:
            neighbours_l1.append(nodes_map[target])
        elif target == original_node:
            neighbours_l1.append(nodes_map[source])
        W.add_edge(nodes_map[source], nodes_map[target])
    return W, list(W)[0], neighbours_l1


# These 2 next functions are from tkipf/gcn 
def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)


if __name__ == "__main__":
    # Get config - we recommend using config parser provided by FELT Labs
    # It automatically provides all input and output paths and other settings
    config = parse_training_args()
    main(config)