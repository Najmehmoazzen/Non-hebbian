# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint        # for saving models
from tensorflow.keras.losses import MeanSquaredError          # loss function
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

# %%
coupling=14.75                               # Coupling value of Kuramoto model network
num_nodes = 100
window_sizes = [2]                # List of window sizes to be used for training
n_epochs= 150                               # Number of epochs for training
model_name = "FCN"

# %%
'''# Load the data
data = np.load(f'./Data/Degree_Radian=1.57_copling={coupling}layer2(time)VS(Node).npz')

print("Available arrays in the .npz file:", data.files)
array_key = data.files[0]                                       # Takes the first array
print(f"Using array: {array_key}")

scale_factor = 100  
array_data = data[array_key].astype(np.float32) / scale_factor

df = pd.DataFrame(array_data)
df.columns = [f'theta{i+1}' for i in range(len(df.columns))]    # Rename columns
'''

# %%
# Read and process the data, skipping the first column (time)
phase_data = []
file_path = f"/home/Najmeh/Non-hebbian/simple_kuramoto/Save/Phases/k={coupling}0000.txt"

with open(file_path, "r") as textFile:
    for line in textFile:
        values = line.strip().split()
        if len(values) > 1:
            phase_data.append(values[1:])  # Skip first column

# Convert to NumPy array WITHOUT transpose
phase_data = np.array(phase_data, dtype=float)  # Shape: (time, nodes)

# Create DataFrame: rows = time, columns = nodes
num_timesteps, num_nodes = phase_data.shape
df = pd.DataFrame(
    phase_data,
    index=[f'Time_{t}' for t in range(num_timesteps)],
    columns=[f'Node_{i}' for i in range(num_nodes)]
)

# Show part of the DataFrame
print(df.head())



# %%
df

# %%
def df_to_X_y(df, window_size):
    df_as_np = df.to_numpy()
    X, y = [], []
    for i in range(len(df_as_np) - window_size):
        X.append(df_as_np[i:i + window_size])           # shape: (window_size, n_features)
        y.append(df_as_np[i + window_size])             # shape: (n_features,)
    return np.array(X), np.array(y)

# %%
def ann(num_nodes, window_size, model_name):
    input_layer = Input(shape=(window_size, num_nodes), dtype=tf.float32)

    # Apply cosine and sine element-wise
    cos_layer = Lambda(lambda x: tf.math.cos(x))(input_layer)
    sin_layer = Lambda(lambda x: tf.math.sin(x))(input_layer)
    x = Concatenate(axis=-1)([cos_layer, sin_layer])  # Shape: (window_size, 2*num_features)

    if model_name == "FCN":
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        #x = Dense(1280, activation='relu')(x)

    elif model_name == "LSTM": 
        x = LSTM(128)(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
   
    output = tf.keras.layers.Dense(num_nodes, activation='linear')(x)
        
    model = Model(inputs=input_layer, outputs=output)
    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
    return model

# %%
def create_and_train_model(model_name, window_size, epochs):
    # Make inputs and actual labels of them
    X, y = df_to_X_y(df, window_size)
    # Split train and test data
    X_train, y_train = X[:35000], y[:35000]
    X_val, y_val = X[35000:40000], y[35000:40000]
    X_test, y_test = X[40000:], y[40000:]
    print("Train:", X_train.shape, y_train.shape)
    print("Val:", X_val.shape, y_val.shape)
    print("Test:", X_test.shape, y_test.shape)
    num_nodes = X.shape[2]
    print("num_nodes = ", X.shape[2])

    model = ann(num_nodes, window_size, model_name)
    model.summary()
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, verbose=0)
    '''
    # Saving model
    model_filename = f'./Results/model_LSTM_hidden6400_J={coupling}_window{window_size}.keras'
    model.save(model_filename)'''

    # Make predictions
    predictions = model.predict(X_test)
    print(predictions)
    
    os.makedirs("./Results", exist_ok=True)
    np.savetxt(f"./Results/J={coupling}_window{window_size}_actual.txt", y_test, fmt='%.2f', delimiter='\t', newline='\n', encoding=None)
    np.savetxt(f"./Results/J={coupling}_window{window_size}_predicted.txt", predictions, fmt='%.2f', delimiter='\t', newline='\n', encoding=None)

    return history, predictions

# %%
# Dictionary to store training histories for each window size
histories = {}

for window_size in window_sizes:
    print(f"Training model with window size: {window_size}")
    history,_ = create_and_train_model(model_name, window_size, n_epochs)   # Create and train the model with the current window size
    histories[window_size] = history    

# %%
plt.figure(figsize=(6, 4))

for window_size, history in histories.items():
    plt.plot(history.history['loss'], label=f'Window Size: {window_size}')

plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.title('Training Loss vs. Epochs for Different Window Sizes')
plt.legend()
plt.savefig(f'./Results/training_loss_vs_epochs_J={coupling}.png')
plt.show()

# %%
def plot_results(window_size):


    # Define the colors for the custom cyclic colormap
    colors = [
        (1.0, 0.0, 1.0),  # Magenta (255, 0, 255)
        (1.0, 0.0, 0.0),  # Red (255, 0, 0)
        (1.0, 1.0, 0.0),  # Yellow (255, 255, 0)
        (0.0, 1.0, 0.0),  # Green (0, 255, 0)
        (0.0, 1.0, 1.0),  # Cyan (0, 255, 255)
        (0.0, 0.0, 1.0),  # Blue (0, 0, 255)
        (1.0, 0.0, 1.0)   # Magenta (255, 0, 255)
    ]

    # Create the custom cyclic colormap
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)

    fig = plt.figure()

    # _________________________________________________________
    ax_1 = plt.subplot(3, 4, (1, 4))
    # Read the input data
    with open(f"./Results/J={coupling}_window{window_size}_actual.txt") as textFile:  # ./Results/J={coupling}_actual.txt
        lines = [line.split() for line in textFile]
    lines = np.array(lines, dtype=float).transpose()  # Convert to numpy array and transpose


    # Subtract 2*pi from values greater than pi
    #lines = np.where(lines > np.pi, lines - 2 * np.pi, lines)
    # Add 2*pi to values less than -pi
    #lines = np.where(lines < -np.pi, lines + 2 * np.pi, lines)

    print(np.min(lines))
    print(np.max(lines))


    # plot
    vmin = 0
    vmax = 2*np.pi
    plt.imshow(lines, cmap=custom_cmap, aspect='auto', interpolation='nearest', vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(ticks=[vmin, 0, vmax])
    #cbar.ax.set_yticklabels(['-π', '0', 'π'])

    plt.title(f'Comparision (window {window_size})', fontsize=10)

    # plot
    # Customize tick parameters
    plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, labeltop=False, labelsize=10,
                    labelcolor='#262626')
    plt.tick_params(axis='y', which='both', left=True, right=False, labelleft=True, labelright=False, labelsize=10,
                    labelcolor='#262626')
    # Reverse the y-axis direction
    plt.gca().set_ylim(plt.gca().get_ylim()[::-1])
    # Set x and y limits
    plt.xlim(0, len(lines[0]))
    plt.ylim(0, num_nodes)
    # Set labels
    plt.ylabel('Node (i)', fontsize=10, labelpad=10)
    plt.xlabel('Time (s)', fontsize=10)
    # _________________________________________________________

    ax_2 = plt.subplot(3, 4, (5, 8))

    # Read the input data
    with open(f"./Results/J={coupling}_window{window_size}_predicted.txt") as textFile:
        lines2 = [line.split() for line in textFile]
    lines2 = np.array(lines2, dtype=float).transpose()  # Convert to numpy array and transpose

    # Subtract 2*pi from values greater than pi
    lines2 = np.where(lines2 > 2*np.pi, lines2 - 2 * np.pi, lines2)
    # Add 2*pi to values less than -pi
    lines2 = np.where(lines2 < 0, lines2 + 2 * np.pi, lines2)

    print(np.min(lines2))
    print(np.max(lines2))
    # plot
    vmin = 0
    vmax = 2*np.pi
    plt.imshow(lines2, cmap=custom_cmap, aspect='auto', interpolation='nearest', vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(ticks=[vmin, 0, vmax])
    #cbar.ax.set_yticklabels(['-π', '0', 'π'])

    # plot
    # Customize tick parameters
    plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, labeltop=False, labelsize=10,
                    labelcolor='#262626')
    plt.tick_params(axis='y', which='both', left=True, right=False, labelleft=True, labelright=False, labelsize=10,
                    labelcolor='#262626')
    # Reverse the y-axis direction
    plt.gca().set_ylim(plt.gca().get_ylim()[::-1])
    # Set x and y limits
    plt.xlim(0, len(lines2[0]))
    plt.ylim(0, num_nodes)
    # Set labels
    plt.ylabel('Node (i)', fontsize=10, labelpad=10)
    plt.xlabel('Time (s)', fontsize=10)
    # _________________________________________________________
    def kuramoto_similarity(theta1, theta2):
        z1 = np.exp(1j * theta1)
        z2 = np.exp(1j * theta2)
        return np.abs((z1 + z2) / 2)


    '''def phase_similarity(theta1, theta2):
        delta = np.angle(np.exp(1j * (theta1 - theta2)))  # in range [-π, π]
        distance = np.abs(delta)  # in range [0, π]
        similarity = 1 - (distance / np.pi)  # normalize to [0, 1]
        return similarity'''
    s = (len(lines[:, 0]),len(lines[0, :]))
    lines3 = np.zeros(s)
    ax_3 = plt.subplot(3, 4, (9, 12))
    for i in range(lines.shape[0]):
        for j in range(lines.shape[1]):
            lines3[i][j] = kuramoto_similarity(lines2[i][j] ,lines[i][j])

    print(np.min(lines3))
    print(np.max(lines3))
    # plot
    plt.imshow(lines3, cmap='binary', aspect='auto', interpolation='nearest')
    cbar = plt.colorbar(ticks=[-1.001, 0, 1])
    # cbar.ax.set_yticklabels(['-1', '0', '1'])
    # Customize tick parameters
    plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, labeltop=False, labelsize=10,
                    labelcolor='#262626')
    plt.tick_params(axis='y', which='both', left=True, right=False, labelleft=True, labelright=False, labelsize=10,
                    labelcolor='#262626')
    # Reverse the y-axis direction
    plt.gca().set_ylim(plt.gca().get_ylim()[::-1])
    # Set x and y limits
    plt.xlim(0, len(lines3[0]))
    print(len(lines3[0]))
    plt.ylim(0, num_nodes)
    # Set labels
    plt.ylabel('Node (i)', fontsize=10, labelpad=10)
    plt.xlabel('Time (s)', fontsize=10)
    # Save the figure as a .png file
    plt.subplots_adjust(top=0.97, bottom=0.08, hspace=0.3, wspace=0.44)


    plt.gcf().set_size_inches(12, 3)
    #plt.savefig(f'/content/drive/My Drive/Colab Notebooks/Forecasting phases project/Results/result_J={coupling}_window{window_size}.pdf')
    plt.savefig(f'./Results/result_J={coupling}_window{window_size}.png' , dpi=300)

# %%
for window_size in window_sizes:
    plot_results(window_size)

# %%


# %%



