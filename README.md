# Dog-Breed-Prediction

**Problem Statement:**

### Dog Breed Prediction Model
The dog breed prediction model is built using a Convolutional Neural Network (CNN) to classify dog images into specific breeds. The breeds considered in this example include Scottish Deerhound, Maltese Dog, and Bernese Mountain Dog.

### Model Architecture
The model is structured using the Keras Sequential API, allowing layers to be added sequentially, which is ideal for creating simple feed-forward networks.

### Input Layer

Input Shape: The model expects input images of size 224x224 pixels with 3 color channels (RGB).

#### Conv2D Layer:

Filters: 64
Kernel Size: 5x5
Activation: ReLU
Purpose: This layer applies 64 convolutional filters to the input image to extract features such as edges, textures, and patterns.
MaxPooling Layer:

Pool Size: 2x2
Purpose: This layer reduces the spatial dimensions (height and width) of the feature maps, reducing the computational load and focusing on the most prominent features.
Second Convolutional Block

Conv2D Layer:
Filters: 32
Kernel Size: 3x3
Activation: ReLU
Regularization: L2 regularization to prevent overfitting by penalizing large weights.
MaxPooling Layer: Another 2x2 pooling layer to further reduce the spatial dimensions.
Third Convolutional Block

Conv2D Layer:
Filters: 16
Kernel Size: 7x7
Activation: ReLU
Regularization: L2 regularization is applied again.
MaxPooling Layer: A 2x2 pooling layer for further dimensionality reduction.
Fourth Convolutional Block

### Conv2D Layer:
Filters: 8
Kernel Size: 5x5
Activation: ReLU
Regularization: L2 regularization is applied.
MaxPooling Layer: Final 2x2 pooling layer to further reduce the feature map size.
Flattening Layer

The Flatten() layer converts the 2D feature maps into a 1D feature vector to prepare it for the fully connected layers.
Fully Connected Layers

### Dense Layer 1:
Units: 128
Activation: ReLU
Regularization: L2 regularization is applied.
Dense Layer 2:
Units: 64
Activation: ReLU
Regularization: L2 regularization is applied.
These layers help in learning complex patterns from the extracted features.
Output Layer

### Dense Layer:
Units: Number of dog breeds (determined by len(CLASS_NAMES))
Activation: Softmax, which outputs a probability distribution over the dog breeds, indicating the model's confidence in each breed prediction.

### Loss Function:
 Categorical Crossentropy, which is standard for multi-class classification problems.
Optimizer: Adam optimizer with a learning rate of 0.0001, which is effective in handling complex, high-dimensional data.
Metrics: Accuracy is used to evaluate the model's performance.

### Process:
#### Step 1 : Loading The Dataset

![image](https://github.com/user-attachments/assets/42754fff-feee-4553-8653-df727b0127b4)

#### Step 2 : Building The Model

![image](https://github.com/user-attachments/assets/b1f90794-e896-48f8-bb18-bf161369e4ef)

#### Step 3 : Training The Model

![image](https://github.com/user-attachments/assets/b6862732-b60d-4b01-a7c1-05e43bac902a)

#### Step 4 : Displaying The Model Accuracy

![image](https://github.com/user-attachments/assets/bfe06fca-a0de-41a3-babd-0f11c151cca4)

#### Step 5 : Testing the model with Image

![image](https://github.com/user-attachments/assets/50fb3022-25e1-481d-bb42-03a7223d2f29)

#### Step 6 : Lauching The App on streamlit 

![image](https://github.com/user-attachments/assets/61604d0a-b322-4c9a-886b-fb77de1f6690)
