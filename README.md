# Iris Flower Classification

## Description

This project demonstrates the use of a neural network to classify Iris flowers into three species (Setosa, Versicolor, and Virginica) based on their sepal and petal measurements. It utilizes the TensorFlow library in Python to build and train the model.

## Purpose

The primary goal of this code is to build and train a machine learning model capable of accurately classifying Iris flowers based on their physical characteristics. This is a classic example of a supervised learning problem, where the model learns from labeled data (Iris species) to make predictions on unseen data.

## Approach

1. **Data Loading and Preprocessing:**
   - The code begins by loading the Iris dataset from a CSV file using the Pandas library.
   - The dataset is then split into features (X) and labels (y), representing the sepal and petal measurements and the corresponding Iris species, respectively.
   - The labels are encoded using one-hot encoding to represent them as numerical values suitable for the neural network.
   - The data is further divided into training and testing sets to evaluate the model's performance.

2. **Model Building:**
   - A sequential neural network model is created using TensorFlow's Keras API.
   - The model consists of two hidden layers with ReLU activation functions and an output layer with a softmax activation function for multi-class classification.

3. **Model Training:**
   - The model is compiled with the 'rmsprop' optimizer and 'categorical_crossentropy' loss function.
   - It is then trained on the training data for a specified number of epochs.
   - The training progress is visualized using plots of loss and accuracy over epochs.

4. **Model Evaluation:**
   - The trained model is evaluated on the testing data to assess its performance.
   - Test loss and accuracy are calculated and reported.
   - Predictions are made on the test data, and the results are compared with the actual labels to determine the model's accuracy in classifying Iris flowers.

## Dependencies

- Python 3.x
- TensorFlow 2.x
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Usage

1. Install the required dependencies.
2. Download the Iris dataset (Iris.csv) and place it in the same directory as the code.
3. Run the code in a Google Colab environment or a Jupyter Notebook.

## Results

The model achieved an accuracy of [insert accuracy value] on the test dataset.


## License

This project is licensed under the MIT License.
