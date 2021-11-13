# Deep Learning - Charity Funding Predictor

## Background

The non-profit foundation Alphabet Soup wants to create an algorithm to predict whether or not applicants for funding will be successful. Created a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup, based on a dataset containing more than 34,000 organizations that have received funding from Alphabet Soup over the years.

### Preprocess the data

Used Pandas and Scikit-Learn’s `StandardScaler()` to preprocess the dataset. Cleaned data, and for columns with more than 10 unique values, determined the number of data points for each unique value. Selected a cutoff point to bin "rare" categorical variables together in a new value, `Other`. Used `pd.get_dummies()` to one-hot encode categorical variables.

### Compile, Train, and Evaluate the Model

Designed a neural network to create a binary classification model that can predict if an Alphabet Soup–funded organization will be successful based on features in this dataset. 

1. Created a neural network model by assigning the number of input features and nodes for each layer using Tensorflow Keras.
2. Created the hidden layers and chose appropriate activation function.
3. Created an output layer with an appropriate activation function.
4. Compiled and trained the model.
5. Created a callback that saves the model's weights every 5 epochs.
6. Evaluated the model using the test data to determine the loss and accuracy.
7. Saved and exported your results to an HDF5 file, and name it `AlphabetSoupCharity.h5`.

### Optimize the Model

Optimized model in order to achieve a target predictive accuracy higher than 75%. 

### Results

See file 'CharityFund Model Optimization Report' for a summary of hypertunings, findings and results.  
