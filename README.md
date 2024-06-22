# Diabetes Prediction using SVM

This project aims to predict whether a person has diabetes or not using the PIMA Indian Diabetes Dataset and a Support Vector Machine (SVM) classifier.

## Table of Contents
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Prediction](#prediction)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Dataset

The dataset used in this project is the PIMA Indian Diabetes Dataset, which consists of 768 instances with 8 features and a target variable `Outcome` indicating whether the person has diabetes (1) or not (0).

| Feature                | Description                    |
|------------------------|--------------------------------|
| Pregnancies            | Number of times pregnant       |
| Glucose                | Plasma glucose concentration   |
| BloodPressure          | Diastolic blood pressure (mm Hg)|
| SkinThickness          | Triceps skin fold thickness (mm)|
| Insulin                | 2-Hour serum insulin (mu U/ml) |
| BMI                    | Body mass index                |
| DiabetesPedigreeFunction | Diabetes pedigree function    |
| Age                    | Age (years)                    |
| Outcome                | Class variable (0 or 1)        |

## Installation

To run this project, you need to have Python installed along with the following libraries:
- numpy
- pandas
- scikit-learn

You can install the required libraries using pip:

```sh
pip install numpy pandas scikit-learn
```

## Usage

1. Clone the repository:

```sh
git clone https://github.com/your-username/diabetes-prediction.git
```

2. Navigate to the project directory:

```sh
cd diabetes-prediction
```

3. Place the `diabetes.csv` file in the project directory.

4. Run the Jupyter notebook or the Python script to train the model and make predictions.

## Model Training and Evaluation

1. **Data Collection and Analysis:**
   - Load the dataset into a pandas DataFrame.
   - Perform exploratory data analysis (EDA) to understand the data distribution.

2. **Data Preprocessing:**
   - Separate the features (X) and target variable (Y).
   - Standardize the features using `StandardScaler`.

3. **Model Training:**
   - Split the dataset into training and testing sets.
   - Train the SVM classifier with a linear kernel on the training data.

4. **Model Evaluation:**
   - Evaluate the model using accuracy scores on both training and testing data.

```python
from sklearn import svm
from sklearn.metrics import accuracy_score

classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Accuracy on training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of the training data:', training_data_accuracy)

# Accuracy on test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the test data:', test_data_accuracy)
```

## Prediction

You can make predictions on new data using the trained model. Example:

```python
input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
std_data = scaler.transform(input_data_reshaped)
prediction = classifier.predict(std_data)

if prediction[0] == 0:
    print('The person is not diabetic')
else:
    print('The person is diabetic')
```

## Results

The model achieved an accuracy of approximately 78.66% on the training data and 77.27% on the test data.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.


---
