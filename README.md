# Classifying-Handwritten-Digits
Classification of Handwritten Digits using ML Algorithms on the MNIST dataset
Introduction
The task of handwritten digit classification is a foundational problem in machine learning and computer vision, serving as a benchmark for evaluating the performance of various algorithms. This project aims to delve deeper into this task by exploring the MNIST dataset and using multiple machine learning models to identify the most effective approach for accurate digit recognition.
Dataset Overview
MNIST Dataset
•	Source: The dataset is sourced from MNIST, a widely used benchmark dataset in the machine learning community, containing 70,000 grayscale images of handwritten digits (0-9).
•	Data Distribution: The dataset is partitioned into 60,000 training images and 10,000 test images, providing a balanced distribution across classes.
Data Characteristics
•	Image Dimensions: Each image is represented as a 28x28 pixel matrix, resulting in a feature vector of length 784 after flattening.
•	Pixel Values: The pixel values range from 0 to 255, indicating the intensity of the grayscale color.
Data Preprocessing
Normalization
•	Objective: To ensure consistent and effective training, the pixel values are normalized to the range (0, 1) by dividing each pixel value by 255.
Data Splitting
•	Training-Test Split: The dataset is divided into training and test sets using an 80:20 ratio, facilitating robust model evaluation on unseen data.
Feature Scaling
•	Standardization: The pixel values are standardized using scikit-learn's StandardScaler to transform the data such that each feature has a mean of 0 and a standard deviation of 1, enhancing the convergence and performance of the models.
Machine Learning Models
Model Selection
•	SVM: Leveraging the versatility of Support Vector Machines, we employ an RBF kernel to capture nonlinear relationships within the data, with hyperparameters C=10 and Gamma=0.01 to control regularization and kernel coefficient, respectively.
•	Random Forest: Harnessing the power of ensemble learning, Random Forest is utilized with 100 estimators to construct multiple decision trees, enabling robust and accurate predictions while mitigating overfitting.
•	k-NN: Utilizing the k-Nearest Neighbors algorithm, we consider the 5 nearest neighbors to classify each instance based on its proximity in the feature space, offering a simple yet effective approach for classification tasks.
Model Training and Evaluation
Training Procedure
•	Model Training: Each selected model is trained on the standardized training data using scikit-learn's fit method, optimizing the model parameters to minimize the classification error.
Evaluation Metrics
•	Accuracy: Serving as the primary evaluation metric, accuracy quantifies the proportion of correct predictions made by the model, providing a holistic measure of its performance.
•	Classification Report: In addition to accuracy, a detailed classification report is generated, encompassing precision, recall, and F1-score for each class, offering insights into the model's performance across different categories.
•	Confusion Matrix: Visual representation of the confusion matrix aids in understanding the distribution of true positives, false positives, true negatives, and false negatives, elucidating the model's predictive behavior.
Overview of Model Performance
The evaluation of the machine learning models on the MNIST dataset yielded the following accuracy scores:
•	k-NN: Achieved an accuracy of 0.9460
•	SVM: Obtained an accuracy of 0.8580
•	Random Forest: Registered an accuracy of 0.9316

