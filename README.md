# Handwritten-Digit-Recognition-using-SVM-Classifier
ML Python Project
---------------------------------------------------------------------------------------
# Preview
![](https://github.com/developer-venish/Handwritten-Digit-Recognition-using-SVM-Classifier/blob/main/demo.png)

![](https://github.com/developer-venish/Handwritten-Digit-Recognition-using-SVM-Classifier/blob/main/Demo(1).png)

---------------------------------------------------------------------------------------

# Accuracy

![](https://github.com/developer-venish/Handwritten-Digit-Recognition-using-SVM-Classifier/blob/main/accuracy.png)

---------------------------------------------------------------------------------------

All the code in this project has been tested and run successfully in Google Colab. I encourage you to try running it in Colab for the best experience and to ensure smooth execution. Happy coding!

---------------------------------------------------------------------------------------

# Working of The Code
The code you provided demonstrates a basic machine learning workflow using the scikit-learn library. Here's a step-by-step explanation of the code:

1. Import necessary libraries:
   - `numpy` for numerical operations.
   - `load_digits` from `sklearn.datasets` to load the digits dataset.
   - `matplotlib.pyplot` for plotting and visualization.
   - `train_test_split` from `sklearn.model_selection` for splitting the data into training and testing sets.
   - `svm` from `sklearn` for the support vector machine model.
   - `accuracy_score` from `sklearn.metrics` for evaluating model accuracy.

2. Load the digits dataset using `load_digits()` function and store it in the `dataset` variable.

3. Print the data and target attributes of the dataset to see the input features and target values.

4. Print the shape of the data and images arrays to understand the dimensions of the dataset.

5. Calculate the length of the images in the dataset.

6. Set up plotting configurations to display grayscale images.

7. Display an image from the dataset using `matshow()` and `show()` functions.

8. Reshape the images array into a two-dimensional array using `reshape()`.

9. Store the reshaped images in `X` and the target values in `Y`.

10. Split the data into training and testing sets using `train_test_split()`. The testing set size is 25% of the data, and a random state is set for reproducibility.

11. Create an SVM model with gamma=0.001.

12. Fit the SVM model on the training data using `fit()`.

13. Predict the target values for a single image using `predict()`.

14. Plot the predicted image with the corresponding label.

15. Predict the target values for the testing set using `predict()`.

16. Calculate and print the accuracy of the model using `accuracy_score()`.

17. Create three additional SVM models with different configurations.

18. Fit the new models on the training data.

19. Predict the target values for the testing set using the new models.

20. Calculate and print the accuracy of each model.

The code demonstrates loading a dataset, preprocessing it, training a machine learning model, making predictions, and evaluating the model's accuracy.

---------------------------------------------------------------------------------------

SVM (Support Vector Machine) Classifier is a supervised machine learning algorithm that can be used for classification and regression tasks. It is particularly effective in solving binary classification problems, but it can also be extended to handle multi-class classification.

The main idea behind SVM is to find an optimal hyperplane in a high-dimensional feature space that maximally separates the different classes. The hyperplane is selected such that it maximizes the margin, which is the distance between the hyperplane and the nearest data points of each class. The data points that lie closest to the hyperplane are called support vectors.

SVM works by transforming the input data into a higher-dimensional feature space using a kernel function, which allows for non-linear classification. In the transformed feature space, SVM constructs a decision boundary that separates the classes. New unseen data points can then be classified based on which side of the decision boundary they fall.

SVM has several advantages:
- It can handle high-dimensional data efficiently.
- It is effective in cases where the number of features is larger than the number of samples.
- It is less prone to overfitting compared to other algorithms.
- It can capture complex relationships in the data using different types of kernel functions.

However, SVM can be computationally expensive, especially for large datasets. Additionally, the performance of SVM heavily depends on selecting the appropriate kernel function and tuning the hyperparameters.

Overall, SVM Classifier is a powerful algorithm for classification tasks, offering flexibility and accuracy in separating different classes in the data.

---------------------------------------------------------------------------------------
