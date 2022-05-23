# Face-Recognition
Face Recognition with Machine learning (PCA+KNN &amp; PCA+DF/RF)

### Introduction
Face recognition system compares the image with the database to perform function of identification. We are interested 
in building a classification model for the individuals in our database and predicting the accurate individual from the 
input image. We would also like to investigate whether we should apply Principal Component Analysis (PCA) to 
simplify the large dataset before classification. We have proposed three approaches for classification. They are using
PCA with K-Nearest Neighbors (KNN), PCA with decision tree and PCA with random forest. Comparison of the three 
approaches would also be made.

### Data Description 
The dataset “archive.zip” consists of 400 images in total from 40 people with 10 different images each (Figure 1). It 
can be downloaded through Kaggle (https://www.kaggle.com/kasikrit/att-database-of-faces). We assign the labels 
𝑠1, … , 𝑠40 for recognizing each person. We divide it into training dataset and testing dataset, each contains 40 people 
with 5 images each, i.e., 50% of the dataset as training data and the other 50% as the testing data.


The rationale of doing dimensional reduction on our data is to reduce the complexity and boost the runtime of the 
program. The main idea is we try to project the data to a smaller dimension to reduce the burden to the computer 
system. Meanwhile, we would like to preserve more information, so we borrow the concept of unitary projection. We
rotate or sign flip of the coordinate system and our vector length and mean square vector length can be conserved. We 
can extend this concept to Karhunen-Loeve transformation by using the eigen matrix of the autocorrelation matrix to 
do the unitary transformation. We can then keep the first n coefficients with minimized mean square error. Therefore, 
we try to apply PCA to reduce the dimensions of our data to see if this can generate a satisfying classification rate
