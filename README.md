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
with 5 images each, i.e., 50% of the dataset as training data and the other 50% as the testing data. The following figure shows 
all the faces in our dataset:

![all_faces](https://user-images.githubusercontent.com/101900124/169803583-eb44b3c2-ae13-4ac9-a9fc-0396cd1802a3.png)


### Data Pre-processing
The rationale of doing dimensional reduction on our data is to reduce the complexity and boost the runtime of the 
program. The main idea is we try to project the data to a smaller dimension to reduce the burden to the computer 
system. Meanwhile, we would like to preserve more information, so we borrow the concept of unitary projection. We
rotate or sign flip of the coordinate system and our vector length and mean square vector length can be conserved. We 
can extend this concept to Karhunen-Loeve transformation by using the eigen matrix of the autocorrelation matrix to 
do the unitary transformation. We can then keep the first n coefficients with minimized mean square error. Therefore, 
we try to apply PCA to reduce the dimensions of our data to see if this can generate a satisfying classification rate.

### Model Description & Method
1.PCA & KNN

We first compute the eigenfaces. We have our training faces I_1, ..., I_200 which has dimension 112 × 92 and we 
flatten each I_i as L_i with dimension 10304 × 1. We delete the common features to preserve the key features by 
subtracting the mean vector from each L_i as M_i and define 𝑆 = [M_1,...,M_200]. The mean face is like this:

![mean_face](https://user-images.githubusercontent.com/101900124/169807200-eeeacd05-a506-4e48-b6ce-9480f6f37cbe.png)

The covariance matrix C is built by S*t(S) with dimension 10304 × 10304 which is not practical. Hence, we compute 
the eigenvectors v_i of t(S)*S as we can obtain the eigenvectors u_i by S*v_i. Then we compute it and select the 
first 𝑘 eigenfaces. We calculate the weight vector 𝑤_𝑖 = t(u_i)*M_i, i = 1, ..., 𝑘. At this point, we have completed 
our PCA extraction part and is ready for the classification.

For an image L_j in our testing dataset, we compute M_j by subtracting the mean vector from L_j and project it to our 
k-eigenspace as its weight 𝜔_𝑗. Then we find the minimum Euclidean distance as ||𝑤𝑖 − 𝜔𝑗||2 for 𝑖 = 1, … ,200 and save 
the corresponding label as result. For example, L_j has a face like this:


