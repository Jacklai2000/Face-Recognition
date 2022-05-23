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

![a_face](https://user-images.githubusercontent.com/101900124/169807563-47f1686c-ea49-47c8-8e81-669beceb27e4.png)

After subtracting the mean face, we have:

![face_subtract_mean](https://user-images.githubusercontent.com/101900124/169807589-0b30183a-52c9-48e1-a639-18e4688f8754.png)


2. PCA & Decision Tree and Random Forest

Based on the prior PCA extracted features, we construct two versions of decision tree to classify the input images
using on the raw data and PCA-processed data. We choose information gain 𝐺𝑎𝑖𝑛(𝑆, 𝐴) as our attribute selection
criteria and further prune the tree by altering the maximum depth. To begin with, we evaluate the difference of original 
entropy of training sample sets (S) and the relative entropy of each feature A, in which the entropy is the measure of
random variable’s uncertainty. 

A reduce in the level of entropy means that the data are grouped with similar feature so difference within a class 
decrease. In short, we opt for the attribute which has the highest information gain and repeat the process until no 
further information gained from splitting the node. To improve predictive accuracy by the reduction of overfitting, we 
then proceed to the pruning process. We increase the maximum length of the longest path of tree from 1 to 30 and 
observe the trend of classification error.

The accuracy of decision tree using raw data increased from 49.5% to 57% with max. length = 6 while that of PCA-processed 
data increased from 1.5% to 3% with max. length = 8. 

The decision tree before pruning:

![Df_pca](https://user-images.githubusercontent.com/101900124/169808704-4ead63f6-4d7c-49dd-b736-d3a143b9a899.png)

The decision tree after pruning:

![tree2 0](https://user-images.githubusercontent.com/101900124/169808726-ba701079-ee7c-41e8-ab71-61bd422f370f.png)


3. PCA & Random Forest

To further improve the model, we adapt random forest as our second approach. We use the 10 PCA extracted features 
to create 100 bootstrap samples with size 100 and create the classification trees based on the bootstrap samples. A 
maximum of 10 features are chosen in deciding the best split in each tree. Then we input the test image and each tree 
returns a predicted person. We calculate the majority vote of the results of decision trees, and it is the predicted person 
of the input image according to the random forest using PCA extracted features. The random forest using raw data is the same 
as the random forest using PCA-processed data except it is using raw data instead of the 10 PCA extracted features.


### Result & Discussion

For the PCA part, we have tried different numbers of eigenfaces for classification. The accuracy nearly reaches its 
peak with the first 10 eigenfaces used and more eigenfaces can just tiny increase the accuracy: 

<img width="204" alt="2021-12-01" src="https://user-images.githubusercontent.com/101900124/169810521-a1953396-c123-412c-b329-cd0d4840e909.png">



Therefore, we have decided to use 10 eigenfaces for the classification part and the accuracy for PCA and KNN is 90%. 
The accuracy of decision tree with the use of PCA and that of random forest with the use of PCA are very low, which are 
3% and 1% respectively. However, the accuracy of random forest using raw data is 91% and that of decision tree using raw
data is 57% which are higher than the those with the use of PCA.

One of the possible reasons of failures of using PCA on decision tree and random forest is that these classification 
methods are designed for high-dimensional data. Since PCA reduces the data dimensions, it greatly reduces the 
amount of information for high-dimensional classification and result in a poor accuracy. Besides, several side factors 
would also affect the accuracy of the classification. One of the side factors is the direction of faces such as a lateral 
face. For the training dataset, using different direction of face would lead to a higher difficulty for recognizing the 
front face test image.

### Comparison

We focus on the comparisons in KNN, decision tree and random forest. In terms of the variable for splitting, distance 
will be used for KNN. However, for the decision tree and random forest, principal component values will be used. 
This causes the difference in the method of the node assignment that KNN minimizes the distance between two 
transformed weight vectors, while decision tree and random forest focus on maximizing information gains. Finally, for
the robustness, KNN is sensitive to outliers since it is based on the distance criteria. Outliers usually have larger values
and hence the result will be affected. The nodes of the decision tree are not sensitive to outliers because the splitting is 
based on the majority vote. This is similar to the random forest and the multiple trees setting will further balance out 
the effect of the outlier.

### Conclusion

We have compared the performances of different methods for face recognition and found that KNN with PCA
generally gives a good classification rate while decision tree and random forest with PCA do not. We also investigated 
the major reasons for the poor performances of PCA with high-dimensional classifiers, decision tree and random forest
due to the low-dimensional data after PCA reduces the strength of these classifiers. Hence, we conclude that PCA is 
suitable when classifiers cannot cope with high-dimensional data.





