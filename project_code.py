#######################################################################
#code for face recognition
# pca+knn and pca+df and pca+rf



#######################################################################
#define a function to check wether a module is installed
#if not, install it

import importlib, os
import sys
def import_neccessary_modules(modname:str)->None:
    '''
        Import a Module,
        and if that fails, try to use the Command Window PIP.exe to install it,
        if that fails, because PIP in not in the Path,
        try find the location of PIP.exe and again attempt to install from the Command Window.
    '''
    try:
        # If Module it is already installed, try to Import it
        importlib.import_module(modname)
        print(f"Importing {modname}")
    except ImportError:
        # Error if Module is not installed Yet,  the '\033[93m' is just code to print in certain colors
        print(f"\033[93mSince you don't have the Python Module [{modname}] installed!")
        print("I will need to install it using Python's PIP.exe command.\033[0m")
        if os.system('PIP --version') == 0:
            # No error from running PIP in the Command Window, therefor PIP.exe is in the %PATH%
            os.system(f'PIP install {modname}')
        else:
            # Error, PIP.exe is NOT in the Path!! So I'll try to find it.
            pip_location_attempt_1 = sys.executable.replace("python.exe", "") + "pip.exe"
            pip_location_attempt_2 = sys.executable.replace("python.exe", "") + "scripts\pip.exe"
            if os.path.exists(pip_location_attempt_1):
                # The Attempt #1 File exists!!!
                os.system(pip_location_attempt_1 + " install " + modname)
            elif os.path.exists(pip_location_attempt_2):
                # The Attempt #2 File exists!!!
                os.system(pip_location_attempt_2 + " install " + modname)
            else:
                # Neither Attempts found the PIP.exe file, So i Fail...
                print(f"\033[91mAbort!!!  I can't find PIP.exe program!")
                print(f"You'll need to manually install the Module: {modname} in order for this program to work.")
                print(f"Find the PIP.exe file on your computer and in the CMD Command window...")
                print(f"   in that directory, type    PIP.exe install {modname}\033[0m")
                exit()
                


package_list = ['os', 'cv2', 'zipfile', 'pandas', 'numpy', 'matplotlib.pyplot', 'sklearn']

for eachpack in package_list:
    import_neccessary_modules(eachpack)

#######################################################################



# import packages
import cv2
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

#############################################################
#read in our raw dataset from zip files

faces = {}   #create a list to store our raw face data

print("This file full path (following symlinks)")
full_path = os.path.realpath("archive.zip")    #get your own path
print(full_path + "\n")      #show your current path

with zipfile.ZipFile(full_path) as facezip:    #import the dataset
    for filename in facezip.namelist():
        if not filename.endswith(".pgm"):
            continue # not a face picture
        with facezip.open(filename) as image:
            # If we extracted files from zip, we can use cv2.imread(filename) instead
            faces[filename] = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
            

#####################################################
#show the raw dataset as photos
            
#show all faces
fig, axes = plt.subplots(20,20,sharex=True,sharey=True,figsize=(8,10))
faceimages = list(faces.values()) #take all faces

for i in range(400):
    axes[i%20][i//20].imshow(faceimages[i], cmap="gray")  #display data as image
    
plt.show()

#show 16 faces in the repot as figure 1
fig, axes = plt.subplots(4,4,sharex=True,sharey=True,figsize=(8,10))
faceimages = list(faces.values())[-16:] # take last 16 images

for i in range(16):
    axes[i%4][i//4].imshow(faceimages[i], cmap="gray")  #display data as image
    
plt.show()

##################
faceshape = list(faces.values())[0].shape  #return dimension of the first face
print("Face image shape:", faceshape)

##################
classes = set(filename.split("/")[0] for filename in faces.keys())  #create a set with class only each class represent 1 person
print("Number of classes:", len(classes))
print("Number of pictures:", len(faces))


##################################################
#Divide the dataset into training dataset and testing dataset 
#Training dataset contains all people with 5 images each
#Testing dataset contains the remaining

facematrix_train = []   #create a list to store training faces
facelabel_train = []   #create a list to store training face labels
facematrix_test = []   #create a list to store testing faces
facelabel_test = []    ##create a list to store testing face labels

#test1 and test2 for classifiying the raw faces into training dataset and testing dataset 
test1 = ["/2","/3","/4","/5","/6"]
test2 = ["/1","/7","/8","/9"]

for key,val in faces.items():
    for each in test1:
        if each in key:
            facematrix_train.append(val.flatten())       #flatten a matrix s.t. it is 1 row 
            facelabel_train.append(key.split("/")[0])    #for making training label as s1,...,s40 in training face labels
    for each in test2:
        if each in key:
            facematrix_test.append(val.flatten())       #flatten a matrix s.t. it is 1 row 
            facelabel_test.append(key)                  #for saving testing label in testing face labels
            

# Create facematrix as (n_samples,n_pixels) matrix
facematrix_train = np.array(facematrix_train)   #our training dataset
facematrix_test = np.array(facematrix_test)    #our testing dataset


##################################################
#### Apply PCA to extract eigenfaces  ############
##################################################

pca = PCA().fit(facematrix_train)    #apply PCA to our training data

pca.var = pca.explained_variance_ratio_   #extarct the variance of each PCs
pca.vars = np.cumsum(pca.var)     #cumsum of the variance of PCs

#plot the pca variance
plt.plot(pca.var)
plt.xlabel("index of component")
plt.ylabel("variance")
plt.show()


# Compute the accuracy of using KNN by different number of eigenfaces used
n_components = [2,4,6,8,10,20,30,40,50,100,200]   #create a list to store different numbers of eigenfaces 
eigenfaces_li = []   #create a list to store eigenfaces
weights_li = []    #create a list to store weight matrix for different numbers of eigenfaces

for eachn in n_components:
    eigenfaces = pca.components_[:eachn]    #extract eigenfaces according to n_components
    eigenfaces_li.append(eigenfaces)     #save the eigenfaces
    # Generate weights as a KxN matrix where K is the number of eigenfaces and N the number of samples
    weights = eigenfaces @ (facematrix_train - pca.mean_).T
    weights_li.append(weights)   #save the weight matrix
   
    
# Show the first 16 eigenfaces as illustration
fig, axes = plt.subplots(4,4,sharex=True,sharey=True,figsize=(8,10))

for i in range(16):
    axes[i%4][i//4].imshow(eigenfaces[i].reshape(faceshape), cmap="gray")
    
plt.show()


##################################################
############### KNN Part #########################
##################################################

test_dic = dict(zip(list(facelabel_test),facematrix_test))  #create a dictionary to store out testing dataset and testing labels
best_match_lili = []    #create a list to store the matching result of different number of eigenfaces used

#compute the Euclidean distance of an testing image with each training image and find the minminimum
for i in range(len(n_components)):
    best_match_li = []   #create a list to store the matching result
    for eachface in test_dic:
        query = np.array(test_dic[eachface]).reshape(1,-1)   #flatten the tetsing matrix
        query_weight = eigenfaces_li[i] @ (query - pca.mean_).T   #compute the weight of the testing image
        euclidean_distance = np.linalg.norm(weights_li[i] - query_weight, axis=0)   #compute the Euclidean distance
        best_match = np.argmin(euclidean_distance)    #minimize the Euclidean distance
        best_match_li.append(facelabel_train[best_match])    #save the result
        print("Best match %s with Euclidean distance %f" % (facelabel_train[best_match], euclidean_distance[best_match]))
    best_match_lili.append(best_match_li)    #save the result
    
    

true_label = []    #create a list to save the testing labels of the testing dataset
correct_count = []   #create a list to save the number of correct matched labels

for eachlabel in facelabel_test:
    label = eachlabel.split("/")[0]   #for making testing label in the format as s1,...,s40 in testing face labels
    true_label.append(label)   

#compare the testing labels with the matched labels for computing the recognition rate
for i in range(len(best_match_lili)):
    count = 0
    for j in range(len(best_match_lili[i])):
        if best_match_lili[i][j] == true_label[j]:
            count +=1
    correct_count.append(count)

#recogntion rate
correct_rate = np.divide(correct_count,len(true_label))*100
print(correct_rate)

###
#output the result
d = {"Eigenfaces used":n_components, "recogntion rate (%)":correct_rate}
d = pd.DataFrame(d)
print(d)   
d.to_csv('PCA_RecognitionRate.csv', encoding='utf-8', index=False)

#output the number of eigenfaces we chose for building our classifier 
#We choose the first 10 eigenfaces

chosen_eigenfaces = pd.DataFrame(eigenfaces_li[4])
chosen_eigenfaces.to_csv('PCA_FeatureVectors.csv', encoding='utf-8', index=False)

chosen_weights = pd.DataFrame(weights_li[4])
chosen_weights.to_csv('PCA_weights.csv', encoding='utf-8', index=False)

###############################################################################
###############################################################################

##################################################
############### PCA & decision tree ##############
##################################################

n=10     #set the number of eigenfaces we chosen in the PCA part
pca = PCA(n_components=n)      #extract the first 10 eigenfaces 
X = pca.fit_transform(facematrix_train)    #reduce the training dataset dimension
X_test=pca.fit_transform(facematrix_test)    #reduce the testing dataset dimension

#build the decision tree by PCA
decision_tree = DecisionTreeClassifier()   #build the tree
fit0 = decision_tree.fit(X,facelabel_train)    #fit our reduced-dimension data into the decision tree
Ypred1 = decision_tree.predict(X_test)   #use the tree to predict the labels of the reduced-dimension testing dataset

#compute the accuracy
count = 0    #for counting the correct number of matches

for j in range(len(Ypred1)):
    if Ypred1[j] == true_label[j]:
        count +=1
r = count/len(Ypred1)   #the accuracy
print(r*100)

sklearn.tree.plot_tree(decision_tree)    #plot the tree
    

###############################################################################
###############################################################################

######################################################################
########## classification tree of raw data with pruning ###########
######################################################################

max_depth = []
acc_gini = []
acc_entropy = []
for i in range(1,30):
 dtree = DecisionTreeClassifier(criterion='gini', max_depth=i)
 dtree.fit(facematrix_train,facelabel_train)
 pred = dtree.predict(facematrix_test)
 acc_gini.append(accuracy_score(true_label, pred))
 ####
 dtree = DecisionTreeClassifier(criterion='entropy', max_depth=i)
 dtree.fit(facematrix_train,facelabel_train)
 pred = dtree.predict(facematrix_test)
 acc_entropy.append(accuracy_score(true_label, pred))
 ####
 max_depth.append(i)
 d = pd.DataFrame({'acc_gini':pd.Series(acc_gini),
 'acc_entropy':pd.Series(acc_entropy),
 'max_depth':pd.Series(max_depth)})# visualizing changes in parameters
plt.plot('max_depth','acc_gini', data=d, label='gini')
plt.plot('max_depth','acc_entropy', data=d, label='entropy')
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.legend()

#Best decision tree for raw data
dtree = DecisionTreeClassifier(criterion='entropy', max_depth=6) #build the best decision tree
fit1 = dtree.fit(facematrix_train,facelabel_train) #fit our training data into the decision tree
Ypred = dtree.predict(facematrix_test) #use the tree to predict the labels of the testing dataset

#compute the accuracy
count = 0    #for counting the correct number of matches

for j in range(len(Ypred)):
    if Ypred[j] == true_label[j]:
        count +=1
r = count/len(Ypred)    #the accuracy
print(r*100)

sklearn.tree.plot_tree(dtree)    #plot the tree


######################################################################
########## classification tree of PCA data with pruning ##############
######################################################################
max_depth = []
acc_gini = []
acc_entropy = []
for i in range(1,30):
 dtree = DecisionTreeClassifier(criterion='gini', max_depth=i)
 dtree.fit(X,facelabel_train)
 pred = dtree.predict(X_test)
 acc_gini.append(accuracy_score(true_label, pred))
 ####
 dtree = DecisionTreeClassifier(criterion='entropy', max_depth=i)
 dtree.fit(X,facelabel_train)
 pred = dtree.predict(X_test)
 acc_entropy.append(accuracy_score(true_label, pred))
 ####
 max_depth.append(i)
 d= pd.DataFrame({'acc_gini':pd.Series(acc_gini),
 'acc_entropy':pd.Series(acc_entropy),
 'max_depth':pd.Series(max_depth)})# visualizing changes in parameters
plt.plot('max_depth','acc_gini', data=d, label='gini')
plt.plot('max_depth','acc_entropy', data=d, label='entropy')
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.legend()

#Best decision tree for pca data
dtree2 = DecisionTreeClassifier(criterion='gini', max_depth=8)
fit2 = dtree2.fit(X,facelabel_train)
Ypred2 = dtree2.predict(X_test)

#compute the accuracy
count = 0    #for counting the correct number of matches

for j in range(len(Ypred2)):
    if Ypred2[j] == true_label[j]:
        count +=1
r = count/len(Ypred2)    #the accuracy
print(r*100)

sklearn.tree.plot_tree(dtree2)    #plot the tree


##################################################
########## random forest with pca data ###########
##################################################

#random forest with pca extracted features
model = RandomForestClassifier(n_estimators=100,max_samples=100,max_features=10)
model.fit(X, facelabel_train)
ypred = model.predict(X_test)
print(classification_report(ypred, true_label))   #print the result


##################################################
########## random forest with raw data ###########
##################################################

#y=label
#andom forest with raw data
model2 = RandomForestClassifier(n_estimators=100,max_samples=100,max_features=101)
model2.fit(facematrix_train, facelabel_train)
ypred2 = model2.predict(facematrix_test)
print(classification_report(ypred2, true_label))  #print the result



###############################################################################
###############################################################################
###############################################################################
###############################################################################
