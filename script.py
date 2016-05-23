import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
#import matplotlib.pyplot as plt
#import pickle

def preprocess():
    """ 
     Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1 
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output: 
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args
    


    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_features + 1, 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    
    train_data=np.insert(train_data, 0, 1.0, axis=1)

#    print train_data.shape
#    print initialWeights.shape
    initialWeights=initialWeights.reshape(-1,1)
#    print initialWeights.shape

    theta=sigmoid(np.dot(initialWeights.T,train_data.T))
    thetaT=theta.T #50000X1
#    print thetaT[49000]
#    print thetaT.shape
    oneminusthetaT=1-thetaT
#    print oneminusthetaT.shape
    
    log_thetaT=np.log(thetaT)
    log_oneminusthetaT=np.log(oneminusthetaT)
#    print log2_thetaT.shape
#    print log2_oneminusthetaT.shape
    stage1=labeli*log_thetaT
    stage2=(1-labeli)*(log_oneminusthetaT)
    
    stage3=stage1+stage2
    #print stage1[0]
    #print stage2[0]
    #print stage3[0]
    #print stage3.shape
    sumstage1=sum(stage3)
    
    div=sumstage1[0]/n_data
    div=div*-1
#    print div
    error =div

    ##gradiant w
    stage6=thetaT-labeli
    #print stage6.shape
    #print train_data.shape
    stage7=stage6*train_data
    #print stage7.shape
    
    
    stage8=np.sum(stage7,0)
#    print stage8.shape
    stage9div=stage8/n_data
#    print stage9div.shape
    error_grad=stage9div

#    print thetaT[15900:16000]
#    print oneminusthetaT[15900:16000]
#    print theta.shape

    print error

    return error, error_grad

def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W 
     of Logistic Regression
     
     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight 
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D
         
     Output: 
         label: vector of size N x 1 representing the predicted label of 
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    data=np.insert(data, 0, 1.0, axis=1)
    #print data.shape
    #print W.shape
    
    beforemax=sigmoid(np.dot(data,W))
    #print beforemax.shape
    
    for i in range(data.shape[0]):
        label[i]=beforemax[i].argmax(axis=0)
    #print label[50:100]
    #print label[16000:16050]
    #
    
#    print stopsdds

    return label


def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector of size (D + 1) x 1
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
#s    print "yoyoa"
    n_class=10
    train_data, Y = args
    
    
    #print train_data.shape
    #print Y.shape
    #print params.shape
    
    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    #print params.shape
    #print params[0]
    #print params[1]
    #print params[2]        
    W_b = params.reshape((n_feature + 1, n_class))
#    print W_b.shape
    train_data=np.insert(train_data, 0, 1.0, axis=1)
    error = 0
    error_grad = np.zeros((n_feature + 1, n_class))
    
#    W_b 716X10
#train_data 50000,716
    theta=np.exp(np.dot(train_data,W_b))
    theta_sum=np.sum(theta,1)
    #print theta_sum.shape
    #print theta.shape

    

    for i in range(theta.shape[0]):
        theta[i]=theta[i]/theta_sum[i]
    

    stage1=theta-Y
#    print stage1.shape
    
    stage2=np.dot(train_data.T,stage1)
#    print stage2.shape
    
    
    error_grad=stage2.flatten()#flatten stage2
#    print error_grad.shape
    
    ##error
    
    stage6=np.log(theta)
    stage7=Y*stage6
    stage8=np.sum(np.sum(stage7))
#    print stage8
    error=-1.0*stage8

#    print error    
    #print error_grad[0]
    #print error_grad[1]
    #print error_grad[2]
    #print error_grad[3]
    #print error_grad[4]                
    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    error=error/n_data#rechange in the assignment
    
    error_grad=error_grad/n_data#rechange in the assignment
    print error

    return error, error_grad


def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))
    
    

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    data=np.insert(data, 0, 1.0, axis=1)
    #print data.shape
    #print W.shape
    
    beforemax=np.exp(np.dot(data,W))
    #print beforemax.shape
    
    for i in range(data.shape[0]):
        label[i]=beforemax[i].argmax(axis=0)
        
    return label


"""
Script for Logistic Regression
"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

#print train_data.shape
#print train_label.shape
#print validation_data.shape
#print test_data.shape



# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()

#print Y.shape


 
# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
#print W.shape

initialWeights = np.zeros((n_feature + 1, 1))
opts = {'maxiter': 100}
for i in range(n_class):
    #print "loop"
    labeli = Y[:, i].reshape(n_train, 1)
#    print labeli.shape


    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))

#with open('params.pickle', 'wb') as f1: 
#    pickle.dump(W, f1) 


# Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

#print rukkkkkkkkkkkksaja


#print ssttoopp
"""
Script for Support Vector Machine
"""

print('\n\n--------------SVM-------------------\n\n')
#
###################
## YOUR CODE HERE #
###################
#
#
from sklearn.svm import SVC
print "CASE 1"

print "step0"
clf = SVC(kernel='linear')
print "step1"
clf.fit(train_data,train_label.ravel())
print "step2"
prediction=clf.predict(train_data)
print "step3"

####training set acc

predicted_label_svm=prediction.reshape(-1,1)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_svm == train_label).astype(float))) + '%')

####validation set acc
prediction=clf.predict(validation_data)
predicted_label_svm=prediction.reshape(-1,1)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_svm == validation_label).astype(float))) + '%')

####testing set acc
prediction=clf.predict(test_data)
predicted_label_svm=prediction.reshape(-1,1)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_svm == test_label).astype(float))) + '%')

#print stopppp

print "CASE 22222222222222222222222222222222222"
print "step0"
clf = SVC(kernel='rbf',gamma=1)
print "step1"
clf.fit(train_data,train_label.ravel())
print "step2"
prediction=clf.predict(train_data)
print "step3"

####training set acc

predicted_label_svm=prediction.reshape(-1,1)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_svm == train_label).astype(float))) + '%')

####validation set acc
prediction=clf.predict(validation_data)
predicted_label_svm=prediction.reshape(-1,1)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_svm == validation_label).astype(float))) + '%')

####testing set acc
prediction=clf.predict(test_data)
predicted_label_svm=prediction.reshape(-1,1)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_svm == test_label).astype(float))) + '%')

print "CASE 33333333333333333333"

print "step0"
clf = SVC(kernel='rbf',gamma='auto')
print "step1"
clf.fit(train_data,train_label.ravel())
print "step2"
prediction=clf.predict(train_data)
print "step3"

####training set acc

predicted_label_svm=prediction.reshape(-1,1)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_svm == train_label).astype(float))) + '%')

####validation set acc
prediction=clf.predict(validation_data)
predicted_label_svm=prediction.reshape(-1,1)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_svm == validation_label).astype(float))) + '%')

####testing set acc
prediction=clf.predict(test_data)
predicted_label_svm=prediction.reshape(-1,1)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_svm == test_label).astype(float))) + '%')



print "CASE 444444444444444444444444"


#train_data=train_data[1:6000]
#train_label=train_label[1:6000]

#print train_data.shape
#print train_label.shape



C_range = [1,10,20,30,40,50,60,70,80,90,100]
Acc_train=[]
Acc_valid=[]
Acc_test=[]
####training set acc
for C in C_range:
    print "C is "+str(C)
    print "step0"
    clf = SVC(C=C,kernel='rbf',gamma='auto')
    print "step1"
    clf.fit(train_data,train_label.ravel())
    print "step2"
    prediction=clf.predict(train_data)
    print "step3"
    predicted_label_svm=prediction.reshape(-1,1)
    print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_svm == train_label).astype(float))) + '%')
    Acc_train.append(100 * np.mean((predicted_label_svm == train_label).astype(float)))
####validation set acc    
    prediction=clf.predict(validation_data)
    predicted_label_svm=prediction.reshape(-1,1)
    print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_svm == validation_label).astype(float))) + '%')
    Acc_valid.append(100 * np.mean((predicted_label_svm == validation_label).astype(float)))
####testing set acc
    prediction=clf.predict(test_data)
    predicted_label_svm=prediction.reshape(-1,1)
    print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_svm == test_label).astype(float))) + '%')
    Acc_test.append(100 * np.mean((predicted_label_svm == test_label).astype(float)))
#
#
##plt.gca().set_color_cycle(['red', 'blue', 'green'])
##
##plt.plot(C_range, Acc_train)
##plt.plot(C_range, Acc_valid)
##plt.plot(C_range, Acc_test)
##
##
##plt.legend(['Accuracy_Train', 'Accuracy_Validation', 'Accuracy_Test'], loc='upper left')
##
##plt.show()
#
##print rukore
#"""
#Script for Extra Credit Part
#"""
# FOR EXTRA CREDIT ONLY
W_b = np.zeros((n_feature + 1, n_class))
#print W_b.shape
initialWeights_b = np.zeros((n_feature + 1, n_class))
#print initialWeights_b.shape
opts_b = {'maxiter': 100}

args_b = (train_data, Y)
#print "here"
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

#with open('params_bonus.pickle', 'wb') as f2:
#    pickle.dump(W_b, f2)

# Find the accuracy on Training Dataset
predicted_label_b = mlrPredict(W_b, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label_b = mlrPredict(W_b, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label_b = mlrPredict(W_b, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')
