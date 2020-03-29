################################################################
import math
import time
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import metrics
from scipy.stats import mode
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns; sns.set(style="ticks", color_codes=True)
################################################################

df = pd.read_csv("data.csv") # reading the dataset from the csv file

#########################################
# variables for logistic-regression
sum_lr = 0
lr = []
score_lr = 0

"""
# variables for deep neural networks
sum_dnn = 0
dnn = []
score_dnn = 0
"""

# variables for support vector machines
sum_svm = 0
svm_ = []
score_svm = 0

"""
# variables for K-Nearest Neighbors
sum_knn = 0
knn = []
score_knn = 0
"""

"""
# variables for Random Forest Classifier
sum_rf = 0
rf = []
score_rf = 0
"""

# variables for the Ensemble Model
sum_total = 0
total = []
score_total = 0
#########################################


# adding a new-feature as age-group
df['AgeGroup'] = pd.Series(np.random.randn(df.shape[0]), index=df.index)

# rearranging the columns to make outcome the last column
columnsTitles=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age', 'AgeGroup', 'Outcome']
df=df.reindex(columns=columnsTitles)

# list of all the columns with features
feat_cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age', 'AgeGroup']


n = 100 # number of iterations over which the average is taken


for j in range(n):

    # splitting the data-set into training-set and test-set
    df_train_set, df_test_set, df_train_lbl, df_test_lbl = train_test_split(df[feat_cols[:-1]], df['Outcome'], test_size = 0.20, random_state = j)

    ########################################################################################################
    # performing data-clean up
    i = 0
    while i < df.shape[0]:
        if(df.at[i,  df.columns[0]] < 0): # pregnancy values which are negative are made positive
            df.at[i,  df.columns[0]] = -1*df.at[i,  df.columns[0]]
        if(df.at[i,  df.columns[2]] < 40): # blood pressure is lower-capped at 40
            df.at[i,  df.columns[2]] = 40
        if(df.at[i,  df.columns[0]] > 6): # uppercapping the pregnancy values to 6
            df.at[i,  df.columns[0]] = 6
        if( df.at[i,  df.columns[0]] ==  df.at[i,  df.columns[0]]): # making pregnancies as integer values
            df.at[i,  df.columns[0]] = math.floor(df.at[i,  df.columns[0]])
        if(df.at[i,  df.columns[3]] < 0): # Skin Thickness in negative is made positive
            df.at[i,  df.columns[3]] = -1*df.at[i,  df.columns[3]]
        if(df.at[i,  df.columns[3]] > 60): # Skin Thickness is upper capped at 60
            df.at[i,  df.columns[3]] = 60
        if(df.at[i,  df.columns[4]] > 300): # uppercapping the insulin values at 300
            df.at[i,  df.columns[4]] = 300
        if(df.at[i,  df.columns[5]] < 0): # BMI which is negative is made positive
            df.at[i,  df.columns[5]] = -1*df.at[i,  df.columns[5]]
        if(df.at[i,  df.columns[5]] < 13): # BMI which is lower-capped at 13
            df.at[i,  df.columns[5]] = 13
        if(df.at[i,  df.columns[5]] > 55): # BMI which is upper-capped at 55
            df.at[i,  df.columns[5]] = 55
        if(df.at[i,  df.columns[6]] > 1.5): # Uppercapping DBF to 1.5
            df.at[i,  df.columns[6]] = 1.5
        i = i + 1
    ########################################################################################################


    # dealing with missing-data
    mean_glucose = (df_train_set['Glucose'].mean(skipna = True)) # taking the mean value of the glucose from the training-set
    mean_BP = (df_train_set['BloodPressure'].mean(skipna = True)) # taking the mean value of the BloodPressure from the training-set
    median_skinThick = (df_train_set['SkinThickness'].median(skipna = True)) # taking the median value of the SkinThickness from the training-set
    median_BMI = (df_train_set['BMI'].median(skipna = True)) # taking the median value of the BMI from the training-set
    median_insulin = (df_train_set['Insulin'].median(skipna = True)) # taking the median value of the insulin from the training-set
    median_DPF = (df_train_set['DiabetesPedigreeFunction'].median(skipna = True)) # taking the median value of the DiabetesPedigreeFunction from the training-set
    mean_age = (df_train_set['Age'].mean(skipna = True)) # taking the mean value of the age from the training-set

    # augmenting the NaN values in the dataset of the corresponding features
    values = {'SkinThickness' : median_skinThick, 'Glucose': mean_glucose, 'Insulin': median_insulin, 'BMI': median_BMI, 'DiabetesPedigreeFunction' : median_DPF, 'BloodPressure' : mean_BP}
    df = df.fillna(value=values)

    # splitting the data-set according to the same seed
    df_train_set, df_test_set, df_train_lbl, df_test_lbl = train_test_split(df[feat_cols[:-1]], df['Outcome'], test_size = 0.20, random_state = j)

    # creating a copy of the data-frame with all the NaN rows dropped
    df_aux = df_train_set.dropna()

    # Using linear-regression to predict skin-thickness in the data where skin-thickness is given 0 on the basis of Insulin and BMI
    # Not used as it negatively affects the accuracy
    """
    blood_insulin_diab_bmi_skin_model = LinearRegression() #to get age
    relevant = df_aux.iloc[:,4:6]
    blood_insulin_diab_bmi_skin_model.fit(relevant, df_aux.iloc[:,3])
    i = 0
    while i < df.shape[0]:
        if(df.at[i,  df.columns[3]] == 0): # pregnancy values which are negative are made positive
            aux = [ [ df.at[i,  df.columns[4]] , df.at[i,  df.columns[6]]] ]
            df.at[i,  df.columns[3]] = blood_insulin_diab_bmi_skin_model.predict(np.array(aux))
        i = i + 1
    """

    # Using linear-regression to predict the age for NaN values on the basis of BloodPressure and Glucose
    blood_glucose_age_model = LinearRegression()
    blood_glucose_age_model.fit((df_aux.iloc[:,1:3].values), df_aux.iloc[:,7])
    i = 0
    while i < df.shape[0]:
        if(df.at[i,  df.columns[7]] != df.at[i,  df.columns[7]]):
            aux = [[df.at[i,  df.columns[1]], df.at[i,  df.columns[2]]]]
            df.at[i,  df.columns[7]] = blood_glucose_age_model.predict(np.array(aux))
        i = i + 1


    # Using linear-regression to predict the number of pregnancies for NaN values on the basis of age
    age_preg_model = LinearRegression()
    age_preg_model.fit(df_aux.iloc[:,7].values.reshape(-1,1), df_aux.iloc[:,0])
    i = 0
    while i < df.shape[0]:
        if(df.at[i,  df.columns[0]] != df.at[i,  df.columns[0]]):
            df.at[i,  df.columns[0]] = age_preg_model.predict(df.at[i,  df.columns[7]].reshape(-1,1))
        i = i + 1

    # Using the age feature in order to construct a new-feature - AgeGroup, which creates groups of 10
    i = 0
    while i < df.shape[0]:
        df.at[i,  df.columns[8]] = (df.at[i,  df.columns[7]] - df.at[i,  df.columns[7]]%10)/10
        i = i + 1

    # splitting the data on the basis of the same seed
    df_train_set, df_test_set, df_train_lbl, df_test_lbl = train_test_split(df[feat_cols], df['Outcome'], test_size = 0.20, random_state = j)

    #standardizing the data
    scaler = StandardScaler()

    #fitting only on the training set
    scaler.fit(df_train_set)

    #transforming on both, the training set and the test set
    df_train_set = scaler.transform(df_train_set)
    df_test_set = scaler.transform(df_test_set)

    #Applying PCA and gathering components that capture 95% of the variance.
    pca1 = PCA(0.95)

    #fitting only on the training set
    pca1.fit(df_train_set)

    #transforming on both, the training set and the test set
    df_train_set = pca1.transform(df_train_set)
    df_test_set = pca1.transform(df_test_set)

    """ Applying Logistic Regression """
    #########################################################
    logistic_reg = LogisticRegression(solver = 'lbfgs')
    logistic_reg.fit(df_train_set,df_train_lbl)
    y_pred_ = logistic_reg.predict(df_test_set)
    score_lr = (logistic_reg.score(df_test_set,df_test_lbl))
    sum_lr = sum_lr + score_lr
    lr.append(score_lr)
    #########################################################

    """ Applying Support Vector Machine with Linear Kernel """
    #########################################################
    clf_svm = svm.SVC(kernel='linear')
    clf_svm.fit(df_train_set, df_train_lbl.values.ravel())
    y_pred = clf_svm.predict(df_test_set)
    score_svm = metrics.accuracy_score(df_test_lbl, y_pred)
    svm_ = svm_ + [score_svm]
    #########################################################

    """
    Applying RandomForestClassifier
    #########################################################
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    clf.fit(df_train_set, df_train_lbl)
    y_pred__ = clf.predict(df_test_set)
    score_rf = metrics.accuracy_score(df_test_lbl, y_pred)
    rf = rf + [score_rf]
    #########################################################
    """

    """
    Applying K-Nearest Neighbor Classificatioin
    #########################################################
    from sklearn.neighbors import KNeighborsClassifier

    model = KNeighborsClassifier(n_neighbors=11)
    model.fit(df_train_set,df_train_lbl)
    y_pred__ = model.predict(df_test_set)
    from sklearn import metrics
    score_knn = metrics.accuracy_score(df_test_lbl, y_pred)
    knn = knn + [score_knn]
    #########################################################
    """

    """
    Applying 4 layered Deep Neural Network
    ##########################################################################################
    from keras import Sequential
    from keras.layers import Dense # for different layers in the network
    classifier = Sequential()
    classifier.add(Dense(4, kernel_initializer = 'random_normal'))
    from keras.layers import LeakyReLU
    classifier.add(LeakyReLU(alpha = 0.95))
    classifier.add(Dense(1, activation = 'sigmoid', kernel_initializer = 'random_normal'))
    classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])
    classifier.fit(df_train_set, df_train_lbl.values, batch_size=10, epochs=20)
    y_pred=classifier.predict(df_test_set)
    y_pred =(y_pred>0.5)
    arr = []
    for each in y_pred:
        if(each == [True]):
            arr.extend([1])
        else:
            arr.extend([0])
    y_pred = np.array(arr)
    ##########################################################################################
    """

    """ Combining the prediction of Logistic Regression and SVM by dot-product """
    #########################################################
    y_pred = np.multiply(y_pred_, y_pred)
    from sklearn import metrics
    score_total = metrics.accuracy_score(df_test_lbl, y_pred)
    total = total + [score_total]
    #########################################################


# Output for Logistic Regression
print("Logisitc Regression: ")
avg = (sum_lr/n)
print('Average: ', avg)
sdsum = 0
for i in range(len(lr)):
    sdsum+=(math.pow((lr[i] - avg),2))
std_dev = math.sqrt(sdsum/len(lr))
print('Standard deviation: ', std_dev)
print('Min: ', min(lr))
print('Max: ', max(lr))

print("\n\n")

# Output for Support Vector Machines
print("Support Vector Machines: ")
avg = (sum(svm_)/n)
print('Average: ', avg)
print('Standard deviation: ', (np.asarray(svm_)).std())
print('Min: ', (np.asarray(svm_)).min())
print('Max: ', (np.asarray(svm_)).max())

"""
# Output for Random Forest Classifier
print("\n\n")
print("Random Forest: ")
avg = (sum(rf)/n)
print('Average: ', avg)
print('Standard deviation: ', (np.asarray(rf)).std())
print('Min: ', (np.asarray(rf)).min())
print('Max: ', (np.asarray(rf)).max())
"""

"""
# Output for K-Nearest Neighbor Classifier
print("\n\n")
print("KNN Model: ")
avg = (sum(knn)/n)
print('Average: ', avg)
print('Standard deviation: ', (np.asarray(knn)).std())
print('Min: ', (np.asarray(knn)).min())
print('Max: ', (np.asarray(knn)).max())
"""


"""
# Output for the Deep-Neural Network
print("DNN Model: ")
avg = (sum(dnn)/n)
print('Average: ', avg)
print('Standard deviation: ', (np.asarray(dnn)).std())
print('Min: ', (np.asarray(dnn)).min())
print('Max: ', (np.asarray(dnn)).max())
"""

# Output for the Ensemble Model
print("\n\n")
print("Ensemble Model: ")
avg = (sum(total)/n)
print('Average: ', avg)
print('Standard deviation: ', (np.asarray(total)).std())
print('Min: ', (np.asarray(total)).min())
print('Max: ', (np.asarray(total)).max())
print("\n\n")
