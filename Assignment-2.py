import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


np.random.seed(0)
n = 15
x = np.linspace(0,10,n) + np.random.randn(n)/5
y = np.sin(x)+x/6 + np.random.randn(n)/10


X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)

# You can use this function to help you visualize the dataset by
# plotting a scatterplot of the data points
# in the training and test sets.
def part1_scatter():
    import matplotlib.pyplot as plt
    %matplotlib notebook
    plt.figure()
    plt.scatter(X_train, y_train, label='training data')
    plt.scatter(X_test, y_test, label='test data')
    plt.legend(loc=4);
    
    
# NOTE: Uncomment the function below to visualize the data, but be sure 
# to **re-comment it before submitting this assignment to the autograder**.   
#part1_scatter()

def answer_one():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures

    # Your code here
    result = np.zeros((4,100))
    
    for i,degree in enumerate([1,3,6,9]):
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X_train.reshape(11,1))
        linreg = LinearRegression().fit(X_poly, y_train)
        y = linreg.predict(poly.fit_transform(np.linspace(0,10,100).reshape(100,1)))
        result[i,:] = y
    return result


# feel free to use the function plot_one() to replicate the figure 
# from the prompt once you have completed question one
def plot_one(degree_predictions):
    import matplotlib.pyplot as plt
    %matplotlib notebook
    plt.figure(figsize=(10,5))
    plt.plot(X_train, y_train, 'o', label='training data', markersize=10)
    plt.plot(X_test, y_test, 'o', label='test data', markersize=10)
    for i,degree in enumerate([1,3,6,9]):
        plt.plot(np.linspace(0,10,100), degree_predictions[i], alpha=0.8, lw=2, label='degree={}'.format(degree))
    plt.ylim(-1,2.5)
    plt.legend(loc=4)



def answer_two():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics.regression import r2_score
    
    r2_train = np.zeros(10)
    r2_test = np.zeros(10)
    
    for i in range(10):
        poly = PolynomialFeatures(degree=i)
        
        #Train and Score X_train
        X_poly = poly.fit_transform(X_train.reshape(11,1))
        linreg = LinearRegression().fit(X_poly, y_train)
        r2_train[i] = linreg.score(X_poly, y_train)

        #Test X_test
        X_test_poly = poly.fit_transform(X_test.reshape(4,1))
        r2_test[i] = linreg.score(X_test_poly, y_test)
        
    return (r2_train,r2_test)

def answer_three():
    
    r2_train, r2_test = answer_two()
    
    df = pd.DataFrame({'trainingr2':r2_train, 'testingr2':r2_test})
    
    df['diff'] = df['trainingr2'] - df['testingr2']
    
    df = df.sort(['diff'])
    good_gen = df.index[0]
    
    df = df.sort(['diff'], ascending = False)
    overfitting = df.index[0]
    
    df = df.sort(['trainingr2'])
    underfitting = df.index[0]
    
    return (underfitting,overfitting,good_gen)

def answer_four():
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Lasso, LinearRegression
    from sklearn.metrics.regression import r2_score

    # Your code here
    poly = PolynomialFeatures(degree=12)
    
    X_train_p = poly.fit_transform(X_train.reshape(11,1))
    X_test_p = poly.fit_transform(X_test.reshape(4,1))
    
    linreg = LinearRegression().fit(X_train_p, y_train)
    linreg_r2 = linreg.score(X_test_p, y_test)
    
    lasso = Lasso(alpha=0.01, max_iter=10000).fit(X_train_p, y_train)
    lasso_r2 = lasso.score(X_test_p, y_test)
    return linreg_r2,lasso_r2


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

mush_df = pd.read_csv('mushrooms.csv')
mush_df2 = pd.get_dummies(mush_df)

X_mush = mush_df2.iloc[:,2:]
y_mush = mush_df2.iloc[:,1]

# use the variables X_train2, y_train2 for Question 5
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_mush, y_mush, random_state=0)

# For performance reasons in Questions 6 and 7, we will create a smaller version of the
# entire mushroom dataset for use in those questions.  For simplicity we'll just re-use
# the 25% test split created above as the representative subset.
#
# Use the variables X_subset, y_subset for Questions 6 and 7.
X_subset = X_test2
y_subset = y_test2

def answer_five():
    from sklearn.tree import DecisionTreeClassifier

    # Your code here
    clf = DecisionTreeClassifier().fit(X_train2, y_train2)
    features = []

    for index, importance in enumerate(clf.feature_importances_):
        features.append([importance, X_train2.columns[index]])
        
    features.sort(reverse=True)
    feature_names = np.array(features)
    top = feature_names[:5,1]
    
    return top.tolist()


def answer_six():
    from sklearn.svm import SVC
    from sklearn.model_selection import validation_curve

    # Your code here
    svc = SVC(kernel='rbf', C=1, random_state=0)
    gamma = np.logspace(-4,1,6)
    train_scores, test_scores = validation_curve(svc, X_subset, y_subset, param_name='gamma',
                                                param_range=gamma, scoring='accuracy')
    
    return train_scores.mean(axis=1), test_scores.mean(axis=1)
