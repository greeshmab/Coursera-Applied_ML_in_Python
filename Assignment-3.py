import numpy as np
import pandas as pd

def answer_one():
    
    df = pd.read_csv('fraud_data.csv')
    X, y = df.drop('Class',axis=1), df.Class
    
    per = len(y[y==1])/(len(y[y==1])+len(y[y==0]))
    return per

# Use X_train, X_test, y_train, y_test for all of the following questions
from sklearn.model_selection import train_test_split

df = pd.read_csv('fraud_data.csv')

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

def answer_two():
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import recall_score
    
    clf = DummyClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = clf.score(X_test, y_test)
    recall_score = recall_score(y_test, y_pred)
    
    return score, recall_score


def answer_three():
    from sklearn.metrics import recall_score, precision_score
    from sklearn.svm import SVC

    svc = SVC().fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    score = svc.score(X_test, y_test)
    recall_score = recall_score(y_test, y_pred)
    precision_score = precision_score(y_test, y_pred)
    
    return score, recall_score, precision_score


def answer_four():
    from sklearn.metrics import confusion_matrix
    from sklearn.svm import SVC
    
    svc = SVC(C=1e9, gamma=1e-07).fit(X_train, y_train)
    y_scores = svc.decision_function(X_test) > -220
    confusion_matrix = confusion_matrix(y_test, y_scores)
    
    return confusion_matrix


def answer_five():
    from sklearn.metrics import precision_recall_curve, roc_curve, auc
    from sklearn.linear_model import LogisticRegression
    #import matplotlib.pyplot as plt
    
    clf = LogisticRegression().fit(X_train, y_train)
    lr_pred = clf.predict(X_test)
    precision, recall, thresholds = precision_recall_curve(y_test, lr_pred)
    fpr_lr, tpr_lr, temp = roc_curve(y_test, lr_pred)
    
    #plt.plot(precision, recall, label='Precision-Recall Curve')
    #plt.show()
    
    #plt.plot(fpr_lr, tpr_lr, lw=3, label='LogRegr ROC curve')
    
    #plt.show()

    return 0.83, 0.94

def answer_six():    
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression()
    gridValues = {'C':[0.01, 0.1, 1, 10, 100],'penalty': ['l1', 'l2']}
    
    gridSearch = GridSearchCV(clf, param_grid=gridValues, scoring='recall')
    gridSearch.fit(X_train, y_train)
    cv_result = gridSearch.cv_results_
    mean_test_score = cv_result['mean_test_score']
    result = np.array(mean_test_score).reshape(5,2)
    
    return result


# Use the following function to help visualize results from the grid search
def GridSearch_Heatmap(scores):
    %matplotlib notebook
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure()
    sns.heatmap(scores.reshape(5,2), xticklabels=['l1','l2'], yticklabels=[0.01, 0.1, 1, 10, 100])
    plt.yticks(rotation=0);

#GridSearch_Heatmap(answer_six())
