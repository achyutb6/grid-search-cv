from __future__ import print_function

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

print(__doc__)

# Loading the Breast cancer dataset
cancer = datasets.load_breast_cancer()


# Preprocessing the dataset
X = cancer['data']
y = cancer['target']
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset in two equal parts into 80:20 ratio for train:test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# This is a key step where you define the parameters and their possible values
# that you would like to check.
tuned_parameters = [[{'kernel': ['rbf', 'linear'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]}],
                    [{'max_depth' : [10,100,1000,10000], 'min_samples_split' : [2,10,100], 'min_samples_leaf': [1,5,10], 'max_features' : ["sqrt","log2"]}],
                    [{'activation' : ['logistic','tanh','relu'],'hidden_layer_sizes' : [(5,),(10,)], 'max_iter' : [200,1000],'alpha' : [0.0001,0.0005]}],
                    [{}], #GaussianNB
                    [{'penalty': ['l1','l2'], 'tol' : [1e-4,1e-5], 'max_iter' : [10,100,1000], 'fit_intercept' : [True, False]}],
                    [{'n_neighbors' : [5,10,20],'weights': ['uniform','distance'], 'algorithm' : ['ball_tree', 'kd_tree', 'brute'],'p' : [1,2,3]}],
                    [{'n_estimators':[10,20,100],'max_samples':[0.5,1.0],'max_features':[0.5,1.0],'random_state':[None]}],
                    [{'n_estimators':[10,20,100],'max_features':[0.5,1.0],'criterion' : ['gini','entropy'],'max_depth': [None,100,200]}],
                    [{'n_estimators' : [50,100,200],'random_state' : [None], 'learning_rate' : [1.,0.8,0.5],'algorithm' : ['SAMME','SAMME.R']}],
                    [{'loss': ['deviance', 'exponential'],'n_estimators' : [100,200,500],'max_features' : ['log2','sqrt'],'max_depth' : [3,10,50] }],
                    [{'booster': ['gbtree', 'gblinear' ,'dart'], 'learning_rate' : [0.1,0.05,0.2], 'min_child_weight' : [1], 'max_delta_step' : [0]}]
                    ]
algorithms = [SVC(),DecisionTreeClassifier(),MLPClassifier(),GaussianNB(),LogisticRegression(),KNeighborsClassifier(),BaggingClassifier(),RandomForestClassifier(),AdaBoostClassifier(),GradientBoostingClassifier(),XGBClassifier()]
algorithm_names = ["SVC","DecisionTreeClassifier","MLPClassifier","GaussianNB","LogisticRegression","KNeighborsClassifier","BaggingClassifier","RandomForestClassifier","AdaBoostClassifier","GradientBoostingClassifier","XGBClassifier"]

for i in range(0, 11):
    print("################   %s   ################" %algorithm_names[i])
    #scores = ['precision_macro','recall_macro','accuracy']
    scores = ['precision_macro']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(algorithms[i], tuned_parameters[i], cv=5,
                           scoring='%s' % score)
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print("Detailed confusion matrix:")
        print(confusion_matrix(y_true, y_pred))
        print("Precision Score: \n")
        print(precision_score(y_true, y_pred))

        print()

