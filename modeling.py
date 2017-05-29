import pandas as pd
import numpy as np
from scipy import stats
import sklearn as sk
import itertools
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from statsmodels.graphics.mosaicplot import mosaic

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import svm
import xgboost as xgb

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

pd.set_option('display.width', 320)
# data load
train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")
combine = pd.concat([train.drop('Survived',1),test])

surv = train[train['Survived']==1]
nosurv = train[train['Survived']==0]

#print(train[train['Embarked'].isnull()])

train['Embarked'].iloc[61] = "C"
train['Embarked'].iloc[829] = "C"

#print(test[test['Fare'].isnull()])

test['Fare'].iloc[152] = combine['Fare'][combine['Pclass'] == 3].dropna().median()

train['Child'] = train['Age']<=10
train['Young'] = (train['Age']>=18) & (train['Age']<=40)
train['Young_m'] = (train['Age']>=18) & (train['Age']<=40) & (train['Sex']=="male")
train['Young_f'] = (train['Age']>=18) & (train['Age']<=40) & (train['Sex']=="female")
train['Cabin_known'] = train['Cabin'].isnull() == False
train['Age_known'] = train['Age'].isnull() == False
train['Family'] = train['SibSp'] + train['Parch']
train['Alone']  = (train['SibSp'] + train['Parch']) == 0
train['Large Family'] = (train['SibSp']>2) | (train['Parch']>3)
train['Deck'] = train['Cabin'].str[0]
train['Deck'] = train['Deck'].fillna(value='U')
train['Ttype'] = train['Ticket'].str[0]
train['Title'] = train['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

train['Shared_ticket'] = 0
for i in range(len(train)):
    if ( (len(train.groupby('Ticket').get_group(train['Ticket'].iloc[i]))) > 1 ):
        train['Shared_ticket'].iloc[i] = 1

test['Child'] = test['Age']<=10
test['Young'] = (test['Age']>=18) & (test['Age']<=40)
test['Young_m'] = (test['Age']>=18) & (test['Age']<=40) & (test['Sex']=="male")
test['Young_f'] = (test['Age']>=18) & (test['Age']<=40) & (test['Sex']=="female")
test['Cabin_known'] = test['Cabin'].isnull() == False
test['Age_known'] = test['Age'].isnull() == False
test['Family'] = test['SibSp'] + test['Parch']
test['Alone']  = (test['SibSp'] + test['Parch']) == 0
test['Large Family'] = (test['SibSp']>2) | (test['Parch']>3)
test['Deck'] = test['Cabin'].str[0]
test['Deck'] = test['Deck'].fillna(value='U')
test['Ttype'] = test['Ticket'].str[0]
test['Title'] = test['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

test['Shared_ticket'] = 0
for i in range(len(test)):
    if ( (len(test.groupby('Ticket').get_group(test['Ticket'].iloc[i]))) > 1 ):
        test['Shared_ticket'].iloc[i] = 1

train["Sex"] = train["Sex"].astype("category")
train["Sex"].cat.categories = [0,1]
train["Sex"] = train["Sex"].astype("int")
train["Embarked"] = train["Embarked"].astype("category")
train["Embarked"].cat.categories = [0,1,2]
train["Embarked"] = train["Embarked"].astype("int")
train["Deck"] = train["Deck"].astype("category")
train["Deck"].cat.categories = [0,1,2,3,4,5,6,7,8]
train["Deck"] = train["Deck"].astype("int")

test["Sex"] = test["Sex"].astype("category")
test["Sex"].cat.categories = [0,1]
test["Sex"] = test["Sex"].astype("int")
test["Embarked"] = test["Embarked"].astype("category")
test["Embarked"].cat.categories = [0,1,2]
test["Embarked"] = test["Embarked"].astype("int")
test["Deck"] = test["Deck"].astype("category")
test["Deck"].cat.categories = [0,1,2,3,4,5,6,7]
test["Deck"] = test["Deck"].astype("int")

train['Bad_ticket'] = train['Ttype'].isin(['3','4','5','6','7','8','A','L','W'])
test['Bad_ticket'] = test['Ttype'].isin(['3','4','5','6','7','8','A','L','W'])

train['Young'] = (train['Age']<=30) | (train['Title'].isin(['Master','Miss','Mlle','Mme']))
test['Young'] = (test['Age']<=30) | (test['Title'].isin(['Master','Miss','Mlle','Mme']))

train['Fare_cat'] = pd.DataFrame(np.floor(np.log10(train['Fare'] + 1))).astype('int')

training, testing = train_test_split(train, test_size=0.2, random_state=0)
print("Total sample size = %i; training sample size = %i, testing sample size = %i"\
     %(train.shape[0],training.shape[0],testing.shape[0]))

#cols = ['Sex','Pclass','Cabin_known','Large Family','Parch','SibSp','Young','Alone','Shared_ticket']
cols = ['Sex','Pclass','Cabin_known','Large Family','Shared_ticket','Young','Alone','Child']
tcols = np.append(['Survived'],cols)

df = training.loc[:,tcols].dropna()
X = df.loc[:,cols]
y = np.ravel(df.loc[:,['Survived']])

clf_log = LogisticRegression()
clf_log = clf_log.fit(X,y)
score_log = cross_val_score(clf_log, X, y, cv=5).mean()
print(score_log)

clf_knn = KNeighborsClassifier(
    n_neighbors=10,
    weights='distance'
    )
clf_knn = clf_knn.fit(X,y)
score_knn = cross_val_score(clf_knn, X, y, cv=5).mean()
print(score_knn)

clf_rf = RandomForestClassifier(
    n_estimators=1000, \
    max_depth=None, \
    min_samples_split=10 \
    #class_weight="balanced", \
    #min_weight_fraction_leaf=0.02 \
    )
clf_rf = clf_rf.fit(X,y)
score_rf = cross_val_score(clf_rf, X, y, cv=5).mean()
print(score_rf)
print (pd.DataFrame(list(zip(X.columns, np.transpose(clf_rf.feature_importances_))) \
            ).sort_values(1, ascending=False))

clf_xgb = xgb.XGBClassifier(
    max_depth=2,
    n_estimators=500,
    subsample=0.5,
    learning_rate=0.1
    )
clf_xgb.fit(X,y)
score_xgb = cross_val_score(clf_xgb, X, y, cv=5).mean()
print(score_xgb)

clf_xgb_grid = xgb.XGBClassifier(
    max_depth=2,
    n_estimators=500,
    subsample=0.5,
    learning_rate=0.1
    )
param_grid = {    'n_estimators': [250, 500, 1000],
        'max_depth': [2, 4, 6, 8],
        'subsample': [0.5, 0.7, 0.9, 1.0],
                  }
gs = GridSearchCV(estimator=clf_xgb_grid, param_grid=param_grid, scoring='accuracy', cv=5)
gs = gs.fit(X,y)
print(gs.best_score_)
print(gs.best_params_)

clf_xgb_grid = xgb.XGBClassifier(
    max_depth=2,
    n_estimators=250,
    subsample=0.7,
    learning_rate=0.1
    )
clf_xgb_grid.fit(X,y)
score_xgb = cross_val_score(clf_xgb_grid, X, y, cv=5).mean()
print(score_xgb)
print (pd.DataFrame(list(zip(X.columns, np.transpose(clf_xgb_grid.feature_importances_))) \
            ).sort_values(1, ascending=False))

clf_vote = VotingClassifier(
    estimators=[
        ('log', clf_log),
        ('knn', clf_knn),
        ('rf', clf_rf),
        ('xgb_grid', clf_xgb_grid)
        ],
    weights=[1,1,1,1],
    voting='hard')
clf_vote.fit(X,y)

scores = cross_val_score(clf_vote, X, y, cv=5, scoring='accuracy')
print("Voting: Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))


df2 = test.loc[:,cols].fillna(method='pad')
clf = clf_vote
clf_list = [clf_vote, clf_rf, clf_xgb_grid, clf_log, clf_knn]
clf_name = ['vote', 'rf', 'xgb_grid', 'clf_log', 'clf_knn']

for i in range(0,len(clf_list)):
    clf = clf_list[i]
    surv_pred = clf.predict(df2)

    submit = pd.DataFrame({'PassengerId': test.loc[:, 'PassengerId'],
                           'Survived': surv_pred.T})
    # 'Survived': stack_pred.T})
    submit_name = "submit" + '_' + clf_name[i] + ".csv"
    submit.to_csv(submit_name, index=False)
