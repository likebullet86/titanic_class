import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats

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
print (train["Sex"])
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

print (train.loc[:,["Sex","Embarked"]].head())

tab = pd.crosstab(train['Ttype'], train['Survived'])
print(tab)

dummy = tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
dummy = plt.xlabel('Ticket type')
dummy = plt.ylabel('Percentage')

train['Bad_ticket'] = train['Ttype'].isin(['3','4','5','6','7','8','A','L','W'])
test['Bad_ticket'] = test['Ttype'].isin(['3','4','5','6','7','8','A','L','W'])

plt.show()

dummy = pd.concat([train.drop('Survived',1),test])
dummy2 = dummy[dummy['Title'].isin(['Mr','Miss','Mrs','Master'])]
foo = dummy2['Age'].hist(by=dummy2['Title'], bins=np.arange(0,81,1))
plt.show()

train['Young'] = (train['Age']<=30) | (train['Title'].isin(['Master','Miss','Mlle','Mme']))
test['Young'] = (test['Age']<=30) | (test['Title'].isin(['Master','Miss','Mlle','Mme']))

#train['Young'] = (train['Age']<=30) | (train['Age'].isnull() & (train['Title'].isin(['Master','Miss','Mlle','Mme'])))
#test['Young'] = (test['Age']<=30) | (train['Age'].isnull() & (test['Title'].isin(['Master','Miss','Mlle','Mme'])))

dummy = plt.hist(np.log10(surv['Fare'].values + 1), color="orange", normed=True, bins=25)
dummy = plt.hist(np.log10(nosurv['Fare'].values + 1), histtype='step', color="blue", normed=True, bins=25)
plt.show()

train['Fare_cat'] = pd.DataFrame(np.floor(np.log10(train['Fare'] + 1))).astype('int')

ax = plt.subplots( figsize =( 12 , 10 ) )
foo = sns.heatmap(train.drop('PassengerId',axis=1).corr(), vmax=1.0, square=True, annot=True)
plt.show()