# age (numeric)
# job : type of job (categorical: “admin”, “blue-collar”, “entrepreneur”, “housemaid”, “management”, “retired”, “self-employed”, “services”, “student”, “technician”, “unemployed”, “unknown”)
# marital : marital status (categorical: “divorced”, “married”, “single”, “unknown”)
# education (categorical: “basic.4y”, “basic.6y”, “basic.9y”, “high.school”, “illiterate”, “professional.course”, “university.degree”, “unknown”)
# default: has credit in default? (categorical: “no”, “yes”, “unknown”)
# housing: has housing loan? (categorical: “no”, “yes”, “unknown”)
# loan: has personal loan? (categorical: “no”, “yes”, “unknown”)
# contact: contact communication type (categorical: “cellular”, “telephone”)
# month: last contact month of year (categorical: “jan”, “feb”, “mar”, …, “nov”, “dec”)
# day_of_week: last contact day of the week (categorical: “mon”, “tue”, “wed”, “thu”, “fri”)
# duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y=’no’). The duration is not known before a call is performed, also, after the end of the call, y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model
# campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
# pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
# previous: number of contacts performed before this campaign and for this client (numeric)
# poutcome: outcome of the previous marketing campaign (categorical: “failure”, “nonexistent”, “success”)
# emp.var.rate: employment variation rate — (numeric)
# cons.price.idx: consumer price index — (numeric)
# cons.conf.idx: consumer confidence index — (numeric)
# euribor3m: euribor 3 month rate — (numeric)
# nr.employed: number of employees — (numeric)
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('loan_dataset.csv', header=0)
data = data.dropna()
print(list(data.columns))

# CONSOLIDATE CATEGORIES
data['education'] = np.where(data['education'] == 'basic.9y', 'Basic', data['education'])
data['education'] = np.where(data['education'] == 'basic.6y', 'Basic', data['education'])
data['education'] = np.where(data['education'] == 'basic.4y', 'Basic', data['education'])
cat_vars = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week',
            'poutcome']
for var in cat_vars:
    cat_list = pd.get_dummies(data[var], prefix=var)
    # print(data[var])
    # print(cat_list)
    data1 = data.join(cat_list)
    data = data1
data_vars = data.columns.values.tolist()
to_keep = [i for i in data_vars if i not in cat_vars]
data_final = data[to_keep]

# LOGISTIC REGRESSION
X = data_final.loc[:, data_final.columns != 'y']
y = data_final.loc[:, data_final.columns == 'y']

# os = SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
cols = ['euribor3m', 'job_blue-collar', 'job_housemaid', 'marital_unknown', 'education_illiterate',
        'month_apr', 'month_aug', 'month_dec', 'month_jul', 'month_jun', 'month_mar',
        'month_may', 'month_nov', 'month_oct', "poutcome_failure", "poutcome_success"]
X = X_train[cols]
y = y_train['y']
logit_model = sm.Logit(y, X)
result = logit_model.fit()
print(result.summary2())

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
# y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

# K-FOLD CROSS VALIDATION
logreg2 = LogisticRegression()
scores = cross_val_score(logreg2, X, y, cv=5)
print(scores)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
# When the cv argument is an integer, cross_val_score uses the KFold or StratifiedKFold strategies by default, the latter being used if the estimator derives from ClassifierMixin.
