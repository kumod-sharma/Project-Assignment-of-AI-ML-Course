import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # Visualization
import seaborn as sns #Visualization
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error , r2_score , mean_squared_error
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier



data = pd.read_csv('AreaTC.csv')
data = data.replace('\.+', np.NaN, regex=True)


df=pd.DataFrame(data) 
# df.info()

# df.notnull().sum()


# df.describe()

# df.isnull().sum()

# df['Telangana'].isnull()
# df['Telangana'].mean()

df['Year'] = df['Year'].replace(['2004-05','2005-06','2006-07'
                                 ,'2007-08','2008-09','2009-10',
                                 '2010-11','2011-12','2012-13',
                                 '2013-14','2014-15','2015-16',
                                 '2016-17','2017-18','2018-19'],
                                [2005,2006,2007,2008,2009,2010,
                                 2011,2012,2013,2014,2015,2016,
                                 2017,2018,2019])


df['Telangana'].fillna(df['Telangana'].mean(), inplace=True)


# plt.scatter(df['Year'], df['Bihar'], c ="black")

# lines = df.plot.line()
# df.plot.line(x='Bihar', y=['Andhra Pradesh','Arunachal Pradesh','Bihar','Chhattisgarh','Uttarakhand',  'West Bengal'])
# plt.xlabel("Year")
# plt.ylabel('Land in thousand Hectare')
# plt.title("Land sown in States")




# df.plot.line(x='Bihar', y=['Arunachal Pradesh'])
# plt.xlabel("Land in thousand Hectare of Bihar")
# plt.ylabel('Land in thousand Hectare of Arunacha Pradesh')







# Assam	Bihar	Chhattisgarh	NCT of Delhi	Goa	Gujarat	Haryana	Himachal Pradesh	Jammu & Kashmir	Jharkhand	Karnataka	Kerala	Madhya Pradesh	Maharashtra	Manipur	Meghalaya	Mizoram	Nagaland	Odisha	Puducherry	Punjab	Rajasthan	Sikkim	Tamil Nadu	Telangana	Tripura	Uttar Pradesh


# splitting train and test data in 80:20 ratio
x_train, x_test, y_train, y_test = train_test_split(df['Year'], df['Chhattisgarh'], test_size=0.20, random_state=0)

# x_train, x_test, y_train, y_test = train_test_split(df['Field of Study'], df['Discount on Fees'], test_size=0.30, random_state=0)


x_train = x_train.values.reshape(len(x_train), 1)
x_train.shape

x_test = x_test.values.reshape(len(x_test), 1)
x_test.shape

# model = LinearRegression()
# model.fit(x_train, y_train)
# y_pred = model.predict(x_test)
# model.score(x_test,y_test)
# RMSE = np.sqrt( mean_squared_error(y_test, y_pred))

# r2 = r2_score(y_test, y_pred)


# svc = SVC()
# svc.fit(x_train, y_train)
# Y_pred = svc.predict(x_test)
# acc_svc = round(svc.score(x_train, y_train) * 100, 2)
# acc_svc

# knn = KNeighborsClassifier(n_neighbors = 3)
# knn.fit(x_train, y_train)
# Y_pred = knn.predict(x_test)
# acc_knn = round(knn.score(x_train, y_train) * 100, 2)
# acc_knn

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
Y_pred = gaussian.predict(x_test)
acc_gaussian = round(gaussian.score(x_train, y_train) * 100, 2)
acc_gaussian

plt.scatter(x_train, y_train, color='blue')
plt.plot(x_train, gaussian.predict(x_train), color='r')
plt.xlabel("Year")
plt.ylabel('Land in thousand Hectare')
plt.title("Land sown in Chhattisgarh using Gaussian Naive Bayes")
plt.show()



