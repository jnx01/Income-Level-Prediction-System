import math
import pandas as pd
import numpy as np
from tkinter import *
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LassoCV

# data cleaning and preprocessing
def clean_preprocess_data(data):
    for i in data.columns:
        data[i] = data[i].replace('?', np.nan)
    for col_name in data.columns:
        series = data[col_name]
        if (type(series[1]) != 'int' and type(series[1]) != 'float'):
            label_encoder = LabelEncoder()
            data[col_name] = pd.Series(
                label_encoder.fit_transform(series[series.notnull()]),
                index=series[series.notnull()].index
            )
    for i in data.columns:
        data[i] = data[i].interpolate(method='linear')
    st = StandardScaler()
    data = st.fit_transform(data)
    data = pd.DataFrame(data)
    return data

# automated feature selection
def select_features(data, data_copy):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    clf = LassoCV().fit(X, y)
    cols = math.ceil((len(feature_names) - 1) / 2)
    importance = np.abs(clf.coef_)
    idx_third = importance.argsort()[-3]
    threshold = importance[idx_third] + 0.01
    idx_features = (-importance).argsort()[:cols]
    to_keep = np.array(feature_names)[idx_features]
    to_keep = list(to_keep)
    check = 'income' in to_keep
    if check == False:
        to_keep.append('income')
    to_drop = [item for item in feature_names if item not in to_keep]
    data = data.drop(to_drop, axis=1)
    data_copy = data_copy.drop(to_drop, axis=1)
    return data, data_copy

def click():
    global cols_to_keep
    a = int(age.get())
    b = int(YearOfEdu.get())
    c = mstatus.get()
    d = sex.get()
    e = float(CG.get())
    f = float(CL.get())
    g = int(hours.get())
    user_input = {
        'age': a,
        'education_num': b,
        'marital_status': c,
        'sex': d,
        'capital_gain': e,
        'capital_loss': f,
        'hours_per_week': g
    }
    user_input = pd.DataFrame([user_input])
    data = data_copy.copy()
    input = pd.concat([data, user_input], ignore_index=True, axis=0)
    input = clean_preprocess_data(input)
    input.columns = cols_to_keep
    user_input_row = input.iloc[-1, :]
    obj = pd.DataFrame([user_input_row])
    res = knn.predict(obj)
    output.delete('1.0', END)
    if res <= 0:
        output.insert(END, 'Class prediction: <= $50,000')
    else:
        output.insert(END, 'Class prediction: > $50,000')

# main
global data
global data_copy
global cols_to_keep
global knn

data = pd.read_csv(r'C:\Users\Naeem\Desktop\Jahanzeb\AI\project\adult.csv')
data = pd.DataFrame(data)
feature_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
                 'marital_status', 'occupation', 'relationship', 'race',
                 'sex', 'capital_gain', 'capital_loss', 'hours_per_week',
                 'native_country', 'income']
data.columns = feature_names
data_copy = data.copy()
data_copy.columns = feature_names
data = clean_preprocess_data(data)
data.columns = feature_names
data, data_copy = select_features(data, data_copy)
data_copy = data_copy.drop('income', axis=1)
cols_to_keep = list(data_copy.columns)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)
knn = KNeighborsRegressor(n_neighbors=25)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print('Mean Absolute Error: ', mean_absolute_error(y_test, y_pred))

# GUI code
root = Tk()
root.geometry("800x600")
root.configure(background="white", border=4, borderwidth=4)

l = Label(root, text="Enter Age:", fg='black', bg='white')
l.pack()
age = Entry(root, width=15, fg='black', bg='white')
age.pack()
space = Label(root, text="", fg='white', bg='white')
space.pack()

# Rest of the GUI elements follow...

root.mainloop()
