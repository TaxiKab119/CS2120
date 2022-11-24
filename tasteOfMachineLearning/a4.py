import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import plot_confusion_matrix



df = pd.read_csv("diabetes_data.csv")


def make_pretty(title):
    """
    Just makes print outputs a little more organized
    :param title: string, whatever you want to title outputs
    :return:Adds two spaces and a dashed line to output
    """
    print('')
    print('')
    print(title)
    print('-------------------------------------------')

def categorical_to_nominal(categorical_column,new_name):
    """
    The function takes a column categorical_column from the csv file
    and turns it into a nominal column with the name new_name using OrdinalEncoder()

    :param categorical_column: the chosen column from the csv file with categorical data
    :param new_name: the name for the new nominal column created from the categorical_column data
    :return: A new column with nominal data named 'new_name'
    """

    #make categorical data ordinal
    ord_enc = OrdinalEncoder()
    df[new_name] = ord_enc.fit_transform(df[[categorical_column]])
    #print(df[new_name])
    return new_name



def nominal_column_to_list(column):
    """
    This function takes a column of nominal data from the dataframe and
    turns it into a list.

    :param column: the chosen column from the dataframe
    :return: a list with the data from column
    """
    # turn new ordinal column into a list
    list_name = df[column].tolist()
    #print(list_name)
    return list_name


#This code is consolidating all the data into a table called ML_table
#ML_table will be fed into prepData when making splitting the data into training and testing sets (i.e. preparing it for machine learning)

age = nominal_column_to_list('Age')

categorical_to_nominal('Obesity', 'Nom_Obesity')
obesity = nominal_column_to_list('Nom_Obesity')

categorical_to_nominal('Gender', 'Nom_Gender')
sex = nominal_column_to_list('Nom_Gender')

categorical_to_nominal('Polyuria', 'Nom_Polyuria')
urine = nominal_column_to_list('Nom_Polyuria')

categorical_to_nominal('weakness', 'Nom_Weakness')
weakness = nominal_column_to_list('Nom_Weakness')


ML_table = pd.concat([df.iloc[:, 0], df.iloc[0:, 17:25], df['class']], axis='columns')
#print(ML_table)
#print(ML_table['class'].value_counts())




def prepData(df,Xstart,Xend,Y,test_size):
    """
    This function prepares data for testing and training from a table df
    it divides the table into attribute data based on Xstart to Xend
    it identifies the class column according to Y
    it divides the rows to have 1-test_size used for training and test_size used for testing

    it returns:
    prepData.X_train -> attribute training data
    prepData.X_test -> attribute test data
    prepData.y_train -> class training data
    prepData.y_test - class test data

    :param df: The data frame (in a table form) that you want to use
    :param Xstart: index of what column to start at for attribute data
    :param Xend: index of what column to end at for attribute data
    :param Y: index of class column
    :param test_size: float value indicating how much of dataset should be used as test data (in this case 0.2)
    :return: prepData.X_train, prepData.X_test, prepData.y_train, prepData.y_test
    """
    #preparing data to be split out into training and testing sets.

    # feature data in a matrix 'X'
    X = df[df.columns[Xstart:Xend]].values
    #print(X)
    #print(X.shape)
    y = df[df.columns[Y]].values #if [5] = nominal data, if [6] = categorical
    #print(y)
    #print(y.shape)


    #create X_train, X_test, y_train, y_test with the function model_selection.train_test_split and set test size to be 20% of the data

    prepData.X_train, prepData.X_test, prepData.y_train, prepData.y_test = model_selection.train_test_split(X,y,test_size = test_size) # this makes 20% of data for testing
    return (prepData.X_train, prepData.X_test, prepData.y_train, prepData.y_test)



def make_confusion_matrix(classifier,X_test,y_test,class_names):
    """
    this function outputs a confusion matrix with and without normalization in the python terminal
    it also visualizes these matrices
    :param classifier: the chosen classifier, gives the predicted values by the classifier
    :param X_test: attribute data in the test set
    :param y_test: true class labels in the test set
    :param class_names: list of the class names corresponding to the test data
    :return:
    """
    # Plot non-normalized confusion matrix
    titles_options = [("Confusion matrix, without normalization", None),
                      ("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(classifier, X_test, y_test,
                                     display_labels=class_names,
                                     cmap=plt.cm.BuGn,
                                     normalize=normalize)
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)


    plt.show()



prepData(ML_table,0,5,5,0.2)


#Using K-Nearest Neighbors Classifier (#1)

make_pretty('K-Nearest Neighbors Classifier')

knn_clf = KNeighborsClassifier()
knn_clf.fit(prepData.X_train, prepData.y_train)
knn_predictions = knn_clf.predict(prepData.X_test)
#print(knn_predictions)
print('Accuracy of the K-Nearest Neighbors Classifier is {}'.format(accuracy_score(prepData.y_test,knn_predictions)))

#Better representation of true accuracy of KNN using k fold Cross Validation (10 fold in this case)
multiple_measures_knn = cross_val_score(knn_clf,prepData.X_train, prepData.y_train, cv = 10).mean()
print('Accuracy of KNN using 10 Fold Cross Validation:', multiple_measures_knn)
print('')
make_confusion_matrix(knn_clf,prepData.X_test,prepData.y_test,['Negative','Positive'])



#Using Decision Tree Classifier (#2)

make_pretty('Decision Tree Classifier')

dt_clf = DecisionTreeClassifier()
dt_clf.fit(prepData.X_train, prepData.y_train)
dt_predictions = dt_clf.predict(prepData.X_test)
#print(dt_predictions)
print('Accuracy of the Decision Tree Classifier is {}'.format(accuracy_score(prepData.y_test,dt_predictions)))

#Better representation of true accuracy of DT using k fold Cross Validation (10 fold in this case)
multiple_measures_dt = cross_val_score(dt_clf,prepData.X_train, prepData.y_train, cv = 10).mean()
print('Accuracy of DT using 10 Fold Cross Validation:', multiple_measures_dt)

print('')
make_confusion_matrix(dt_clf,prepData.X_test,prepData.y_test,['Negative','Positive'])





