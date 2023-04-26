import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact, IntSlider
from sklearn.datasets import make_classification
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
X, y = make_classification(n_samples=100, n_features=20, n_informative=15,
                           n_redundant=2, n_repeated=0, n_classes=2,
                           class_sep=2.0, random_state=42)
# create a sequential forward feature selector
sfs = SequentialFeatureSelector(LogisticRegression(), direction='forward', n_features_to_select=10)
# create a sequential backward feature selector
sbs = SequentialFeatureSelector(LogisticRegression(), direction='backward', n_features_to_select=10)
# initialize empty lists to store the number of selected features and corresponding scores
n_features_fwd = []
scores_fwd = []
n_features_bwd = []
scores_bwd = []
# loop through a range of iterations to select a variable number of features
for i in range(1, 20):
    # create a new instance of the logistic regression model
    lr = LogisticRegression()
    sfs = SequentialFeatureSelector(LogisticRegression(), direction='forward', n_features_to_select=i) 
    # fit the sequential forward feature selector
    sfs.fit(X, y)
    # get the selected features and their corresponding scores
    selected_features_fwd = sfs.transform(X)
    lr.fit(selected_features_fwd, y)
    score_fwd = accuracy_score(y, lr.predict(selected_features_fwd))
    # append the number of selected features and the corresponding score to the lists
    n_features_fwd.append(i)
    scores_fwd.append(score_fwd)
    print(score_fwd)
    # create a new instance of the logistic regression model
    lr = LogisticRegression()
    sbs = SequentialFeatureSelector(LogisticRegression(), direction='backward', n_features_to_select=i)
    # fit the sequential backward feature selector
    sbs.fit(X, y)
    # get the selected features and their corresponding scores
    selected_features_bwd = sbs.transform(X)
    lr.fit(selected_features_bwd, y)
    score_bwd = accuracy_score(y, lr.predict(selected_features_bwd))
    # append the number of selected features and the corresponding score to the lists
    n_features_bwd.append(i)
    scores_bwd.append(score_bwd)
#     print(score_bwd)
    st.write(score_bwd)

# define a function to plot the scores
# def plot_scores(n):
#     plt.plot(n_features_fwd[:n], scores_fwd[:n], label='Forward')
#     plt.plot(n_features_bwd[:n], scores_bwd[:n], label='Backward')
#     plt.xlabel('Number of Features')
#     plt.ylabel('Accuracy')
#     plt.title('Sequential Feature Selection')
#     plt.legend()
#     st.pyplot()
def plot_scores(n):
    plt.plot(n_features_fwd[:n], scores_fwd[:n], label='Forward')
    plt.plot(n_features_bwd[:n], scores_bwd[:n], label='Backward')
    plt.xlabel('Number of Features')
    plt.ylabel('Accuracy')
    plt.title('Sequential Feature Selection')
    plt.legend()
    st.pyplot()
    fig, ax = plt.subplots()
    ax.plot(n_features_fwd[:n], scores_fwd[:n], label='Forward')
    ax.plot(n_features_bwd[:n], scores_bwd[:n], label='Backward')
    ax.set_xlabel('Number of Features')
    ax.set_ylabel('Accuracy')
    ax.set_title('Sequential Feature Selection')
    ax.legend()
    st.pyplot(fig)

# create a slider for the number of iterations
iterations_slider = IntSlider(min=1, max=len(n_features_fwd), value=len(n_features_fwd), description='Iterations:')
# use the interact function to link the slider to the plot
interact(plot_scores, n=iterations_slider)
st.write("hello world")
