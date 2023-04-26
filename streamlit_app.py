import streamlit as st
from sklearn.datasets import make_classification
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# set page title
st.set_page_config(page_title="Sequential Feature Selection")

# define a function to generate the data and feature selectors
@st.cache(allow_output_mutation=True)
def generate_data_and_selectors():
    # generate the dataset
    X, y = make_classification(n_samples=100, n_features=20, n_informative=15,
                               n_redundant=2, n_repeated=0, n_classes=2,
                               class_sep=2.0, random_state=42)

    # create a sequential forward feature selector
    sfs = SequentialFeatureSelector(LogisticRegression(), direction='forward', n_features_to_select=10)

    # create a sequential backward feature selector
    sbs = SequentialFeatureSelector(LogisticRegression(), direction='backward', n_features_to_select=10)

    return X, y, sfs, sbs

# initialize empty lists to store the number of selected features and corresponding scores
n_features_fwd = []
scores_fwd = []
n_features_bwd = []
scores_bwd = []

# define the Streamlit app function
def app():
    # get the data and feature selectors
    X, y, sfs, sbs = generate_data_and_selectors()

    # create a slider for the number of iterations
    iterations_slider = st.slider(min_value=1, max_value=20, value=1, step=4, label='Iterations:')

    # loop through a range of iterations to select a variable number of features
    for i in range(1, iterations_slider):
        # create a new instance of the logistic regression model
        lr = LogisticRegression()

        # fit the sequential forward feature selector
        sfs.n_features_to_select = i
        sfs.fit(X, y)
        selected_features_fwd = sfs.transform(X)
        lr.fit(selected_features_fwd, y)
        score_fwd = accuracy_score(y, lr.predict(selected_features_fwd))
        # append the number of selected features and the corresponding score to the lists
        n_features_fwd.append(i)
        scores_fwd.append(score_fwd)

        # create a new instance of the logistic regression model
        lr = LogisticRegression()

        # fit the sequential backward feature selector
        sbs.n_features_to_select = i
        sbs.fit(X, y)
        selected_features_bwd = sbs.transform(X)
        lr.fit(selected_features_bwd, y)
        score_bwd = accuracy_score(y, lr.predict(selected_features_bwd))
        # append the number of selected features and the corresponding score to the lists
        n_features_bwd.append(i)
        scores_bwd.append(score_bwd)

    # define a function to plot the scores
    def plot_scores(n):
        # create a figure and axis object
        fig, ax = plt.subplots()
        # plot the scores for the forward feature selection
        ax.plot(n_features_fwd[:n], scores_fwd[:n], label='Forward Selection')
        # plot the scores for the backward feature selection
        ax.plot(n_features_bwd[:n], scores_bwd[:n], label='Backward Elimination')
        # set the plot title and labels
        ax.set_title('Sequential Feature Selection')
        ax.set_xlabel('Number of Features')
        ax.set_ylabel('Accuracy')
        # show the legend
        ax.legend()
        # display the plot
        st.pyplot(fig)



    # display the plot based on the value of the slider
    plot_scores(iterations_slider)
    
# run the app
if __name__ == '__main__':
    app()
