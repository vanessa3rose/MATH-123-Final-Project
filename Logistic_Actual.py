import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load the data
data = pd.read_excel('PROJ/code/S&P_Stock_data.xlsx')

# Obtain average volatility and avg monthly return for each stock
avg_data = data[['Average Monthly Return', 'Beta (Volatility)']]
avg_data['Average Monthly Return'] *= 100

# Assign labels based on Beta values
avg_data['Label'] = (avg_data['Beta (Volatility)'] >= 1).astype(int)

# Prepare the feature and target variables
X = avg_data[['Average Monthly Return']]
y = avg_data['Label']

for i in range(1, 4):
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=i)

    # Initialize and train logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predict and evaluate the model on the test data
    y_pred = model.predict(X_test)

    # Create a figure and a set of subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 6), sharey=True)

    # Determine colors based on the labels
    colors_train = np.where(y_train == 1, 'tab:blue', 'tab:red')
    colors_test = np.where(y_test == 1, 'tab:blue', 'tab:red')

    # Subplot 1: Training data with logistic regression line
    x_values = np.linspace(X_train.min(), X_train.max(), 300)
    y_probs = model.predict_proba(x_values.reshape(-1, 1))[:, 1]
    axs[0].plot(x_values, y_probs, color='darkkhaki', linewidth=2.5)
    axs[0].scatter(X_train, y_train, color=colors_train, alpha=0.75)
    axs[0].set_title('Training Data')
    axs[0].set_ylabel('Beta (Volatility)')
    axs[0].set_ylim(-0.2, 1.2)
    axs[0].set_yticks([0, 1])
    axs[0].set_yticklabels(['Low', 'High'])

    # Subplot 2: Actual labels
    axs[1].scatter(X_test, y_test, color=colors_test, alpha=0.75)
    axs[1].set_title('Actual Labels of Test Data')
    axs[1].set_xlabel('Average Monthly Return (%)')
    axs[1].set_ylim(-0.2, 1.2)
    axs[1].set_yticks([0, 1])
    axs[1].set_yticklabels(['Low', 'High'])

    # Subplot 3: Predictions
    axs[2].scatter(X_test, y_pred, color=colors_test, alpha=0.75)
    axs[2].set_title('Model Predictions of Test Data Labels')
    axs[2].set_ylim(-0.2, 1.2)
    axs[2].set_yticks([0, 1])
    axs[2].set_yticklabels(['Low', 'High'])

    # Set main title for all subplots with the iteration number
    fig.suptitle(f"LOGISTIC REGRESSION MODEL: ITERATION {i} - {model.score(X_test, y_test) * 100:.2f}% ACCURACY",
                 fontsize=15, y=0.975, color="darkkhaki", fontweight='bold', style='oblique')

    plt.tight_layout()
    plt.show()