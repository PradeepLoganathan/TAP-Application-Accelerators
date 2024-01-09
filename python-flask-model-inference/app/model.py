import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

# Sample data (X: input feature, y: target)
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Save the model to a file using pickle
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

filename
