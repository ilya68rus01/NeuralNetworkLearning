from NewNeuralNet import *
import numpy as np
import pandas as pd


ann = NewNeuralNet()
(X_train, y_train) = ann.create_learning_data()
ann.model.compile("nadam", loss="log_cosh", metrics=['accuracy'])
history = ann.model.fit(X_train[:2831747], y_train[:2831747], epochs=1)
ann.model.save_weights("weight.h5")
