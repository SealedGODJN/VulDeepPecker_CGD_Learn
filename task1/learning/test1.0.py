import _pickle as pickle
import numpy as np

(X_test),(X_train) = pickle.load(open('cwe119_cgd_gadget_vectors.pkl', 'rb'))

X_show = X_test[:100000]

print(X_show, X_show.__class__)