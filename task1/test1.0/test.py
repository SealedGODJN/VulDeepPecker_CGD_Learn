import sys
import os
import pandas
from network import Classifier

def main():
    filename = sys.argv[1]
    vector_filename = filename + "_gadget_vectors.pkl"
    df = pandas.read_pickle(vector_filename)
    classifier = Classifier(df,name=filename)
    classifier.train()
    classifier.test()

if __name__ == "__main__":
    main()