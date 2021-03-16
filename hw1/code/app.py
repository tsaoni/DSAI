import os
from train1 import Model

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--training', default='training_data.csv', help='input training data file name')
    parser.add_argument('--output', default='submission.csv', help='output file name')
    args = parser.parse_args()

    import pandas as pd
    cur_path = os.path.dirname(__file__)
    new_path = os.path.relpath("../dataset/" + args.training, cur_path)
    df_training = pd.read_csv(new_path)

    model = Model()
    model.train(df_training)
    df_result = model.predict(n_step=7)
#    df_result.to_csv = (args.output, index=0)
