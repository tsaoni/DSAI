from DDQN import Trader 
import pandas as pd
import os
import timeit


def load_data(trainfile, testfile):
    cur_path = os.path.dirname(__file__)
    new_path1 = os.path.relpath("../dataset/" + trainfile, cur_path)
    new_path2 = os.path.relpath("../dataset/" + testfile, cur_path)
    col = ["open", "high", "low", "close"]
    return pd.read_csv(new_path1, names = col), pd.read_csv(new_path2, names = col)

if __name__ == "__main__":
    import argparse
    start = timeit.default_timer()

    parser = argparse.ArgumentParser()
    parser.add_argument("--training", default = "training_data.csv", help = "input training data file name")
    parser.add_argument("--testing", default = "testing_data.csv", help = "input testing data file name")
    parser.add_argument("--output", default = "output.csv", help = "output file name")
    args = parser.parse_args()

    training_data, testing_data = load_data(args.training, args.testing)
    trader = Trader()
    trader.train(training_data, testing_data)

    cur_path = os.path.dirname(__file__)
    new_path = os.path.relpath("../output/" + args.output, cur_path)

    with open(new_path, "w") as output_file:
        for row in range(0, len(testing_data) - 1):
            action = trader.predict_action(testing_data.iloc[row, :])
            output_file.write(action)
            #trader.re_training(i)

    stop = timeit.default_timer()
    print("\n\n\n\nexecution time: ", stop - start)

