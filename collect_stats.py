import pandas as pd
import glob

def get_accuracy(ground_truth_tag:str, prediction_tag:str, df:pd.DataFrame):
    positive = 0
    negative = 0
    for i in range(df.shape[0]):
        if df.iloc[i][ground_truth_tag] == df.iloc[i][prediction_tag]:
            positive += 1
        else:
            negative += 1
    return positive, negative, positive / (positive + negative)

def all_csvs():
    return glob.glob("*.csv")

def main():
    csvs = all_csvs()
    for csv in csvs:
        df = pd.read_csv(csv)
        print(csv)
        original = get_accuracy("ground_truth_output", "original_output", df)
        perturbed = get_accuracy("ground_truth_output", "perturbed_output", df)
        
        print("Original:\n\tPositive: {0}\n\tNegative: {1}\n\tAccuracy: {2}".format(original[0], original[1], original[2]))
        print("Purturbed:\n\tPositive: {0}\n\tNegative: {1}\n\tAccuracy: {2}".format(perturbed[0], perturbed[1], perturbed[2]))
        print("\n\n")


if __name__ == "__main__":
    main()