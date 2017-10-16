import pandas as pd


def eval_with_threshold(df, truth_col_name, pred_col_name,
                        confidence_col_name, confidence_threshold):
    """
    :param df: DataFrame with ground truth and predicted value in a column
    :param truth_col_name: string name of columng having ground truth
    :param pred_col_name: string name of columng having prediction
    :param confidence_col_name: string name of columng having confidence of the prediction
    :param confidence_threshold: threshold below which to return no answer
    :return: object of correct_classified_rate, misclassified_rate, not_confident_rate
    """
    count_subjects = df.shape[0]
    count_correct = df[(df[confidence_col_name]>=confidence_threshold) & (df[truth_col_name]==df[pred_col_name])].shape[0]
    count_misclassified = df[(df[confidence_col_name]>=confidence_threshold) & (df[truth_col_name]!=df[pred_col_name]) ].shape[0]
    count_not_confident =  count_subjects - count_correct - count_misclassified
    correct_classified_rate = count_correct / count_subjects
    misclassified_rate = count_misclassified / count_subjects
    not_confident_rate = count_not_confident / count_subjects
    return {"correct_classified_rate" : correct_classified_rate,
            "misclassified_rate" : misclassified_rate,
            "not_confident_rate" : not_confident_rate}

def run_thresholds(df, truth_col_name, pred_col_name,
                   confidence_col_name, confidence_thresholds):
    list_rates = []
    for threshold in confidence_thresholds:
        list_rates.append(eval_with_threshold(df, truth_col_name, pred_col_name,
                   confidence_col_name, threshold))
    return pd.DataFrame.from_records(list_rates)