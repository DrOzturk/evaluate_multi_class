import pandas as pd
import matplotlib.pyplot as plt

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
        results = eval_with_threshold(df, truth_col_name, pred_col_name,
                   confidence_col_name, threshold)
        results["threshold"]=threshold
        list_rates.append(results)
    return pd.DataFrame.from_records(list_rates)

# def chart_notconfident_vs_misclassified_highest_threshold(rates_on_range):
#     rates_on_range = rates_on_range.iloc[::-1]
#     fig, ax = plt.subplots()
#     ax.scatter(rates_on_range["not_confident_rate"],rates_on_range["misclassified_rate"])#,"ro")
#     prevx = 0
#     prevy = 0
#     for i, ratesi in rates_on_range.iterrows():
#         if (prevx != ratesi["not_confident_rate"] or prevy != ratesi["misclassified_rate"]):
#             ax.text(ratesi["not_confident_rate"],ratesi["misclassified_rate"], str(ratesi["threshold"])) #textcoords={"offset points":"3"})
#             prevx = ratesi["not_confident_rate"]
#             prevy = ratesi["misclassified_rate"]

def chart_notconfident_vs_misclassified(rates_on_range):
    fig, ax = plt.subplots()
    ax.scatter(rates_on_range["not_confident_rate"],rates_on_range["misclassified_rate"])#,"ro")
    prevx = 0
    prevy = 0
    for i, ratesi in rates_on_range.iterrows():
        if (prevx != ratesi["not_confident_rate"] or prevy != ratesi["misclassified_rate"]):
            ax.text(ratesi["not_confident_rate"]+0.005,ratesi["misclassified_rate"]+0.005, str(ratesi["threshold"])) #textcoords={"offset points":"3"})
            prevx = ratesi["not_confident_rate"]
            prevy = ratesi["misclassified_rate"]
