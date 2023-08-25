from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


def compute_scores(true_labels, predicted_labels):
    true_labels, predicted_labels = prepare_input(true_labels, predicted_labels)
    precision = format(precision_score(true_labels, predicted_labels), '.3f')
    recall = format(recall_score(true_labels, predicted_labels), '.3f')
    f1 = format(f1_score(true_labels, predicted_labels), '.3f')
    auc = calculate_auc(true_labels, predicted_labels)
    print(f"AUC: {auc} F1: {f1} Precision: {precision} Recall: {recall}")


def prepare_input(y_true, y_scores):
    # Ensure the correct data type

    y_true = y_true.astype(float)
    y_scores = y_scores.astype(float)
    y_true = np.where(y_true == 0, -1, y_true)
    y_scores = np.where(y_scores == 0, -1, y_scores)

    return y_true, y_scores


def calculate_auc(y_true, y_scores):
    y_true, y_scores = prepare_input(y_true, y_scores)
    return roc_auc_score(y_true, y_scores)


def calculate_pauc(y_true, y_scores, p=0.1):
    y_true, y_scores = prepare_input(y_true, y_scores)
    sorted_indices = np.argsort(y_scores)[::-1]
    sorted_scores = y_scores[sorted_indices]
    sorted_labels = y_true[sorted_indices]

    n_negative = np.sum(1 - sorted_labels)
    p_idx = int(p * n_negative)

    pauc_scores = sorted_scores[:p_idx]
    pauc_labels = sorted_labels[:p_idx]

    return roc_auc_score(pauc_labels, pauc_scores)


def calculate_harmonic_mean(auc_score, pauc_score):
    return 2 / (1 / auc_score + 1 / pauc_score)


def calculate_scores_for_machine_types(machine_types, y_true_dict, y_scores_dict, p):
    auc_scores = []
    pauc_scores = []
    harmonic_scores = []

    for machine_type in machine_types:
        y_true = y_true_dict[machine_type]
        y_scores = y_scores_dict[machine_type]

        auc = calculate_auc(y_true, y_scores)
        pauc = calculate_pauc(y_true, y_scores, p)
        harmonic = calculate_harmonic_mean(y_true, y_scores, p)

        auc_scores.append(auc)
        pauc_scores.append(pauc)
        harmonic_scores.append(harmonic)

    return auc_scores, pauc_scores, harmonic_scores

