import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")

def map_data(shopping_row):
    row_evidence = list(shopping_row[0:17])
    row_labels = 1 if shopping_row[17].upper() == 'TRUE' else 0

    normalized_row_evidence = []
    index_single_decimal = [1, 3, 5, 6, 7, 8, 9]
    index_month = [10]
    index_visitor_type = [15]
    index_weekend = [16]
    for index, cell in enumerate(row_evidence):
        # round to 2 decimal for floats
        if index in index_single_decimal:
            normalized_row_evidence.append(round(float(cell), 2))
            continue

        # convert month to int
        if index in index_month:
            date_format = "%b"
            m = datetime.strptime(cell[:3], date_format).month
            normalized_row_evidence.append(m - 1)
            continue

        # visitor vs non-visitor to int
        if index in index_visitor_type:
            normalized_row_evidence.append(1 if cell == 'Returning_Visitor' else 0)
            continue

        # weekend visit to int
        if index in index_weekend:
            normalized_row_evidence.append(1 if cell.upper() == 'TRUE' else 0)
            continue

        # all else should be ints?
        normalized_row_evidence.append(int(cell))

    # print(f"Evidence: {normalized_row_evidence} labels: {row_labels}")
    return normalized_row_evidence, row_labels

def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    normalized_evidence = list()
    normalized_labels = list()
    with open(filename) as f:
        reader = csv.reader(f)
        next(reader)
        for index, row in enumerate(reader):
            try:
                row_evidence, row_labels = map_data(row)
                # print(f"Success reading row {index}: {row}")
            except ValueError:
                print(f"Error reading row {index}: {row}")
                continue
            normalized_evidence.append(row_evidence)
            normalized_labels.append(row_labels)
    # print(f"evidence {normalized_evidence}, labels:{normalized_labels}")
    return normalized_evidence, normalized_labels




def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """

    k = KNeighborsClassifier(n_neighbors=1)
    return k.fit(evidence, labels)


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    true_positive = true_negative = false_positive = false_negative = 0
    total = 0
    for actual, predicted  in zip(labels, predictions):
        total += 1
        if actual == 1:
            if predicted == 1: true_positive += 1
            else: false_negative += 1
        else:
            if predicted == 0: true_negative += 1
            else: false_positive += 1


    sensitivity = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0
    specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) else 0

    return (sensitivity, specificity)



if __name__ == "__main__":
    main()
