from collections import defaultdict
from imblearn.datasets import fetch_datasets
from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from os_sklearn.ensemble._forest import OSRandomForestClassifier


class Comparator:
    def __init__(self, test_size=0.2):
        self.test_size = test_size
        self.metrics = ["precision", "recall", "f1", "support"]

    def compare(self, os_method, dataset_name, runs=10):
        dataset = fetch_datasets()[dataset_name]
        X = dataset.data
        y = dataset.target

        if os_method == "random":
            sampler = RandomOverSampler()
        elif os_method == "SMOTE":
            sampler = SMOTE()
        elif os_method == "BorderlineSMOTE":
            sampler = BorderlineSMOTE()
        elif os_method == "ADASYN":
            sampler = ADASYN()
        else:
            raise ValueError(f"Oversampling method {os_method} is not supported.")


        for i in range(runs):
            print(f"======= Run {i} =======")
            report_before, report_after = self.evaluate(X, y, sampler, os_method)
            # recall_before = report_before["-1"]["recall"]
            # recall_after = report_after["-1"]["recall"]

            # for metric in self.metrics:
            #     print(f"{metric} before: {report_before[metric]}")
            #     print(f"{metric} after: {report_after[metric]}")
                

    def evaluate(self, X, y, sampler, os_method):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=self.test_size
        )

        os_before_bagging = Pipeline([
            (os_method, sampler),
            ('rf', RandomForestClassifier())
        ])
        os_before_bagging.fit(X_train, y_train)
        y_pred_before = os_before_bagging.predict(X_test)

        os_after_bagging = OSRandomForestClassifier(oversampling_strategy=os_method)
        os_after_bagging.fit(X_train, y_train)
        y_pred_after = os_after_bagging.predict(X_test)

        report_before = classification_report(y_test, y_pred_before, output_dict=True, zero_division=0)
        report_after = classification_report(y_test, y_pred_after, output_dict=True, zero_division=0)

        print("OS before bagging")
        print(classification_report(y_test, y_pred_before))
        print("OS after bagging")
        print(classification_report(y_test, y_pred_after))

        return report_before, report_after


if __name__ == "__main__":
    comparator = Comparator()
    comparator.compare('BorderlineSMOTE', 'oil')
    