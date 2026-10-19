import pandas as pd
from sklearn.metrics import classification_report


class CrossValidationReport:
    def __init__(self, results: list[dict], title: str = "MODEL PERFORMANCE REPORT"):
        self.results = results
        self.title = title

    def print_detailed_report(self) -> None:
        y_true = pd.concat([result["y_true"] for result in self.results])
        y_pred = pd.concat([result["y_pred"] for result in self.results])

        report_dict = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )
        df_report = pd.DataFrame(report_dict).transpose()
        df_report = df_report.rename(
            columns={
                "precision": "Precision",
                "recall": "Recall",
                "f1-score": "F1-Score",
                "support": "Support",
            }
        )

        print("\n" + "=" * 65)
        print(f" {self.title} ")
        print("=" * 65)

        display_df = df_report.copy()
        for column in ["Precision", "Recall", "F1-Score"]:
            if column in display_df.columns:
                display_df[column] = display_df[column].apply(
                    lambda value: f"{value * 100:.2f}%"
                )
        display_df["Support"] = display_df["Support"].astype(int)

        print(display_df.to_string())
        print("=" * 65 + "\n")
