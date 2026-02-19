import argparse
import csv
from statistics import median
from scipy.stats import mannwhitneyu


def load_metrics(path):
    data = {"precision": [], "recall": [], "f1_score": []}
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in data.keys():
                try:
                    data[key].append(float(row[key]))
                except (KeyError, ValueError):
                    pass
    return data


def main():
    parser = argparse.ArgumentParser(description="Two-tailed Mann–Whitney U test on metrics from two CSV files.")
    parser.add_argument("csv1", help="First CSV file (e.g., model A results)")
    parser.add_argument("csv2", help="Second CSV file (e.g., model B results)")
    args = parser.parse_args()

    data1 = load_metrics(args.csv1)
    data2 = load_metrics(args.csv2)

    print(f"Comparing {args.csv1} vs {args.csv2}\n")
    
    for metric in ["precision", "recall", "f1_score"]:
        x, y = data1[metric], data2[metric]
        if not x or not y:
            print(f"{metric:<10} insufficient data")
            continue

        stat, p = mannwhitneyu(x, y, alternative="two-sided")
        line = (
            f"Metric: {metric:<10}  "
            f"n1={len(x):<3}  "
            f"n2={len(y):<3}  "
            f"Median1={median(x):<8.4f}  "
            f"Median2={median(y):<8.4f}  "
            f"U={stat:<10.3f}  "
            f"p-value={p:<10.4g}"
        )
        print(line)


    print("\nInterpretation: p < 0.05 suggests a significant difference between distributions.")


if __name__ == "__main__":
    main()
