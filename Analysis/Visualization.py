import pandas as pd
import matplotlib.pyplot as plt
import os

# File path to the Excel file
file_path = "5930final/analysis/evaluated_responses_analysis.xlsx"

# Load the Raw Data sheet
df = pd.read_excel(file_path, sheet_name="Raw Data")

# Get the directory of the file
output_dir = os.path.dirname(file_path)

# Function to create bar charts for average scores
def create_bar_chart(df, output_dir):
    categories = df["Category"].unique()
    metrics = ["Base Relevance", "Base Coherence", "Base Adequacy",
               "Instruct Relevance", "Instruct Coherence", "Instruct Adequacy"]

    avg_scores = {metric: [] for metric in metrics}
    for category in categories:
        cat_data = df[df["Category"] == category]
        for metric in metrics:
            avg_scores[metric].append(cat_data[metric].mean())

    avg_df = pd.DataFrame(avg_scores, index=categories)
    avg_df.plot(kind="bar", figsize=(10, 6))
    plt.title("Average Scores by Category and Metric")
    plt.ylabel("Average Score")
    plt.xlabel("Category")
    plt.legend(title="Metrics", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    output_path = os.path.join(output_dir, "bar_chart.png")
    plt.savefig(output_path)
    plt.show()

# Function to create a pie chart for win distribution
def create_pie_chart(df, output_dir):
    win_counts = df["Winner"].value_counts()
    win_counts.plot(kind="pie", autopct="%1.1f%%", startangle=90, figsize=(6, 6))
    plt.title("Distribution of Wins")
    plt.ylabel("")  # Remove default y-label
    plt.tight_layout()
    output_path = os.path.join(output_dir, "pie_chart.png")
    plt.savefig(output_path)
    plt.show()

# Main function to generate all visualizations
def main():
    create_bar_chart(df, output_dir)
    create_pie_chart(df, output_dir)

if __name__ == "__main__":
    main()
