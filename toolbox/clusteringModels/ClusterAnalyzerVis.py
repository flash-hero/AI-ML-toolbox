import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, chi2_contingency

class ClusterAnalyzer:
    """
    This class analyzes the results of a clustering algorithm.
    It helps us understand 'who' is in each cluster by looking at their features.
    """
    
    def __init__(self, data, labels):
        """
        Initializes the analyzer.
        
        Parameters:
        - data: The original dataset (before encoding), so we can read the actual values.
        - labels: The cluster IDs assigned to each row by the model.
        """
        self.data = data.reset_index(drop=True).copy()

        # Add the cluster labels to the dataframe
        labels_series = pd.Series(labels, index=self.data.index, name='Cluster')

        # If 'Cluster' column already exists, remove it to avoid duplicates
        if 'Cluster' in self.data.columns:
            self.data = self.data.drop(columns='Cluster')

        self.data['Cluster'] = labels_series

        # Count how many unique clusters we have
        self.n_clusters = len(np.unique(labels))


    def describe_clusters(self):
        """
        Calculates statistics (mean, min, max, count) for each cluster.
        This gives a quick numerical summary of each group.
        """
        print("\n▶ Cluster Descriptive Statistics")
        try:
            # Group data by 'Cluster' and calculate stats
            print(self.data.groupby('Cluster').describe().T)
        except Exception as e:
            print(f"Error describing clusters: {e}")

    def plot_numerical_distributions(self, plot_type='box'):
        """
        Plots the distribution of numeric variables for each cluster.
        
        Parameters:
        - plot_type: 'box' for Boxplot, 'violin' for Violinplot. Defaults to 'box'.
        """
        numerical_cols = [col for col in self.data.select_dtypes(include=['int64', 'float64']).columns if col != 'Cluster']
        if not numerical_cols:
            print("No numerical variables to plot.")
            return

        print(f"\n▶ Generating {plot_type} plots for numerical variables...")

        n_cols = 2
        n_rows = int(np.ceil(len(numerical_cols) / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
        # Ensure axes is always a list/array we can index
        axes = axes.flatten() if len(numerical_cols) > 1 else [axes]

        for i, col in enumerate(numerical_cols):
            ax = axes[i]
            if plot_type == "box":
                sns.boxplot(x='Cluster', y=col, data=self.data, ax=ax, width=0.4, fliersize=3)
            elif plot_type == "violin":
                sns.violinplot(x='Cluster', y=col, data=self.data, ax=ax, inner="box", alpha=0.6)
            else:
                 # Default to boxplot
                 sns.boxplot(x='Cluster', y=col, data=self.data, ax=ax, width=0.4, fliersize=3)

            ax.set_title(f"{col} by Cluster", pad=15)
            ax.set_xlabel("Cluster")
            ax.set_ylabel(col)
            ax.grid(True)

        # Remove empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle("Numerical Distributions by Cluster", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()


    def plot_categorical_distributions(self, plot_type='bar'):
        """
        Plots the distribution of categorical variables for each cluster.
        
        Parameters:
        - plot_type: 'bar' for Barplot.
        """
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns
        if categorical_cols.empty:
            print("No categorical variables to plot.")
            return

        print("\n▶ Generating plots for categorical variables...")

        for col in categorical_cols:
            # Calculate counts and percentages
            counts = self.data.groupby(['Cluster', col]).size().reset_index(name='Count')
            counts['Total'] = counts.groupby('Cluster')['Count'].transform('sum')
            counts['Percentage'] = 100 * counts['Count'] / counts['Total']

            plt.figure(figsize=(10, 6))
            ax = sns.barplot(data=counts, x=col, y='Percentage', hue='Cluster')
            
            # Add labels
            for container in ax.containers:
                ax.bar_label(container, fmt='%.1f%%', label_type='edge')
            
            ax.set_title(f'{col} distribution by Cluster')
            ax.set_ylabel("Percentage (%)")
            ax.grid(True)
            plt.tight_layout()
            plt.show()
            
    def feature_importance_analysis(self, significance_level=0.05):
        """
        Statistically tests which features differ significantly between clusters.
        This helps identify which variables are 'driving' the clustering.
        """
        print("\n▶ Analyzing significant features (Discriminant Analysis)")

        numerical_cols = [col for col in self.data.select_dtypes(include=['int64', 'float64']).columns if col != 'Cluster']
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns

        print("\n* Numerical Variables (ANOVA / T-test):")
        for col in numerical_cols:
            try:
                # Group values by cluster
                values = [group[col].values for name, group in self.data.groupby("Cluster")]
                # Need at least 2 groups
                if len(values) < 2: continue

                if self.n_clusters == 2:
                    stat, p = ttest_ind(values[0], values[1])
                else:
                    from scipy.stats import f_oneway
                    stat, p = f_oneway(*values)
                
                if p < significance_level:
                    print(f" - {col}: Significant (p={p:.4f})")
            except Exception as e:
                print(f"Error checking {col}: {e}")

        print("\n* Categorical Variables (Chi-Square):")
        for col in categorical_cols:
            try:
                crosstab = pd.crosstab(self.data[col], self.data["Cluster"])
                chi2, p, _, _ = chi2_contingency(crosstab)
                if p < significance_level:
                    print(f" - {col}: Significant (p={p:.4f})")
            except Exception as e:
                print(f"Error checking {col}: {e}")

    def run_analysis(self):
        """
        Runs the full analysis pipeline.
        """
        print("\n" + "*"*80)
        print("Post-Clustering Analysis".center(80))
        print("*"*80)
        
        self.describe_clusters()
        
        # We automate the plots to avoid interrupting the user flow with inputs
        # Defaults: Boxplots for numbers, Barplots for categories
        self.plot_numerical_distributions(plot_type='box')
        self.plot_categorical_distributions(plot_type='bar')
        
        self.feature_importance_analysis()
        
        print("\n" + "*"*80)
