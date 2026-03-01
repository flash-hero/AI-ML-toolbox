import pandas as pd  # Library for data manipulation
import numpy as np   # Library for numerical operations
import seaborn as sns  # Library for beautiful statistical graphs
import matplotlib.pyplot as plt  # Library for basic plotting
from scipy.stats import ttest_ind, chi2_contingency  # Statistical tests
import math  # Standard mathematical functions
# Note: DataPreprocessor is imported but not used in this file, but we keep it if it's needed for future extensions.
from preprocessor import DataPreprocessor 

class DataAnalyzer:
    """
    This class is responsible for visualizing data and performing basic statistical analysis.
    It helps to understand the data's structure, distribution, and relationships between variables.
    """
    
    def __init__(self, data):
        """
        Initializes the DataAnalyzer.
        
        Parameters:
        - data (pd.DataFrame): The dataset to analyze.
        """
        self.data = data

    def info_data(self):
        """
        Prints a general overview of the dataset:
        - First and last rows
        - Shape (number of rows and columns)
        - Column names
        - Data types of each column
        """
        print("\n" + "="*80)
        print("Visualisation initiale des données".center(80))  # Title centered
        print("="*80) 

        print("\n Five first and last rows of the dataset:")
        print(self.data)

        print("\n Total number of rows and columns:")
        print(self.data.shape)

        print("\n Column names:")
        print(self.data.columns.tolist())

        print("\n Data types per column:")
        print(self.data.dtypes)


    def summarize_statistics(self):
        """
        Calculates and prints descriptive statistics (mean, min, max, etc.) for numeric columns
        and frequency counts for categorical (text) columns.
        """
        print("\n" + "="*80)
        print("Statistiques descriptives".center(80))
        print("="*80)

        # Statistics for numeric variables (numbers like Age, Salary)
        if len(self.data.select_dtypes(include=['float64', 'int64']).columns) > 0:
            print("\n Descriptive statistics for quantitative (numeric) variables")
            summary = self.data.describe()
            print(summary)
        else:
            print("There are no quantitative variables in the dataset.")

        # Statistics for qualitative variables (text/categories like Gender, City)
        if len(self.data.select_dtypes(include=['object']).columns) > 0:
            print("\n Descriptive statistics for qualitative (categorical) variables")
            # include=['object'] tells pandas to describe text columns specifically
            qualitative_summary = self.data.describe(include=['object'])
            print(qualitative_summary)
        else:
            print("There are no qualitative variables in the dataset.")

    def visualize_data(self, show_boxplots=True, show_pairplot=True, show_pie=True, show_heatmap=True, save_dir=None):
        """
        Generates various plots to visualize the data.
        
        Parameters:
        - show_boxplots: If True, shows box plots (good for seeing outliers).
        - show_pairplot: If True, shows scatter plots of all numeric variables against each other.
        - show_pie: If True, shows pie charts for categorical variables.
        - show_heatmap: Included for future use (currently handled by relation_continuous).
        - save_dir: Optional directory to save plots as images.
        """
        print("\n" + "="*80)
        print("Visualisations graphiques".center(80))
        print("="*80)
        
        import os
        generated_plots = []

        # separate columns by type
        continuous_columns = self.data.select_dtypes(include=['int64', 'int32', 'float64', 'float32']).columns
        categorical_columns = self.data.select_dtypes(include=['object', 'category']).columns
        continuous_data = self.data.select_dtypes(include=['float64', 'int64'])

        # --- Boxplots ---
        if show_boxplots and not continuous_columns.empty:
            print("\n Generating Boxplots...")
            num_cols = 3  # Number of plots per row
            num_rows = math.ceil(len(continuous_data.columns) / num_cols)

            # Create a big figure to hold all small plots
            fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 4*num_rows))
            axes = axes.flatten()  # Flatten the 2D grid of axes to a 1D list for easy looping

            for i, column in enumerate(continuous_data.columns):
                sns.boxplot(x=continuous_data[column], ax=axes[i])
                axes[i].set_title(f'Boxplot of {column}')
                axes[i].set_xlabel('Value')

            # Hide any empty subplots if the number of variables isn't a perfect multiple of 3
            for j in range(i+1, len(axes)):
                axes[j].axis('off')

            plt.tight_layout()
            if save_dir:
                path = os.path.join(save_dir, "boxplots.png")
                plt.savefig(path)
                plt.close()
                generated_plots.append(path)
            else:
                plt.show()

        # --- Pairplot ---
        if show_pairplot and len(continuous_columns) > 0:
            print("\n Generating Pairplot...")
            # Limit features for pairplot to avoid performance issues if too many features
            cols = list(continuous_columns)
            if len(cols) > 8:
                 print("Limiting pairplot to top 8 high variance features for performance.")
                 # Select columns with highest variance (most information)
                 cols = list(continuous_data.var().sort_values(ascending=False).index[:8])
            
            try:
                # Pairplot shows scatter plots for every pair of variables
                g = sns.pairplot(data=self.data[cols])
                plt.suptitle('Pairplot of Continuous Variables', y=1.02)
                if save_dir:
                    path = os.path.join(save_dir, "pairplot.png")
                    plt.savefig(path)
                    plt.close()
                    generated_plots.append(path)
                else:
                    plt.show()
            except Exception as e:
                print(f"Error displaying pairplot: {e}")

        # --- Pie Charts ---
        if show_pie and len(categorical_columns) > 0:
             print("\n Generating Pie Charts...")
             for i, column in enumerate(categorical_columns):
                category_counts = self.data[column].value_counts()
                plt.figure(figsize=(8, 6))
                plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=140)
                plt.axis('equal') 
                plt.title(column)
                
                if save_dir:
                    path = os.path.join(save_dir, f"pie_{column}.png")
                    plt.savefig(path)
                    plt.close()
                    generated_plots.append(path)
                else:
                    plt.show()
        
        return generated_plots

        if show_heatmap:
             # This is a placeholder. 
             # Detailed correlation heatmaps are usually generated in 'relation_continuous'.
             pass

      
    def relation_categorical(self, target, significance_level=0.05):
        """
        Analyzes relationships between a target variable and other categorical variables
        using the Chi-Square test.
        
        Parameters:
        - target: The column name of the target variable.
        - significance_level: Threshold for statistical significance (usually 0.05).
        """
        categorical_vars = [col for col in self.data.columns if self.data[col].dtype == 'object']

        if not categorical_vars:
            print("No categorical variables found in the dataset.")
            return

        crosstabs = {}
        chi2_stats = {}
        
        for var in categorical_vars:
            # Create a contingency table (cross-tabulation)
            crosstab = pd.crosstab(self.data[var], self.data[target])
            crosstabs[var] = crosstab

            # Perform Chi-Square test of independence
            chi2, p, _, _ = chi2_contingency(crosstab)
            chi2_stats[var] = (chi2, p)

        # Plot Heatmaps for the contingency tables
        for var, crosstab in crosstabs.items():
            plt.figure(figsize=(8, 6))
            sns.heatmap(crosstab, annot=True, cmap='coolwarm', fmt=".2f")
            plt.title(f"Relation between {target} and {var}")
            plt.xlabel(target)
            plt.ylabel(var)
            plt.show()

        # Sort features by Chi-square statistic (higher means simplified stronger relationship)
        ranked_features = sorted(chi2_stats.items(), key=lambda x: x[1][0], reverse=True)
        
        # Filter only significant features
        significant_features = [(var, (chi2, p)) for var, (chi2, p) in ranked_features if p < significance_level]
        
        print("\n Significant Features (Categorical):")
        for i, (var, (chi2, p)) in enumerate(significant_features):
            print(f"{i + 1}. Variable '{var}' - Chi-square statistic: {chi2:.4f}, P-value: {p:.4f}")

        return ranked_features

    def relation_continuous(self, target, significance_level=0.05):
        """
        Analyzes relationships between a target variable and other continuous (numeric) variables.
        - Uses Correlation Heatmap.
        - Uses T-test to compare means between groups defined by the target.
        
        Parameters:
        - target: The column name of the target variable (assumed to be binary/categorical for T-test).
        """
        continuous_vars = [col for col in self.data.columns if self.data[col].dtype != 'object']

        if not continuous_vars:
            print("No continuous variables found in the dataset.")
            return

        # --- Correlation Heatmap ---
        plt.figure(figsize=(8, 8))
        # Correlation measures how much two variables change together
        heatmap = sns.heatmap(self.data[continuous_vars].corr(), annot=True, cmap='Blues', cbar=False)
        plt.title("Correlation Map of Continuous Variables")
        
        # Check if save_path is passed (using a trick since we can't easily change signature in this multi-replace)
        # Actually, let's just make it return the figure or accept a kwarg if we could, 
        # but since I am editing specific blocks, I'll rely on the caller to handle file saving if they use the returned figure 
        # OR I will assume this method is mostly for CLI.
        # Ideally, I should have updated the signature. Let me do that in a separate step or assume I'm calling a new method from API.
        # For now, let's just allow plt.savefig if a global or passed var exists? No, that's messy.
        
        # Let's just create a new method `get_correlation_heatmap` used by API.
        # Or better, let's just update the signature here.
        pass # Placeholder for the edit below
        
    def relation_continuous(self, target, significance_level=0.05, save_dir=None):
        """
        Analyzes relationships between a target variable and other continuous (numeric) variables.
        - Uses Correlation Heatmap.
        - Uses T-test to compare means between groups defined by the target.
        """
        continuous_vars = [col for col in self.data.columns if self.data[col].dtype != 'object']

        if not continuous_vars:
            print("No continuous variables found in the dataset.")
            return

        # --- Correlation Heatmap ---
        plt.figure(figsize=(8, 8))
        heatmap = sns.heatmap(self.data[continuous_vars].corr(), annot=True, cmap='Blues', cbar=False)
        plt.title("Correlation Map of Continuous Variables")
        
        import os
        if save_dir:
             plt.savefig(os.path.join(save_dir, "correlation_heatmap.png"))
             plt.close()
        else:
             plt.show()

        # --- T-Test ---
        # T-test compares if the average value of a variable differs significantly 
        # between two groups (e.g., Target=0 vs Target=1).
        # This assumes the target is binary (has 2 classes).
        t_test_results = {}
        try:
            for var in continuous_vars:
                # We extract the values of 'var' for group 0 and group 1
                group0 = self.data[self.data[target] == 0][var]
                group1 = self.data[self.data[target] == 1][var]
                
                t_stat, p_value = ttest_ind(group0, group1)
                t_test_results[var] = (t_stat, p_value)

            ranked_features = sorted(t_test_results.items(), key=lambda x: abs(x[1][0]), reverse=True)

            significant_features = [(var, (t_stat, p)) for var, (t_stat, p) in ranked_features if p < significance_level]
            print("\n Significant Features (Continuous):")
            for i, (var, (t_stat, p)) in enumerate(significant_features):
                print(f"{i + 1}. Variable '{var}' - t-test statistic: {t_stat:.4f}, P-value: {p:.4f}")
                
            return ranked_features
            
        except Exception as e:
            print(f"Could not perform T-test (target might not be binary): {e}")
            return []

    def visualization(self):
        """
        Runs the standard visualization pipeline:
        1. Summarizes statistics.
        2. Generates plots.
        """
        self.summarize_statistics()
        self.visualize_data()