import pandas as pd  # Library for data manipulation
import numpy as np   # Library for numerical operations
# Tools for statistical tests to find important features
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, chi2
from scipy.stats import f_oneway, kruskal, spearmanr, shapiro, levene
# Tools to prepare data (convert text to numbers, scale numbers)
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

class FeatureSelector:
    """
    This class selects the most important features (columns) from a dataset.
    It helps to remove irrelevant data that might confuse a machine learning model.
    """
    
    def __init__(self, X, y):
        """
        Initializes the FeatureSelector.
        
        Parameters:
        - X (pd.DataFrame): The input features (data to learn from).
        - y (pd.Series): The target variable (what we want to predict).
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Features (X) must be a pandas DataFrame.")
        if not isinstance(y, pd.Series):
            raise ValueError("Target (y) must be a pandas Series (single column).")

        self.X = X
        self.y = y
        self.results = {}          # Stores the scores of each feature
        self.removed_features = [] # List of features that were thrown away
        self.selected_features = [] # List of features that were kept

    def _check_variable_types(self):
        """
        Checks if columns are numeric (numbers) or categorical (text/categories).
        This helps decide which statistical test to use.
        """
        X_types = {
            col: 'catégorielle' if pd.api.types.is_object_dtype(self.X[col]) or pd.api.types.is_categorical_dtype(self.X[col]) else 'numérique'
            for col in self.X.columns
        }
        y_type = 'catégorielle' if pd.api.types.is_object_dtype(self.y) or pd.api.types.is_categorical_dtype(self.y) else 'numérique'
        return X_types, y_type

    def _apply_chi2_test(self):
        """
        Applies the Chi-Square test. 
        Used when both the input feature and the target are categorical (categories).
        """
        # We must convert text to numbers because Chi2 only understands numbers
        X_encoded = self.X.apply(LabelEncoder().fit_transform)
        y_encoded = LabelEncoder().fit_transform(self.y)
        
        # Calculate Chi2 scores and P-values
        chi2_scores, p_values = chi2(X_encoded, y_encoded)
        
        results = {col: {'method': 'Chi-square', 'score': score, 'pvalue': p_value} 
                   for col, score, p_value in zip(self.X.columns, chi2_scores, p_values)}
        return results

    def _apply_mutual_info(self, is_categorical_y, k=10):
        """
        Calculates Mutual Information (MI).
        MI measures how much knowing one variable tells us about the other.
        """

        # Work on a copy of the data
        X_encoded = self.X.copy()

        # Convert all categorical text columns to numbers
        for col in X_encoded.select_dtypes(include=['object', 'category']).columns:
            encoder = LabelEncoder()
            X_encoded[col] = encoder.fit_transform(X_encoded[col])

        # Normalize numeric columns to be between 0 and 1
        scaler = MinMaxScaler()
        num_cols = X_encoded.select_dtypes(include=['int64', 'float64']).columns
        if not num_cols.empty:
            X_encoded[num_cols] = scaler.fit_transform(X_encoded[num_cols])

        # Calculate Mutual Information scores
        if is_categorical_y:
            # Classification problem (predicting a category)
            mi_scores = mutual_info_classif(X_encoded, self.y)
        else:
            # Regression problem (predicting a number)
            # Ensure y is numeric
             if self.y.dtype == 'object':
                  y_encoded = LabelEncoder().fit_transform(self.y)
             else:
                  y_encoded = self.y
             mi_scores = mutual_info_regression(X_encoded, y_encoded)
 
        # Sort features by score (highest information first)
        mi_results = sorted(zip(self.X.columns, mi_scores), key=lambda x: x[1], reverse=True)

        # Keep only the top 'k' best features
        k = min(k, len(self.X.columns))
        selected_features_list = [col for col, _ in mi_results[:k]]

        # Return the results for these top features
        return {col: {'method': 'Mutual Information', 'score': score} for col, score in mi_results if col in selected_features_list}
    
    def _apply_spearman_correlation(self, col):
        """
        Calculates Spearman Correlation.
        Used to see if two variables increase/decrease together (monotonic relationship).
        """
        if self.X[col].dtype == 'object' or pd.api.types.is_categorical_dtype(self.X[col]):
            encoder = LabelEncoder()
            X_encoded = encoder.fit_transform(self.X[col])
        else:
            X_encoded = self.X[col]
        
        if self.y.dtype == 'object' or pd.api.types.is_categorical_dtype(self.y):
             y_encoded = LabelEncoder().fit_transform(self.y)
        else:
             y_encoded = self.y

        try:
            stat, pvalue = spearmanr(X_encoded, y_encoded)
            return stat, pvalue
        except ValueError:
            return 0.0, 1.0 # Return no correlation if calculation fails

    def _apply_anova_or_kruskal(self, col):
        """
        Applies ANOVA or Kruskal-Wallis test.
        Used when Input is Numeric and Target is Categorical.
        Checks if the average value of the input changes significantly for different target groups.
        """
        # Ensure target is encoded as numbers (group IDs)
        if not np.issubdtype(self.y.dtype, np.number):
            encoder = LabelEncoder()
            y_encoded = encoder.fit_transform(self.y)
        else:
            y_encoded = self.y

        # Group the input values by the target group
        groups = [self.X[col][y_encoded == level] for level in np.unique(y_encoded)]
        # Filter out very small groups
        groups = [g for g in groups if len(g) > 1]
        
        if len(groups) < 2:
             return 0, 1.0, "Insufficient Data"

        # Check if data looks like a normal "Bell Curve" (Normality)
        normality = all((len(group) > 3 and np.ptp(group) > 0 and shapiro(group).pvalue > 0.05) for group in groups)
        # Check if groups have similar spread/variance (Homogeneity)
        try:
             homogeneity = levene(*groups).pvalue > 0.05 if len(groups) > 1 else True
        except:
             homogeneity = False

        if all(len(group) > 3 for group in groups):
            # If data is normal and homogeneous, use ANOVA (Parametric test, powerful)
            if normality and homogeneity:
                try:
                    stat, pvalue = f_oneway(*groups)
                    return stat, pvalue, "ANOVA"
                except ValueError:
                    pass
        
        # Otherwise, use Kruskal-Wallis (Non-parametric test, safer)
        try:
            stat, pvalue = kruskal(*groups)
        except ValueError:
             stat, pvalue = 0, 1.0
        return stat, pvalue, "Kruskal-Wallis"

    def select_features(self, method='auto', k=10):
        """
        Main method to run feature selection.
        
        Parameters:
        - method: 'auto', 'chi2', 'mutual_info', 'spearman', 'anova'.
        - k: Number of top features to keep (used for mutual_info).
        """
        
        X_types, y_type = self._check_variable_types()
        # print(f"Types detected: X -> {X_types}, y -> {y_type}")

        if len(self.X) != len(self.y):
            raise ValueError(f"Number of rows in X ({len(self.X)}) does not match y ({len(self.y)})")

        self.results.clear()

        # Deciding the best method automatically ('auto')
        if method == 'auto':
            # Both categorical -> Chi-Square
            if all(val == 'catégorielle' for val in X_types.values()) and y_type == 'catégorielle':
                method = 'chi2'
            # Both numeric -> Spearman Correlation
            elif all(val == 'numérique' for val in X_types.values()) and y_type == 'numérique':
                method = 'spearman' 
            # Numeric Input & Categorical Target -> ANOVA
            elif all(val == 'numérique' for val in X_types.values()) and y_type == 'catégorielle':
                method = 'anova'
            # Mixed or other -> Mutual Information (works for almost everything)
            else:
                method = 'mutual_info'

        # Running the chosen method
        if method == 'chi2':
            self.results.update(self._apply_chi2_test())
        elif method == 'mutual_info':
            self.results = self._apply_mutual_info(is_categorical_y=(y_type == 'catégorielle'), k=k)
        elif method == 'spearman':
            for col in self.X.columns:
                stat, pvalue = self._apply_spearman_correlation(col)
                self.results[col] = {'method': 'Spearman', 'stat': stat, 'pvalue': pvalue}
        elif method == 'anova':
             for col in self.X.columns:
                stat, pvalue, method_name = self._apply_anova_or_kruskal(col)
                self.results[col] = {'method': method_name, 'stat': stat, 'pvalue': pvalue}

        # Filtering features based on results
        if method == 'mutual_info':
            # For Mutual Info, we already selected the top K features
            self.selected_features = list(self.results.keys())
        else:
            # For statistical tests, we check if the relationship is statistically significant (p-value < 0.05)
            # A low p-value means it's unlikely the relationship is due to random chance.
            self.selected_features = [col for col, res in self.results.items() if 'pvalue' in res and res['pvalue'] < 0.05]

        # Identify which features were removed
        self.removed_features = [col for col in self.X.columns if col not in self.selected_features]
        
        # Return the simplified X (only selected columns) and the original y
        return self.X[self.selected_features], self.y


    def run_feature_selection(self, X_feature, y_target, method='auto', k=10):
        """
        Helper function to easily run the entire feature selection process.
        """
        selector = FeatureSelector(X_feature, y_target)
        X_selected, y_selected = selector.select_features(method=method, k=k)
        return X_selected, y_selected
