import pandas as pd  # Library for data manipulation and analysis
import numpy as np   # Library for numerical operations
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, LabelEncoder  # Tools for scaling and encoding data
from sklearn.model_selection import train_test_split  # Tool to split data into training and testing sets
import re  # Library for regular expressions (text pattern matching)
import category_encoders as ce  # Library for advanced categorical encoding techniques

class DataPreprocessor:
    """
    This class is responsible for cleaning and preparing data before it can be used for machine learning.
    It handles tasks like filling missing values, converting text to numbers, and scaling data.
    """
    
    def __init__(self, data):
        """
        Initializes the DataPreprocessor.
        
        Parameters:
        - data (pd.DataFrame): The raw dataset to be processed.
        """
        # We check if the input is actually a pandas DataFrame to avoid errors later
        if not isinstance(data, pd.DataFrame):
            raise ValueError("The provided data is not a DataFrame.")
        
        # We work on a copy of the data so we don't accidentally modify the original dataset outside this class
        self.data = data.copy()

    def infer_frequency(self, date_series):
        """
        Guesses the frequency of a date column (e.g., daily, weekly, monthly).
        
        Parameters:
        - date_series (pd.Series): A column containing dates.
        
        Returns:
        - str: A code representing the frequency (e.g., 'D' for Day, 'M' for Month), or None if it can't be guessed.
        """
        # We need at least 2 dates to calculate a difference/frequency
        if len(date_series) < 2:
            return None
            
        # .diff() calculates the time difference between consecutive dates
        # .value_counts() counts how often each time difference occurs
        deltas = date_series.diff().dropna().value_counts()
        
        if deltas.empty:
            return None
            
        # The most common time difference is likely the frequency
        most_common_delta = deltas.idxmax()
        
        # We map the time difference to a pandas frequency code string
        if most_common_delta == pd.Timedelta(days=1):
            return 'D'  # Daily
        elif most_common_delta == pd.Timedelta(weeks=1):
            return 'W'  # Weekly
        elif most_common_delta in [pd.Timedelta(days=28), pd.Timedelta(days=30), pd.Timedelta(days=31)]:
            # Monthly - we check if it starts at the beginning of the month (MS) or end (M)
            if date_series.dt.is_month_start.all():
                return 'MS'
            else:
                return 'M'
        elif most_common_delta in [pd.Timedelta(days=90), pd.Timedelta(days=91), pd.Timedelta(days=92), pd.Timedelta(days=93)]:
            # Quarterly
            if date_series.dt.is_quarter_start.all():
                return 'QS'
            else:
                return 'Q'
        elif most_common_delta in [pd.Timedelta(days=365), pd.Timedelta(days=366)]:
            # Yearly
            if date_series.dt.is_year_start.all():
                return 'AS'
            else:
                return 'A'
        else:
            # If the pattern is irregular or unknown
            return None

    def fill_missing_dates(self, date_col, fill_value=0, cat_fill_value='Non applicable'):
        """
        Fills in gaps in a time series (e.g., missing days) and fills the empty rows.
        
        Parameters:
        - date_col (str): The name of the date column.
        - fill_value: value to fill in numeric columns for the new rows (default 0).
        - cat_fill_value: value to fill in text columns (default 'Non applicable').
        """
        df = self.data.copy()
        # Ensure the column is converted to datetime objects
        df[date_col] = pd.to_datetime(df[date_col])

        # Find the range of dates in the data
        start_date = df[date_col].min()
        end_date = df[date_col].max()

        # Guess the frequency (e.g., is it daily data? weekly?)
        freq = self.infer_frequency(df[date_col])

        # Generate a complete list of dates from start to end with that frequency
        all_dates = pd.date_range(start=start_date, end=end_date, freq=freq)

        # Create a new dataframe with ALL possible dates
        df_all_dates = pd.DataFrame(all_dates, columns=[date_col])

        # Merge the original data into this complete list.
        # 'how=left' ensures we keep all dates in df_all_dates. 
        # Dates present in 'df' will have data; missing dates will have NaNs (empty values).
        df_merged = df_all_dates.merge(df, on=date_col, how='left')

        # Identify types of columns to fill them appropriately
        categorical_cols = [col for col in df.columns if df[col].dtype == 'object' and col != date_col]
        numerical_cols = [col for col in df.columns if df[col].dtype != 'object' and col != date_col]

        # Fill the NaNs (gaps) with the specified default values
        df_merged.fillna({col: fill_value for col in numerical_cols}, inplace=True)
        df_merged.fillna({col: cat_fill_value for col in categorical_cols}, inplace=True)

        # Restore original data types (merging can sometimes change types)
        for col in df.columns:
            if col != date_col: 
                df_merged[col] = df_merged[col].astype(df[col].dtype)

        print(f"Filled missing dates in '{date_col}'. Filled categories with '{cat_fill_value}'.")
        self.data = df_merged
        return self.data
    
    def prepare_time_series_data(self, date_col, separate_date_components=False):
        """
        Sorts data by date and optionally breaks the date into separate features (Year, Month, Day).
        
        Parameters:
        - date_col (str): The name of the date column.
        - separate_date_components (bool): If True, creates new columns for Year, Month, Day, etc.
        """
        if date_col not in self.data.columns:
             raise ValueError(f"Column {date_col} not found in data.")
             
        # Convert the column to datetime objects if it isn't already
        if not pd.api.types.is_datetime64_any_dtype(self.data[date_col]):
             self.data[date_col] = pd.to_datetime(self.data[date_col], errors='coerce')
        
        # Sort the rows by date (oldest first) so the sequence is correct
        self.data = self.data.sort_values(by=date_col).reset_index(drop=True)
        
        if separate_date_components:
            # Create new columns representing parts of the date
            # This helps models that can't understand 'datetime' objects directly
            self.data['Year'] = self.data[date_col].dt.year
            self.data['Month'] = self.data[date_col].dt.month
            self.data['Day'] = self.data[date_col].dt.day
            self.data['Hour'] = self.data[date_col].dt.hour
            self.data['Minute'] = self.data[date_col].dt.minute
            self.data['Second'] = self.data[date_col].dt.second
        
        return self.data

    def convert_date_columns(self):
        """
        Automatically looks for text columns that look like dates and converts them to datetime objects.
        """
        # Patterns to look for (e.g., YYYY-MM-DD)
        date_patterns = [
            re.compile(r'^\d{4}-\d{2}-\d{2}$'),  # YYYY-MM-DD
            re.compile(r'^\d{4}/\d{2}/\d{2}$'),  # YYYY/MM/DD
            re.compile(r'^\d{2}/\d{2}/\d{4}$'),  # DD/MM/YYYY
            re.compile(r'^\d{4}-\d{2}$')          # YYYY-MM
        ]

        for col in self.data.columns:
            try:
                # Only check text (object) columns
                if self.data[col].dtype == 'object':
                     sample = self.data[col].dropna().astype(str)
                     # If all non-empty values match a date pattern, convert the whole column
                     if not sample.empty and sample.apply(lambda x: any(pat.match(x) for pat in date_patterns)).all():
                        self.data[col] = pd.to_datetime(self.data[col], errors='coerce')
            except Exception as e:
                pass # If conversion fails, we just skip it
        return self.data

    def change_dtype(self, type_map):
        """
        Manually changes the data type of specific columns.
        
        Parameters:
        - type_map (dict): A dictionary where keys are column names and values are new types (e.g., {'age': 'float'}).
        """
        for col, dtype in type_map.items():
            if col in self.data.columns:
                try:
                    self.data[col] = self.data[col].astype(dtype)
                except Exception as e:
                    print(f"Error converting {col} to {dtype}: {e}")
        return self.data

    def get_categorical_columns(self):
        """
        Returns a list of all categorical (text-based) columns, excluding dates.
        """
        # Find date columns
        date_columns = [col for col in self.data.columns if pd.api.types.is_datetime64_any_dtype(self.data[col])]
        # Find non-number columns that are NOT dates
        categorical_columns = [col for col in self.data.select_dtypes(exclude=[np.number]).columns if col not in date_columns]
        return categorical_columns
    
    def remove_duplicates(self, drop_duplicate_cols=True, drop_duplicate_rows=True):
        """
        Removes duplicate rows and duplicate columns from the data.
        """
        if drop_duplicate_cols:
            # Keep only columns that are not duplicates of previous columns
            self.data = self.data.loc[:, ~self.data.columns.duplicated()]
            # Transpose, drop duplicate rows (original columns), and transpose back
            # This removes columns that have identical content
            self.data = self.data.T.drop_duplicates().T

        if drop_duplicate_rows:
            # Remove rows that are exact copies of other rows
            self.data.drop_duplicates(inplace=True)
        
        return self.data

    def handle_missing_values(self, numeric_strategy='mean', categorical_strategy='mode'):
        """
        Fills or removes missing values (NaNs).
        
        Parameters:
        - numeric_strategy: How to handle numbers ('mean', 'median', 'drop', etc.)
        - categorical_strategy: How to handle text ('mode', 'drop', 'unknown')
        """
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        non_numeric_columns = self.data.select_dtypes(exclude=[np.number]).columns

        # --- Handle Numeric Missing Values ---
        if numeric_strategy == 'drop':
             self.data.dropna(subset=numeric_columns, inplace=True)
        elif numeric_strategy == 'mean':
             # Fill with the average value
             self.data[numeric_columns] = self.data[numeric_columns].fillna(self.data[numeric_columns].mean())
        elif numeric_strategy == 'median':
             # Fill with the middle value
             self.data[numeric_columns] = self.data[numeric_columns].fillna(self.data[numeric_columns].median())
        elif numeric_strategy == 'mode':
             # Fill with the most common value
             for col in numeric_columns:
                  if not self.data[col].mode().empty:
                       self.data[col] = self.data[col].fillna(self.data[col].mode()[0])
        elif numeric_strategy == 'min':
             self.data[numeric_columns] = self.data[numeric_columns].fillna(self.data[numeric_columns].min())
        elif numeric_strategy == 'max':
             self.data[numeric_columns] = self.data[numeric_columns].fillna(self.data[numeric_columns].max())

        # --- Handle Categorical Missing Values ---
        if categorical_strategy == 'drop':
             self.data.dropna(subset=non_numeric_columns, inplace=True)
        elif categorical_strategy == 'mode':
             # Fill with the most common category
             for col in non_numeric_columns:
                  if not self.data[col].mode().empty:
                       self.data[col] = self.data[col].fillna(self.data[col].mode()[0])
        elif categorical_strategy == 'unknown':
             # Fill with a new specific category "Unknown"
             self.data[non_numeric_columns] = self.data[non_numeric_columns].fillna("Unknown")

        return self.data

    def normalize_numeric_columns(self, method='minmax'):
        """
        Scales number columns so they are on a similar scale (e.g., typically 0 to 1).
        This is important for many machine learning models.
        
        Parameters:
        - method: 'minmax' (scales to 0-1) or 'standard' (scales to mean 0, std 1)
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if method == 'minmax':
            scaler = MinMaxScaler()
            self.data[numeric_cols] = scaler.fit_transform(self.data[numeric_cols])
        elif method == 'standard':
            scaler = StandardScaler()
            self.data[numeric_cols] = scaler.fit_transform(self.data[numeric_cols])
        return self.data

    def encode_categorical_columns(self, method='onehot', target_col=None):  
        """
        Converts text categories into numbers so the model can understand them.
        
        Parameters:
        - method: 'onehot', 'label', or 'target'
        - target_col: needed only if using 'target' encoding
        """
        categorical_columns = self.get_categorical_columns()
        if not categorical_columns:
            return self.data, []

        new_features = []

        if method == 'onehot':
            # Creates new binary columns for each category (e.g., Color_Red, Color_Blue)
            self.data = pd.get_dummies(self.data, columns=categorical_columns, prefix=categorical_columns)
            # Find the new column names we just created
            for col in categorical_columns:
                 new_features.extend([c for c in self.data.columns if c.startswith(f"{col}_")])
            
        elif method == 'label':
            # Assigns a unique number to each category (e.g., Red=0, Blue=1)
            encoder = LabelEncoder()
            for col in categorical_columns:
                self.data[col] = self.data[col].astype(str)
                self.data[col] = encoder.fit_transform(self.data[col])
            new_features = categorical_columns

        elif method == 'target':
            # Replaces the category with the average target value for that category
            # Useful for high cardinality features
            if target_col is None or target_col not in self.data.columns:
                 raise ValueError("Target column must be specified for target encoding.")
            encoder = ce.TargetEncoder(cols=categorical_columns)
            self.data[categorical_columns] = encoder.fit_transform(self.data[categorical_columns], self.data[target_col])
            new_features = categorical_columns

        return self.data, new_features

    def select_features(self, exclude_cols=None):
        """
        Removes specific columns that we don't want to use for training.
        """
        if exclude_cols:
            self.data.drop(columns=[c for c in exclude_cols if c in self.data.columns], inplace=True)
        return self.data

    def split_data(self, target_col, test_size=0.2, random_state=42):
        """
        Splits the data into Training set (to teach the model) and Test set (to evaluate it).
        
        Parameters:
        - target_col: The column we are trying to predict (the label).
        - test_size: The portion of data to keep for testing (0.2 means 20%).
        """
        if target_col not in self.data.columns:
             raise ValueError(f"Target column {target_col} not found.")

        # X contains the features (everything EXCEPT the target)
        X = self.data.drop(columns=[target_col])
        # y contains only the target (the answer)
        y = self.data[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test

    def split_sequential_data(self, target_col, sequence_length=10, test_size=0.2):
        """
        Splits data for Time Series models (like LSTM). 
        Sequence order matters here, so we don't shuffle randomly.
        
        Parameters:
        - sequence_length: How many past time steps to look at to predict the next one.
        """
        if target_col not in self.data.columns:
             raise ValueError(f"Target column {target_col} not found.")
             
        # Determine where to cut the data between train/test based on time order
        split_idx = int(len(self.data) * (1 - test_size))
        
        train_data = self.data.iloc[:split_idx]
        test_data = self.data.iloc[split_idx:]
        
        # Helper function to create sequences of data (sliding window)
        def create_sequences(df, target_name, seq_len):
            X, y = [], []
            data_array = df.drop(columns=[target_name]).values
            target_array = df[target_name].values
            
            # Slide a window over the data
            for i in range(len(df) - seq_len):
                X.append(data_array[i:i+seq_len]) # The sequence of features
                y.append(target_array[i+seq_len]) # The target value immediately after the sequence
            return np.array(X), np.array(y)

        # Create sequences for both train and test sets
        X_train_seq, y_train_seq = create_sequences(train_data, target_col, sequence_length)
        X_test_seq, y_test_seq = create_sequences(test_data, target_col, sequence_length)

        return X_train_seq, X_test_seq, y_train_seq, y_test_seq

    def preprocess(self):
        """
        Automated pipeline that runs standard cleaning steps:
        1. Cleans column names (removes special chars).
        2. Removes duplicates.
        3. Handles missing values.
        """
        # Clean column names (keep only alphanumeric)
        self.data.columns = [re.sub(r'[^a-zA-Z0-9]+', '', col) for col in self.data.columns]
        self.remove_duplicates()
        self.handle_missing_values()        
        return self.data