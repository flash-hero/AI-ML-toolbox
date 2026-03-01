import pandas as pd  # Library for data manipulation
import os  # Library to interact with the operating system (e.g., checking file extensions)
import csv  # Library to handle CSV files
import tkinter as tk  # Library for creating GUI windows
from tkinter import filedialog, simpledialog  # Tools to open file selection dialogs

class DataImporter:
    """
    This class handles importing data from files.
    It opens a window for the user to select a file (CSV, Excel, or Text) and loads it.
    """
    
    def __init__(self):
        """
        Initializes the DataImporter.
        Sets up a hidden Tkinter window because we only need the file dialog, not a full GUI app.
        """
        self.root = tk.Tk()
        self.root.withdraw() # Hide the main window
 
    def detect_delimiter(self, filename):
        """
        Tries to guess the separator used in a CSV file (e.g., comma, semicolon, tab).
        
        Parameters:
        - filename: The path to the file.
        """
        try:
            # Open the file and read the first 1024 bytes (just a sample)
            with open(filename, 'r') as file:
                content = file.read(1024)
                # Use the csv Sniffer tool to guess the format
                sniffer = csv.Sniffer()
                dialect = sniffer.sniff(content, delimiters=[',', '\t', ';', '|', ' '])
                return dialect.delimiter
        except csv.Error:
            # If the Sniffer fails, try a manual check
            delimiters = [',', '\t', ';', '|', ' ']
            # Check if any of these delimiters exist in the content
            # Note: This is a bit simplistic, but acts as a fallback
            for delim in delimiters:
                 # Limitation: this checks 'content' variable which might not be defined if the 'try' failed before reading?
                 # Actually content is defined inside 'try'. If open fails, we go to except but content is not defined.
                 # Python variable scoping allows access if defined before error, but if error is csv.Error from sniff, content is defined.
                if 'content' in locals() and delim in content:
                    return delim
            
            print("No standard delimiter detected.")
            # If all else fails, ask the user manually
            delim = simpledialog.askstring("Input Required", "Please enter the separator used in the file:")
            if delim:
                return delim
            else:
                print("No separator provided. Cannot continue.")
                return None

    def import_data_file(self, delim=None):
        """
        Opens a file dialog for the user to select a dataset.
        Supports CSV, Excel, and Text files.
        """
        print("\n" + "="*80)
        print("Chargement des données".center(80))
        print("="*80)      
        
        # Open the file selection window
        chemin_fichier = filedialog.askopenfilename(
            filetypes=[("Fichiers CSV", "*.csv"), ("Fichiers Excel", "*.xlsx"), ("Fichiers texte", "*.txt")]
        )

        if not chemin_fichier:
            print("No file selected.")
            return None

        # Get the file extension (e.g., .csv)
        extension_fichier = os.path.splitext(chemin_fichier)[1].lower()

        if extension_fichier == '.csv':
            # Identify the separator if not provided
            if not delim:
                delim = self.detect_delimiter(chemin_fichier)
            # Read the CSV file into a pandas DataFrame
            return pd.read_csv(chemin_fichier, delimiter=delim)

        elif extension_fichier == '.txt':
            # Read text file as a raw string
            with open(chemin_fichier, 'r') as file:
                data = file.read()
            return data

        elif extension_fichier == '.xlsx':
            # Read Excel file into a pandas DataFrame
            return pd.read_excel(chemin_fichier)
