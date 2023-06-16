# FeatureSelection
Feature Selection with Nearest Neighbor. CS205 Project II

# Feature Selection Program

This project is a feature selection program implemented in Python. It allows you to perform forward search and backward elimination feature selection algorithms on datasets. The program prompts the user for inputs and requires Python 3 to execute.

## Requirements

- Python 3

## Usage

1. Ensure that the dataset file is present in the same directory as the Python script.

2. Open a terminal or command prompt.

3. Navigate to the directory containing the Python script and dataset file.

4. Run the following command to execute the program:

   ```
   python3 FeatureSelection.py
   ```

5. Follow the prompts displayed on the screen to interact with the program:

   - Specify the dataset file name or choose the default option.
   - Select the algorithm you want to run: forward selection or backward elimination.
   - If running forward selection, choose whether to enable early stopping for consecutive decreases in accuracy.

6. The program will display the results of the feature selection algorithm, including the best feature subset and its accuracy.

## Datasets

The program supports various datasets, including:

- Small dataset
- Large dataset
- XXXL dataset
- Wisconsin breast cancer dataset (data.csv)

Ensure that the dataset file is present in the same directory as the Python script to use it with the program.

## Additional Notes

- The program performs feature selection using the nearest neighbor classification algorithm.
- The program includes comments in the source code to explain each function's purpose and logic.

Please feel free to reach out to sshiv012@ucr.edu/ bgada001@ucr.edu if you have any questions or need further assistance.
