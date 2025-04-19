import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from visualize_composition_heatmap_16_3_3 import visualize_composition_heatmap
# both versions 16_3_3 (the last version) and 16_3_4 (the previous version) are working well. Need to compare the code

# Declare data_filtered as a global variable
data_filtered = None


def load_data(file_name):
    """
    Load data from an Excel file and validate its contents.
    Parameters:
        file_name (str): Path to the Excel file.
    Returns:
        DataFrame: Loaded dataset.
    """
    try:
        # Load the Excel file, ensuring the 'Name' column is treated as a string
        data = pd.read_excel(file_name, dtype={'Name': str})
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_name}")
    except Exception as e:
        raise Exception(f"Error loading file '{file_name}': {e}")
    print("\nDataset Validation:")
    print(f"Rows: {len(data)}, Columns: {len(data.columns)}")
    # Print the columns for verification
    print("Columns:", data.columns.tolist())
    # Verify that the 'Name' column is correctly loaded
    if 'Name' not in data.columns:
        raise ValueError("The input file must contain a 'Name' column.")
    if data['Name'].isnull().any():
        print("Warning: Missing values detected in the 'Name' column.")
    return data


def calculate_pearson_r(X, y):
    """
    Calculate Pearson's r correlation between each feature in X and the target y.
    Parameters:
        X (DataFrame): Feature matrix.
        y (Series): Target variable.
    Returns:
        dict: Dictionary of Pearson's r values for each feature.
    """
    correlations = {}
    for col in X.columns:
        r, _ = pearsonr(X[col], y)
        correlations[col] = round(r, 4)
    return correlations


def remove_outliers(X, y, model, sigma_threshold=2):
    """
    Remove outliers based on residuals.
    Parameters:
        X (DataFrame): Feature matrix.
        y (Series): Target variable.
        model: Fitted regression model.
        sigma_threshold (float): Standard deviation multiplier for outlier detection.
    Returns:
        tuple: Cleaned X, y, and outlier mask.
    """
    X_np = np.asarray(X)
    y_np = np.asarray(y).flatten()
    if X_np.shape[1] != (len(model.params) - 1):
        X_with_const = sm.add_constant(X_np)
    else:
        X_with_const = X_np
    y_pred = model.predict(X_with_const)
    residuals = y_np - y_pred
    residual_std = np.std(residuals)
    outliers_mask = np.abs(residuals) > sigma_threshold * residual_std
    print(f"\nOutliers removed: {outliers_mask.sum()}")
    print(f"Sample SD (Original): {np.std(y):.4f}")
    print(f"Sample SD (Cleaned): {np.std(y[~outliers_mask]):.4f}")
    return X[~outliers_mask], y[~outliers_mask], outliers_mask


def plot_regression(X_full, y_full, model, outliers_mask, title, residual_sd, sample_sd):
    """
    Plot observed vs predicted values with outliers highlighted.
    Parameters:
        X_full (DataFrame): Full feature matrix.
        y_full (Series): Full target variable.
        model: Fitted regression model.
        outliers_mask (array): Boolean array indicating outliers.
        title (str): Plot title.
        residual_sd (float): Residual standard deviation.
        sample_sd (float): Sample standard deviation.
    """
    print(f"X_full shape: {X_full.shape}")
    print(f"y_full shape: {y_full.shape}")
    print(f"outliers_mask shape: {outliers_mask.shape}")
    # Ensure the constant term is added to X_full if needed
    X_full_with_const = sm.add_constant(X_full)
    X_cleaned = X_full_with_const[~outliers_mask]
    y_cleaned = y_full[~outliers_mask]
    y_pred_cleaned = model.predict(X_cleaned)
    residuals_cleaned = y_cleaned - y_pred_cleaned
    residual_sd = np.std(residuals_cleaned)
    sample_sd = np.std(y_cleaned)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_cleaned, y_pred_cleaned, color='blue', alpha=0.7, label='Inliers')
    # Check if there are any outliers to plot
    if np.any(outliers_mask):
        X_outliers = X_full_with_const[outliers_mask]
        y_outliers = y_full[outliers_mask]
        y_pred_outliers = model.predict(X_outliers)
        plt.scatter(y_outliers, y_pred_outliers, color='red', alpha=0.7, label='Outliers (highlighted)')
    min_val = min(y_full.min(), y_pred_cleaned.min())
    max_val = max(y_full.max(), y_pred_cleaned.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1)
    plt.title(f"{title}\nSample SD: {sample_sd:.4f} | Residual SD: {residual_sd:.4f}")
    plt.xlabel("Observed Values")
    plt.ylabel("Predicted Values")
    plt.legend()
    plt.grid(True)
    plt.show()


def process_feature_statistics(results_df, significant_features, coefficients, t_values):
    """
    Process feature statistics from the results DataFrame.
    Parameters:
        results_df (DataFrame): Results DataFrame containing 'Name', 'Observed_Property', 'Predicted_Property', and feature columns.
        significant_features (list): List of significant feature names.
        coefficients (list): Coefficients corresponding to significant features.
        t_values (list): t-values corresponding to significant features.
    Returns:
        str: Formatted string containing feature statistics.
        dict: Dictionary of names with non-zero counts.
    """
    # Initialize the table header with commas (,) as delimiters
    table_header = (
        "Feature,Coefficient,t-value,Non-zero Counts,Sum of Counts,Average Count,Composition (Coefficient * Average Count),Names with Non-zero Counts\n"
    )
    table_rows = []  # To store rows of the table
    names_with_non_zero_counts = {}  # To store the "Names with Non-zero Counts" information
    for i, feature in enumerate(significant_features):
        if feature == "Intercept":  # Skip intercept as it doesn't have counts
            continue
        # Extract non-zero counts for the feature
        non_zero_counts = results_df[feature][results_df[feature] != 0]
        num_non_zero = len(non_zero_counts)
        sum_counts = non_zero_counts.sum()
        avg_count = sum_counts / num_non_zero if num_non_zero > 0 else 0
        # Collect names with non-zero counts
        names_with_non_zero = results_df['Name'][results_df[feature] != 0].tolist()
        # Composition: Coefficient * Average Count
        coefficient = coefficients[i]
        composition = coefficient * avg_count
        # Append formatted row for the feature
        names_with_non_zero_str = ", ".join(names_with_non_zero)
        table_rows.append(
            f"{feature},{coefficient:.4f},{t_values[i]:.4f},{num_non_zero},{sum_counts:.4f},{avg_count:.4f},{composition:.4f},{names_with_non_zero_str}"
        )
        # Update names_with_non_zero_counts dictionary
        for name in names_with_non_zero:
            names_with_non_zero_counts[name] = names_with_non_zero_counts.get(name, 0) + 1
    # Combine the table header and rows
    table_output = table_header + "\n".join(table_rows)
    return table_output, names_with_non_zero_counts


def create_occurrences_table(names_with_non_zero_counts):
    """
    Create an occurrences table showing how many times each name appears across significant features.
    Parameters:
        names_with_non_zero_counts (dict): Dictionary containing names with their non-zero counts.
    Returns:
        str: Formatted string containing occurrences table.
    """
    # Initialize the table header with commas (,) as delimiters
    table_header = "Name,Occurrence\n"
    table_rows = []
    # Generate rows for the occurrences table
    for name, count in sorted(names_with_non_zero_counts.items()):
        table_rows.append(f"{name},{count}")
    # Combine the table header and rows
    table_output = table_header + "\n".join(table_rows)
    return table_output


def tsvd_regression(X, y, feature_names, title="TSVD Regression", t_threshold=2, sigma_threshold=2, outliers_mask=None):
    """
    Perform regression using Truncated Singular Value Decomposition (TSVD).
    Parameters:
        X (DataFrame): Feature matrix.
        y (Series): Target variable.
        feature_names (list): List of feature names corresponding to columns in X.
        title (str): Title for the regression summary.
        t_threshold (float): T-value threshold for feature selection.
        sigma_threshold (float): Standard deviation multiplier for outlier detection.
        outliers_mask (array): Boolean array indicating outliers.
    Returns:
        TSVModel: Fitted regression model.
        DataFrame: Results DataFrame.
        list: Significant features.
    """
    global data_filtered  # Access the global 'data_filtered' DataFrame
    X_np = np.asarray(X)
    y_np = np.asarray(y).flatten()
    X_with_const = sm.add_constant(X_np)
    feature_names_with_const = ["Intercept"] + feature_names
    # Full model SVD
    U, Sigma, VT = np.linalg.svd(X_with_const, full_matrices=False)
    sigma_threshold_svd = 1e-10 * Sigma[0]
    k = np.sum(Sigma >= sigma_threshold_svd)
    Sigma_k = Sigma[:k]
    VT_k = VT[:k, :]
    U_k = U[:, :k]
    beta = VT_k.T @ np.linalg.inv(np.diag(Sigma_k)) @ U_k.T @ y_np
    # Compute t-values for the full model (including intercept)
    residuals_full = y_np - (X_with_const @ beta)
    residual_std_full = np.std(residuals_full, ddof=1)
    n = len(y_np)
    p = len(beta)
    sse_full = np.sum(residuals_full ** 2)
    mse_full = sse_full / (n - p)
    try:
        cov_matrix_full = mse_full * np.linalg.inv(X_with_const.T @ X_with_const)
    except:
        cov_matrix_full = mse_full * np.linalg.pinv(X_with_const.T @ X_with_const)
    se_full = np.sqrt(np.diag(cov_matrix_full))
    t_values = np.abs(beta / se_full)
    # Initialize significant_indices (same length as original features + intercept)
    significant_indices = np.zeros_like(t_values, dtype=bool)
    significant_indices[0] = True  # Always keep intercept
    significant_indices[1:] = t_values[1:] >= t_threshold  # Exclude intercept's t-value
    converged = False
    iteration = 0
    max_iterations = 100
    while not converged and iteration < max_iterations:
        iteration += 1
        # Prune features and re-estimate model parameters
        X_pruned = X_with_const[:, significant_indices]
        # Recompute SVD for the pruned model
        U_pruned, Sigma_pruned, VT_pruned = np.linalg.svd(X_pruned, full_matrices=False)
        sigma_threshold_pruned = 1e-10 * Sigma_pruned[0]
        k_pruned = np.sum(Sigma_pruned >= sigma_threshold_pruned)
        Sigma_k_pruned = Sigma_pruned[:k_pruned]
        VT_k_pruned = VT_pruned[:k_pruned, :]
        U_k_pruned = U_pruned[:, :k_pruned]
        beta_pruned = VT_k_pruned.T @ np.linalg.inv(np.diag(Sigma_k_pruned)) @ U_k_pruned.T @ y_np
        # Compute residuals and covariance matrix for pruned model
        residuals_pruned = y_np - (X_pruned @ beta_pruned)
        p_pruned = np.sum(significant_indices)  # Number of features including intercept
        sse_pruned = np.sum(residuals_pruned ** 2)
        mse_pruned = sse_pruned / (n - p_pruned)
        try:
            cov_matrix_pruned = mse_pruned * np.linalg.inv(X_pruned.T @ X_pruned)
        except:
            cov_matrix_pruned = mse_pruned * np.linalg.pinv(X_pruned.T @ X_pruned)
        se_pruned = np.sqrt(np.diag(cov_matrix_pruned))
        new_t_values = np.abs(beta_pruned / se_pruned)
        # Initialize new_significant_indices with the same length as the original
        new_significant_indices = significant_indices.copy()  # Same length
        # Keep intercept
        new_significant_indices[0] = True
        # Update non-intercept features based on new_t_values
        selected_indices = np.where(significant_indices)[
            0]  # Indices of current selected features (including intercept)
        for i in range(1, len(new_t_values)):  # Start at 1 (exclude intercept)
            original_index = selected_indices[i]
            if new_t_values[i] >= t_threshold:
                new_significant_indices[original_index] = True
            else:
                new_significant_indices[original_index] = False
        # Check for convergence
        if np.all(new_significant_indices == significant_indices):
            converged = True
        else:
            significant_indices = new_significant_indices.copy()
        # Debugging output
        print(f"Iteration {iteration}:")
        print(f"  Significant Features: {sum(significant_indices)}")
        print(f"  t-values: {new_t_values}")
    if not converged:
        print(f"Warning: Maximum iterations ({max_iterations}) reached without convergence.")
    # After convergence, finalize the pruned model
    X_pruned = X_with_const[:, significant_indices]
    U_pruned, Sigma_pruned, VT_pruned = np.linalg.svd(X_pruned, full_matrices=False)
    sigma_threshold_pruned = 1e-10 * Sigma_pruned[0]
    k_pruned = np.sum(Sigma_pruned >= sigma_threshold_pruned)
    Sigma_k_pruned = Sigma_pruned[:k_pruned]
    VT_k_pruned = VT_pruned[:k_pruned, :]
    U_k_pruned = U_pruned[:, :k_pruned]
    beta_pruned = VT_k_pruned.T @ np.linalg.inv(np.diag(Sigma_k_pruned)) @ U_k_pruned.T @ y_np
    # Compute final residuals and metrics
    residuals_pruned = y_np - (X_pruned @ beta_pruned)
    SSE = np.sum(residuals_pruned ** 2)
    R_sq = 1 - (SSE / np.sum((y_np - np.mean(y_np)) ** 2))
    sigma_res = np.std(residuals_pruned, ddof=0)
    # Finalize covariance and t-values for the pruned model
    p_pruned = np.sum(significant_indices) - 1  # Exclude intercept for degrees of freedom
    n = len(y_np)
    try:
        cov_matrix_pruned = (SSE / (n - p_pruned - 1)) * np.linalg.inv(X_pruned.T @ X_pruned)
    except:
        cov_matrix_pruned = (SSE / (n - p_pruned - 1)) * np.linalg.pinv(X_pruned.T @ X_pruned)
    se_pruned = np.sqrt(np.diag(cov_matrix_pruned))
    t_values_pruned = np.abs(beta_pruned / se_pruned)
    # Calculate Pearson's r for the pruned model
    y_pred_pruned = X_pruned @ beta_pruned
    r, _ = pearsonr(y_np, y_pred_pruned)
    r_squared_pearson = r ** 2
    F_stat = (R_sq / (1 - R_sq)) * ((n - p_pruned - 1) / p_pruned) if (1 - R_sq) != 0 else 0
    # Map significant indices to feature names
    significant_features = [feature_names_with_const[i] for i, sig in enumerate(significant_indices) if sig]

    class TSVModel:
        def __init__(self, params, se, t_values, rsquared, f_stat, rmse, sigma_res, significant_indices,
                     significant_features):
            self.params = params
            self.se = se
            self.t_values = t_values
            self.rsquared = rsquared
            self.f_stat = f_stat
            self.rmse = rmse
            self.sigma_res = sigma_res
            self.significant_indices = significant_indices
            self.significant_features = significant_features

        def predict(self, X):
            X_np = np.asarray(X)
            # Always add the constant column if it's not already present
            if X_np.shape[1] != len(self.significant_indices):
                X_with_const = sm.add_constant(X_np)
            else:
                X_with_const = X_np
            # Debugging output to verify shapes
            print(f"X_with_const shape in predict: {X_with_const.shape}")
            print(f"significant_indices shape in predict: {self.significant_indices.shape}")
            # Ensure consistent indexing
            return X_with_const[:, self.significant_indices] @ self.params

    model = TSVModel(
        beta_pruned,
        se_pruned,
        t_values_pruned,
        R_sq,
        F_stat,
        np.sqrt(SSE / n),
        sigma_res,
        significant_indices,
        significant_features
    )

    # Print regression summary
    print(f"\nTSVD Regression Summary for {title}:")
    print(f"sigma_threshold: {sigma_threshold:.2f}")  # Corrected to print the provided sigma_threshold
    print(f"t_threshold: {t_threshold:.2f}")
    print(f"Pearson's r: {r:.4f}")
    print(f"R-squared: {model.rsquared:.4f}")
    print(f"F-statistic: {model.f_stat:.2f}")
    print(f"RMSE: {model.rmse:.4f}")
    print(f"Residual SD: {model.sigma_res:.4f}")
    print("Significant Features:", model.significant_features)
    print("Coefficients:", model.params)
    print("t-values:", model.t_values)

    # Create a DataFrame with significant features, observed property, predicted property, and Name
    results_df = pd.DataFrame({
        "Name": data_filtered['Name'].reset_index(drop=True)[:len(y_np)],  # Align Name with current dataset size
        "Observed_Property": y_np,
        "Predicted_Property": y_pred_pruned
    })

    # Add significant features to the results DataFrame
    results_df = pd.concat([results_df, pd.DataFrame(X_pruned, columns=model.significant_features)], axis=1)

    # Debugging: Print results_df to ensure it contains the expected data
    print("\nDebug: results_df DataFrame:")
    print(results_df.head())

    # Save results to a file
    output_file = f"{file_name.split('.')[0]}-{title.replace(' ', '_').replace('(', '').replace(')', '')}_full.csv"
    try:
        with open(output_file, 'w', newline='') as f:
            # Write model summary to the file
            f.write(f"--- TSVD Regression Summary for {title} ---\n")
            f.write(f"sigma_threshold: {sigma_threshold:.2f}\n")
            f.write(f"t_threshold: {t_threshold:.2f}\n")
            f.write(f"Pearson's r: {r:.4f}\n")
            f.write(f"R-squared: {model.rsquared:.4f}\n")
            f.write(f"F-statistic: {model.f_stat:.2f}\n")
            f.write(f"RMSE: {model.rmse:.4f}\n")
            f.write(f"Residual SD: {model.sigma_res:.4f}\n")
            f.write("Significant Features:\n")
            f.write(", ".join(model.significant_features) + "\n")
            f.write("Coefficients:\n")
            f.write(", ".join(map(str, model.params)) + "\n")
            f.write("t-values:\n")
            f.write(", ".join(map(str, model.t_values)) + "\n")
            # Write outlier mask if provided
            if outliers_mask is not None:
                outlier_indices = np.where(outliers_mask)[0]
                f.write("\nOutlier Indices:\n")
                f.write(", ".join(map(str, outlier_indices)) + "\n")
                # Get the names of the outliers
                outlier_names = data_filtered['Name'].iloc[outlier_indices].tolist()
                f.write("\nOutlier Names:\n")
                f.write(", ".join(outlier_names) + "\n")
            # Write the results DataFrame to the file
            f.write("\n--- Results Table ---\n")
            results_df.to_csv(f, index=False, sep=',', lineterminator='\n')  # Use comma as separator
        # Append feature statistics to the same output file
        with open(output_file, 'a', newline='') as f:
            f.write("\n--- Feature Statistics ---\n")
            feature_statistics, names_with_non_zero_counts = process_feature_statistics(
                results_df=results_df,
                significant_features=model.significant_features,
                coefficients=model.params,
                t_values=model.t_values
            )
            f.write(feature_statistics)
        # Append occurrences table to the same output file
        with open(output_file, 'a', newline='') as f:
            f.write("\n--- Occurrences Table ---\n")
            occurrences_table = create_occurrences_table(names_with_non_zero_counts)
            f.write(occurrences_table)
        print(f"Full results saved to {output_file}")
    except PermissionError:
        print(f"Error: Permission denied when trying to save {output_file}. "
              f"Ensure the file is not open in another program.")
    except Exception as e:
        print(f"Error saving results: {e}")
    return model, results_df, significant_features  # Ensure all three are returned


def calculate_contributions(results_df, significant_features, coefficients):
    """
    Calculate contributions (impact) for each significant feature by multiplying the feature count by the coefficient.
    Parameters:
        results_df (DataFrame): Results DataFrame containing 'Name', 'Observed_Property', 'Predicted_Property', and feature columns.
        significant_features (list): List of significant feature names.
        coefficients (list): Coefficients corresponding to significant features.
    Returns:
        DataFrame: Contributions DataFrame with 'Name' as the index and significant features as columns.
    """
    # Initialize a dictionary to store contributions
    contributions_dict = {'Name': results_df['Name']}
    for i, feature in enumerate(significant_features):
        if feature == "Intercept":  # Skip intercept as it doesn't have counts
            continue
        # Multiply feature count by coefficient
        contributions_dict[feature] = results_df[feature] * coefficients[i]

    # Create a DataFrame from the contributions dictionary and set 'Name' as the index
    contributions_df = pd.DataFrame(contributions_dict).set_index('Name')

    # Replace non-finite values with zero
    contributions_df = contributions_df.replace([np.inf, -np.inf, np.nan], 0)

    return contributions_df


if __name__ == "__main__":
    # file_name = '#2025 Acid dyes data_sdf_from_xlsx_OB-10-0.95-100-Names.xlsx'
    file_name = '#2020 Zhang data_sdf_from_xlsx-pH4-10-10.xlsx'
    # file_name = '#2020 Zhang data_sdf_from_xlsx-pH6-10-10.xlsx'
    # file_name = '#2020 Zhang data_sdf_from_xlsx-pH9-10-10.xlsx'
    sigma_threshold = 3
    t_threshold = 2.5

    # Load and preprocess data
    data = load_data(file_name)
    prop_name = data.columns[-1]
    print(f"Automatically detected property name: {prop_name}")
    data_filtered = data[(data[prop_name].notna()) & (data[prop_name] != 0)]
    if len(data_filtered) < 2:
        print(f"Skipping '{prop_name}' due to insufficient data.")
        exit()

    print(f"Processing Property: {prop_name}")
    print(f"Dataset size: {len(data_filtered)}")
    X_full = data_filtered.drop(columns=[prop_name]).select_dtypes(include=[np.number])
    y_full = data_filtered[prop_name].values
    feature_names = X_full.columns.tolist()

    # Build the initial model (UNPACK ALL RETURN VALUES)
    print("\n--- Initial Model ---")
    initial_model, initial_results_df, initial_significant_features = tsvd_regression(
        X_full.values,
        y_full,
        feature_names=feature_names,
        title=f"{prop_name} (Initial)",
        t_threshold=t_threshold,
        sigma_threshold=sigma_threshold,
    )

    # Remove outliers
    X_cleaned, y_cleaned, outliers_mask = remove_outliers(
        X_full.values,
        y_full,
        initial_model,
        sigma_threshold=sigma_threshold
    )

    # Build the final model (UNPACK ALL RETURN VALUES)
    print("\n--- Final Model ---")
    final_model, final_results_df, final_significant_features = tsvd_regression(
        X_cleaned,
        y_cleaned,
        feature_names=feature_names,
        title=f"{prop_name} (Final)",
        t_threshold=t_threshold,
        sigma_threshold=sigma_threshold,
        outliers_mask=outliers_mask
    )

    # Plot results
    sample_sd_initial = np.std(y_full)
    plot_regression(X_full.values, y_full, initial_model, outliers_mask, title="Initial Model with Outliers",
                    residual_sd=initial_model.sigma_res, sample_sd=sample_sd_initial)

    sample_sd_final = np.std(y_cleaned)
    plot_regression(X_cleaned, y_cleaned, final_model, np.zeros(len(y_cleaned), dtype=bool),
                    title="Final Model (Cleaned Data)", residual_sd=final_model.sigma_res, sample_sd=sample_sd_final)

    # Print final metrics
    print("\nFinal Metrics:")
    print(f"  R-squared: {final_model.rsquared:.4f}")
    print(f"  Residual SD: {final_model.sigma_res:.4f}")
    print(f"  Sample SD (Original): {np.std(y_full):.4f}")
    print(f"  Sample SD (Cleaned): {np.std(y_cleaned):.4f}")
    print("  Note: Residual SD (model error) vs Sample SD (data variability)")

    # Calculate contributions (impact) for each significant feature
    contributions_df = calculate_contributions(
        results_df=final_results_df,
        significant_features=final_significant_features,
        coefficients=final_model.params
    )

    # Debugging: Print contributions_df to ensure it contains the expected data
    print("\nDebug: contributions_df DataFrame:")
    print(contributions_df.head())

    # Filter out the Intercept from final_significant_features
    final_significant_features_no_intercept = [feature for feature in final_significant_features if
                                               feature != "Intercept"]

    # Call the visualization function with correct parameters
    visualize_composition_heatmap(contributions_df, final_significant_features_no_intercept, tick_interval=1)
