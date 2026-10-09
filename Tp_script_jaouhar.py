
import pandas as pd
import matplotlib.pyplot as plt


# 1. Load the dataset
def load_data(filename):
    df = pd.read_csv(filename)
    return df


# 2. Display dataset information
def show_info(df):
    print("\nDataset dimensions:")
    print(df.shape)

    print("\nDataset information:")
    df.info()

    print("\nFirst 5 rows:")
    print(df.head())


# 3. Check missing values
def check_missing_values(df):
    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nMissing percentages:")
    print((df.isnull().mean() * 100).round(2))

    print("\nRows with missing values:")
    print(df.isnull().any(axis=1).sum())


# 4. Check duplicate records
def check_duplicates(df):
    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nDuplicate candidate IDs:")
    print(df["enrollee_id"].duplicated().sum())


# 5. Check invalid numerical values
def check_invalid_values(df):
    print("\nInvalid training hours:")
    print(len(df[df["training_hours"] <= 0]))

    print("\nInvalid city development index:")
    print(len(df[
        (df["city_development_index"] < 0) |
        (df["city_development_index"] > 1)
    ]))

    print("\nInvalid target values:")
    print(len(df[~df["target"].isin([0, 1])]))


# 6. Check categorical values
def check_categories(df):
    print("\nGender categories:")
    print(df["gender"].value_counts(dropna=False))

    print("\nEducation level categories:")
    print(df["education_level"].value_counts(dropna=False))

    print("\nCompany size categories:")
    print(df["company_size"].value_counts(dropna=False))

    print("\nCompany type categories:")
    print(df["company_type"].value_counts(dropna=False))

    print("\nExperience categories:")
    print(df["experience"].value_counts(dropna=False))

    print("\nLast new job categories:")
    print(df["last_new_job"].value_counts(dropna=False))


# 7. Check missing major discipline by education
def check_major_discipline(df):
    print("\nEducation levels of candidates with missing majors:")

    print(
        df[df["major_discipline"].isnull()]
        ["education_level"].value_counts(dropna=False)
    )


# 8. Clean missing values
def clean_missing_values(df):
    df_clean = df.copy()

    # School-level candidates do not have a university major
    df_clean.loc[
        df_clean["education_level"] == "Primary School",
        "major_discipline"
    ] = "Not Applicable"

    df_clean.loc[
        df_clean["education_level"] == "High School",
        "major_discipline"
    ] = "Not Applicable"

    # Fill missing categorical values
    columns = [
        "gender",
        "company_type",
        "company_size",
        "major_discipline",
        "education_level",
        "enrolled_university",
        "experience",
        "last_new_job"
    ]

    for col in columns:
        df_clean[col] = df_clean[col].fillna("Unknown")

    return df_clean


# 9. Fix inconsistent company size format
def clean_company_size(df):
    df["company_size"] = df["company_size"].replace(
        "10/49", "10-49"
    )

    return df


# 10. Detect outliers using IQR
def check_outliers(df):
    Q1 = df["training_hours"].quantile(0.25)
    Q3 = df["training_hours"].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = df[
        (df["training_hours"] < lower_limit) |
        (df["training_hours"] > upper_limit)
    ]

    print("\nLower limit:", lower_limit)
    print("Upper limit:", upper_limit)
    print("Number of potential outliers:", len(outliers))

    # Keep outliers because they may be valid
    return outliers


# 11. Show descriptive statistics
def show_statistics(df):
    print("\nNumerical statistics:")

    print(df[[
        "city_development_index",
        "training_hours"
    ]].describe())

    print("\nEducation level distribution:")
    print(df["education_level"].value_counts())

    print("\nTarget distribution:")
    print(df["target"].value_counts())

    print("\nMean training hours:")
    print(df["training_hours"].mean())

    print("\nMedian training hours:")
    print(df["training_hours"].median())


# 12. Visualize missing values
def plot_missing_values(df):
    missing = df.isnull().mean() * 100
    missing = missing[missing > 0]

    if len(missing) > 0:
        missing.sort_values().plot(kind="barh")

        plt.title("Missing Values by Column")
        plt.xlabel("Missing Values (%)")
        plt.ylabel("Columns")
        plt.tight_layout()
        plt.show()


# 13. Show statistical graphs
def show_graphs(df):

    # Training hours histogram
    plt.hist(df["training_hours"], bins=20)

    plt.title("Distribution of Training Hours")
    plt.xlabel("Training Hours")
    plt.ylabel("Number of Candidates")
    plt.show()

    # Training hours boxplot
    plt.boxplot(df["training_hours"])

    plt.title("Training Hours Boxplot")
    plt.ylabel("Training Hours")
    plt.show()

    # Education level distribution
    df["education_level"].value_counts().plot(kind="bar")

    plt.title("Education Level Distribution")
    plt.xlabel("Education Level")
    plt.ylabel("Number of Candidates")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Target distribution
    df["target"].value_counts().sort_index().plot(kind="bar")

    plt.title("Job Change Distribution")
    plt.xlabel("Target (0 = No, 1 = Yes)")
    plt.ylabel("Number of Candidates")
    plt.xticks(rotation=0)
    plt.show()


# 14. Analyze job change by education level
def analyze_education(df):

    # Exclude Unknown education levels
    education_data = df[
        df["education_level"] != "Unknown"
    ]

    job_change = (
        education_data.groupby("education_level")["target"].mean() * 100
    )

    print("\nJob change rate by education level:")
    print(job_change)

    job_change.plot(kind="bar")

    plt.title("Job Change Rate by Education Level")
    plt.xlabel("Education Level")
    plt.ylabel("Job Change Rate (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# 15. Compare original and cleaned datasets
def compare_data(df, df_clean):

    print("\nMissing values before and after cleaning:")

    comparison = pd.DataFrame({
        "Before": df.isnull().sum(),
        "After": df_clean.isnull().sum()
    })

    print(comparison)

    print("\nOriginal dataset:", df.shape)
    print("Cleaned dataset:", df_clean.shape)

    print("\nCompany size categories after cleaning:")
    print(df_clean["company_size"].value_counts())


# 16. Save cleaned dataset
def save_data(df):
    df.to_csv("aug_train_cleaned.csv", index=False)
    print("\nCleaned dataset saved successfully!")


# ------------------------------
# MAIN PROGRAM
# ------------------------------

if __name__ == "__main__":

    # Load data
    df = load_data("aug_train.csv")

    # Analyze original dataset
    print("\n===== ORIGINAL DATASET =====")

    show_info(df)
    check_missing_values(df)
    check_duplicates(df)
    check_invalid_values(df)
    check_categories(df)
    check_major_discipline(df)
    check_outliers(df)
    plot_missing_values(df)

    # Clean dataset
    print("\n===== DATA CLEANING =====")

    df_clean = clean_missing_values(df)
    df_clean = clean_company_size(df_clean)

    # Verify cleaning
    print("\n===== CLEANED DATASET =====")

    check_missing_values(df_clean)
    check_duplicates(df_clean)
    check_invalid_values(df_clean)
    compare_data(df, df_clean)

    # Statistical analysis
    print("\n===== STATISTICAL ANALYSIS =====")

    show_statistics(df_clean)
    show_graphs(df_clean)
    analyze_education(df_clean)

    # Save result
    save_data(df_clean)

    print("\nAnalysis completed!")
