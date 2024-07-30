# Telco Churn Prediction

## Project Overview

TELCO Inc is a phoning company facing a churn problem. This project aims to help TELCO Inc by:

1. Ranking their customers according to the probability of churn.
2. Identifying which clients they should contact and suggesting personalized discounts to prevent churn while maximizing future profit (balancing churn prevention and reduced profit per customer after discount).

## Dataset

The dataset contains two files:

1. `train.csv` - Training set including the labels in the column `CHURNED`.
2. `test.csv` - Test set with the same columns but without the labels.


- Docker
- Docker Compose

### Setup

1. **Clone the repository:**

    ```bash
    git clone https://github.com/Toufdraaicha/Teleco_Churn_Prediction.git
    cd Teleco_Churn_Prediction
    ```

2. **Build the Docker image:**

    ```bash
    docker-compose build
    ```

3. **Run the Docker container:**

    ```bash
    docker-compose up
    ```

### Usage

The `main.py` script handles the entire workflow:

1. Load and preprocess the data.
2. Train the model.
3. Evaluate the model.
4. Rank customers by churn probability.
5. Save the ranked customers to a CSV file.

The output file will be saved as `data/ranked_customers.csv`.
