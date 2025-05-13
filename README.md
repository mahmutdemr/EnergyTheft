# Energy Theft Detection Project

## Overview

This project aims to detect energy theft using machine learning techniques. It includes scripts for training a model on historical data and making predictions on new data.

## Project Structure

```
EnergyTheft/
├── PRD.md
├── README.md
├── requirements.txt
├── config/
│   └── customer_types.json
├── data/
│   ├── Test.csv
│   ├── Training.csv
│   ├── Transformer.csv
│   └── TransformerTest.csv
├── docs/
│   └── Problem Statement v2.docx
├── results/
│   └── predictions.csv
└── src/
    ├── datagen.py
    ├── main.py
    ├── predict.py
    └── train.py
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd EnergyTheft
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training the Model

To train the model, run the `train.py` script:
```bash
python src/train.py
```
Make sure your training data (`Training.csv`, `Transformer.csv`) is in the `data/` directory.

### Making Predictions

To make predictions on new data, use the `predict.py` script:
```bash
python src/predict.py
```
This will likely use the `Test.csv` and `TransformerTest.csv` files from the `data/` directory and save the output to `results/predictions.csv`.

The `main.py` script might serve as an entry point for running different parts of the project. Check its content for more details.

The `datagen.py` script might be used for generating synthetic data or preprocessing existing data.

### Configuration

Customer types and other configurations might be managed in `config/customer_types.json`.

## Data

The project uses the following data files located in the `data/` directory:
*   `Training.csv`: Data used for training the model.
*   `Test.csv`: Data used for testing the model and making predictions.
*   `Transformer.csv`: Transformer data related to the training set.
*   `TransformerTest.csv`: Transformer data related to the test set.

The `results/` directory will store the output of predictions, such as `predictions.csv`.

## Documentation

Further details about the problem statement can be found in `docs/Problem Statement v2.docx`.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details (if one exists, otherwise, you can add one).
