# LLMTime

## Usage

### Running Predictions
- The `LLMTime_predictions.zip` file contains all the prediction runs. These were generated by executing either the `LLMTime_multi_region_multi_prediction.ipynb` or the `LLMTime_multi_region_single_prediction.ipynb` scripts, depending on the specific computation requirements (multi-region or single-region predictions).

### Evaluating Metrics
- The `llmtime_metrics.ipynb` notebook is used to calculate and save metrics results in a JSON file.

### General Steps
1. **Set Up the Project**:
   - Download all required notebooks (`*.ipynb`) and place them in the same folder.

2. **Prepare Data**:
   - Ensure all necessary data files (`.csv`) are placed in the `nyc_data` folder.

3. **Run Predictions**:
   - Open either `LLMTime_multi_region_multi_prediction.ipynb` or `LLMTime_multi_region_single_prediction.ipynb` in a Jupyter environment and execute the cells as needed to generate a prediction folder with the run results.

4. **Generate Metrics**:
   - Use the `llmtime_metrics.ipynb` notebook to compute evaluation metrics. The metrics results will be stored in a JSON file, we placed our results in the `metrics` folder.

# STDN
## installation
Requirements

- Python v3.11.8
- Ubuntu 22.04.4 LTS
- TensorFlow version: 2.16.1
- Keras: Integrated with TensorFlow (use `tensorflow.keras`)

## Usage

1. **Set Up the Project**:
   - Download all Python scripts (`*.py`) and place them in the same folder (let's name it `stdn`) (`stdn/*.py`).
   - Create a `data` folder in the same directory (`stdn/data/`).
   - Create an `hdf5s` folder for logs if it does not already exist (`stdn/hdf5s/`).

2. **Prepare the Data**:
   - Download and extract all data files (`*.npz`) from `data.zip` and place them in the `data` folder (`stdn/data/*.npz`).

3. **Training the Model**:
   - Open a terminal in the project folder (`stdn/`).
   - To train the model and get performance results for all test datasets, run the following command for the NYC taxi dataset:
     ```bash
     python main.py
     ```
   - For the NYC bike dataset, use:
     ```bash
     python main.py --dataset=bike
     ```

4. **Check Results**:
   - After running the commands, check the output results (e.g., RMSE, MAE, MAPE, and CRPS).
   - The trained models will be saved in the `hdf5s` folder for further use.

5. **Evaluate Performance for our project**:
   - After training the model, follow these steps to evaluate its performance:
     - For the NYC taxi dataset:
       - Save the trained model in the `hdf5s_taxi` folder.
       - Create `taxi_result` folder and run the script:
         ```bash
         python first_seq_taxi.py
         ```
     - For the NYC bike dataset:
       - Save the trained model in the `hdf5s_bike` folder.
       - Create `bike_result` folder and run the script 
         ```bash
         python first_seq_bike.py
         ```

6. **Output**:
   - Check the evaluation results in `taxi_result` and `bike_result`  folder  .



## Hyperparameters:
Please check the hyperparameters defined in main.py
