## installation
Requirements

- Python v3.11.8
- Ubuntu 22.04.4 LTS
- TensorFlow version: 2.16.1
- Keras: Integrated with TensorFlow (use `tensorflow.keras`)

## Usage
 - Download all codes (*\*.py*) and put them in the same folder (let's name it "stdn") (*stdn/\*.py*)
  - Create "data" folder in the same folder (*stdn/data/*)
  - Create "hdf5s" folder for logs (if not exist) (*stdn/hdf5s/*)
  - Download and extract all data files (*\*.npz*) from data.zip and put them in "data" folder (*stdn/data/\*.npz*)
  - Open terminal in the same folder (*stdn/*)
  - To train the model and get performance results for all test datastes, run with "python main.py" for NYC taxi dataset, or "python main.py --dataset=bike" for NYC bike dataset
  ```
  python main.py
  ```
  ```
  python main.py --dataset=bike
  ```
  - Check the output results (RMSE, MAE, MAPE and CRPS). Models are saved to "hdf5s" folder for further use.
  - After training the model, to get the performance results in our project, run with "first_seq_taxi.py" for NYC taxi dataset, or " "first_seq_bike.py" for NYC bike dataset
## Hyperparameters:
Please check the hyperparameters defined in main.py
