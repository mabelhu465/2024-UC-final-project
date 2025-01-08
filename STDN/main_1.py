import sys
import json
import properscoring as ps
#import keras
import tensorflow.keras as keras
from attention import Attention
from tensorflow.keras.losses import MeanSquaredError

import numpy as np


#import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session

# TensorFlow 2.x
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth is set")
    except RuntimeError as e:
        print(f"An error occurred: {e}")


import file_loader
import models
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# set_session(tf.Session(config=config))
from keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
import argparse
import os
import re

parser = argparse.ArgumentParser(description='Spatial-Temporal Dynamic Network')
parser.add_argument('--dataset', type=str, default='taxi', help='taxi or bike')
parser.add_argument('--batch_size', type=int, default=64,
                    help='size of batch')
parser.add_argument('--max_epochs', type=int, default=250,
                    help='maximum epochs')
parser.add_argument('--att_lstm_num', type=int, default=3,
                    help='the number of time for attention (i.e., value of Q in the paper)')
parser.add_argument('--long_term_lstm_seq_len', type=int, default=3,
                    help='the number of days for attention mechanism (i.e., value of P in the paper)')
parser.add_argument('--short_term_lstm_seq_len', type=int, default=7,
                    help='the length of short term value')
parser.add_argument('--cnn_nbhd_size', type=int, default=3,
                    help='neighbors for local cnn (2*cnn_nbhd_size+1) for area size')
parser.add_argument('--nbhd_size', type=int, default=2,
                    help='for feature extraction')
parser.add_argument('--cnn_flat_size', type=int, default=128,
                    help='dimension of local conv output')
parser.add_argument('--model_name', type=str, default='stdn',
                    help='model name')

args = parser.parse_args()
print(args)


class CustomStopper(tf.keras.callbacks.EarlyStopping):
    # add argument for starting epoch
    def __init__(self, monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', start_epoch=40):
        super().__init__(monitor=monitor, min_delta=min_delta, patience=patience, verbose=verbose, mode=mode)
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.start_epoch:
            super().on_epoch_end(epoch, logs)


def eval_together(y, pred_y, threshold):
    mask = y > threshold
    if np.sum(mask) == 0:
        return -1
    mape = np.mean(np.abs(y[mask] - pred_y[mask]) / y[mask])
    rmse = np.sqrt(np.mean(np.square(y[mask] - pred_y[mask])))

    return rmse, mape


def eval_lstm(y, pred_y, threshold):
    pickup_y = y[:, 0]
    dropoff_y = y[:, 1]
    pickup_pred_y = pred_y[:, 0]
    dropoff_pred_y = pred_y[:, 1]
    pickup_mask = pickup_y > threshold
    dropoff_mask = dropoff_y > threshold
    # pickup part
    if np.sum(pickup_mask) != 0:
        avg_pickup_mape = np.mean(np.abs(pickup_y[pickup_mask] - pickup_pred_y[pickup_mask]) / pickup_y[pickup_mask])
        avg_pickup_rmse = np.sqrt(np.mean(np.square(pickup_y[pickup_mask] - pickup_pred_y[pickup_mask])))
        avg_pickup_mae = np.mean(np.abs(pickup_y[pickup_mask] - pickup_pred_y[pickup_mask]))
        avg_pickup_crps =np.mean( ps.crps_gaussian(pickup_y[pickup_mask], mu=pickup_pred_y[pickup_mask], sig=1e-6))
    
    
    # dropoff part
    if np.sum(dropoff_mask) != 0:
        avg_dropoff_mape = np.mean(
            np.abs(dropoff_y[dropoff_mask] - dropoff_pred_y[dropoff_mask]) / dropoff_y[dropoff_mask])
        avg_dropoff_rmse = np.sqrt(np.mean(np.square(dropoff_y[dropoff_mask] - dropoff_pred_y[dropoff_mask])))
        avg_dropoff_mae = np.mean(np.abs(dropoff_y[dropoff_mask] - dropoff_pred_y[dropoff_mask]))
        avg_dropoff_crps =np.mean( ps.crps_gaussian(dropoff_y[dropoff_mask], mu=dropoff_pred_y[dropoff_mask], sig=1e-6))
    #return (avg_pickup_rmse, avg_pickup_mape), (avg_dropoff_rmse, avg_dropoff_mape)
    return (avg_pickup_rmse, avg_pickup_mape, avg_pickup_mae, avg_pickup_crps), (avg_dropoff_rmse, avg_dropoff_mape, avg_dropoff_mae, avg_dropoff_crps)


def main(batch_size=64, max_epochs=100, validation_split=0.2, early_stop=EarlyStopping()):
    model_hdf5_path = "./hdf5s/"
    

    if args.dataset == 'taxi':
        sampler = file_loader.file_loader()
    elif args.dataset == 'bike':
        sampler = file_loader.file_loader(config_path = "data_bike.json")
    else:
        raise Exception("Can not recognize dataset, please enter taxi or bike")
    

    
    modeler = models.models()
    initial_epoch = 0


    
    if args.model_name == "stdn":
        # Check for latest saved model
        saved_models = [f for f in os.listdir(model_hdf5_path) if f.endswith('.keras')]
        latest_model_path = None

        # loading training data
        att_cnnx, att_flow, att_x, cnnx, flow, x, y = sampler.sample_stdn(datatype="train",
                                                                          att_lstm_num=args.att_lstm_num, \
                                                                          long_term_lstm_seq_len=args.long_term_lstm_seq_len,
                                                                          short_term_lstm_seq_len=args.short_term_lstm_seq_len, \
                                                                          nbhd_size=args.nbhd_size,
                                                                          cnn_nbhd_size=args.cnn_nbhd_size)

        print("Start training {0} with input shape {2} / {1}".format(args.model_name, x.shape, cnnx[0].shape))


        if saved_models:
            latest_model_path = max([model_hdf5_path + f for f in saved_models], key=os.path.getctime)
            model = tf.keras.models.load_model(latest_model_path,custom_objects={'Attention': Attention,
             'mse': MeanSquaredError()})
            print(f"Loaded model from {latest_model_path}")

            match = re.search(r'Epoch(\d+)', latest_model_path)
            if match:
              initial_epoch = int(match.group(1))
              print(f"Resuming training from epoch {initial_epoch}")


        else:
           model = modeler.stdn(att_lstm_num=args.att_lstm_num, att_lstm_seq_len=args.long_term_lstm_seq_len, \
                             lstm_seq_len=len(cnnx), feature_vec_len=x.shape[-1], \
                             cnn_flat_size=args.cnn_flat_size, nbhd_size=cnnx[0].shape[1], nbhd_type=cnnx[0].shape[-1])
        
        # Model checkpoint callback
        total_samples = x.shape[0]
        samples_per_epoch = (total_samples + batch_size - 1) // batch_size  
        save_freq = samples_per_epoch * 90 # Save every 100 epochs

        
        checkpoint = ModelCheckpoint(model_hdf5_path + args.model_name + 'Epoch{epoch:03d}.keras', save_freq=save_freq, save_weights_only=False, verbose=1)
       
        

        model.fit( \
            x=att_cnnx + att_flow + att_x + cnnx + flow + [x, ], \
            y=y, \
            batch_size=batch_size, validation_split=validation_split, epochs=max_epochs,
            initial_epoch=initial_epoch,
            callbacks=[early_stop, checkpoint])

        att_cnnx, att_flow, att_x, cnnx, flow, x, y = sampler.sample_stdn(datatype="test", nbhd_size=args.nbhd_size,
                                                                          cnn_nbhd_size=args.cnn_nbhd_size)
        y_pred = model.predict( \
            x=att_cnnx + att_flow + att_x + cnnx + flow + [x, ], )
        threshold = float(sampler.threshold) / sampler.config["volume_train_max"]
        print("Evaluating threshold: {0}.".format(threshold))
        (prmse, pmape, pmae, pcrps), (drmse, dmape, dmae, dcrps) = eval_lstm(y, y_pred, threshold)
        print(
            "Test on model {0}:\npickup rmse = {1}, pickup mape = {2}%\ndropoff rmse = {3}, dropoff mape = {4}%".format(
                args.model_name, prmse, pmape * 100, drmse, dmape * 100))
        print(
    "Test on model {0}:\n"
    "Pickup Metrics:\n"
    "  RMSE: {1}\n"
    "  MAPE: {2:.2f}%\n"
    "  MAE: {3}\n"
    "  CRPS: {4}\n"
    "Dropoff Metrics:\n"
    "  RMSE: {5}\n"
    "  MAPE: {6:.2f}%\n"
    "  MAE: {7}\n"
    "  CRPS: {8}".format(
        args.model_name, 
        prmse, pmape * 100, pmae, pcrps, 
        drmse, dmape * 100, dmae, dcrps
    ) )
        results = {
          "model": args.model_name,
          "pickup_metrics": {
        "RMSE": prmse,
        "MAPE": pmape * 100,
        "MAE": pmae,
        "CRPS": pcrps
    },
        "dropoff_metrics": {
            "RMSE": drmse,
            "MAPE": dmape * 100,
            "MAE": dmae,
            "CRPS": dcrps
        }
                  }           
        result_file = f"{args.model_name}_results.json"
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=4)
        currTime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        #model.save(model_hdf5_path + args.model_name + currTime + ".hdf5")
        model.save(model_hdf5_path + args.model_name + currTime + ".keras")
        return

    else:
        print("Cannot recognize parameter...")
        return


if __name__ == "__main__":
    stop = CustomStopper(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='min', start_epoch=40)
    main(batch_size=args.batch_size, max_epochs=args.max_epochs, early_stop=stop)
