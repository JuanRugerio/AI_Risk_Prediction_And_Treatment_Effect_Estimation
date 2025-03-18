#Decision Tree Bayesian parameters
Bay_max_depth_low = 10
Bay_max_depth_high = 50
Bay_min_samples_split_low = 2
Bay_min_samples_split_high = 20
Bay_min_samples_leaf_low = 1
Bay_min_samples_leaf_high = 6
Bay_n_iter = 50

#LSTM Parametrization First Implementation
units_min_value = 32
units_max_value = 256
kernel_min_value = 2
kernel_max_value = 5
units_step = 32
dropout_min_value = 0.2
dropout_max_value = 0.5
dropout_step = 0.1
learning_rate_min_value = 1e-4
learning_rate_max_value = 1e-2
tuner_max_trials = 75
num_epochs = 20
training_batch_size = 16
Run_num = 8

#LSTM Second Implementation Parametrization

#Initial LSTM
dropout_rate = 0.2
test_split_size = 0.2
batch_size_num = 32
hidden_dim_size = 64
num_layers_n = 2
output_dim_n = 1
dropout_rate_n = 0.7
num_epochs_b = 10

#Hyperparameter tuning
num_epochs_hpt = 10
hidden_dim_low = 16
hidden_dim_high = 48
hidden_dim_step = 16
num_layers_dim_low = 1
num_layers_dim_high = 2
num_layers_dim_step = 1
dropout_range_low = 0.35
dropout_range_high = 0.7
lr_range_low = -10
lr_range_high = -5.3
weight_decay_range_low = -6
weight_decay_range_high = -3
max_evals_num = 20

#Last run best found hyperparameters
num_epochs_final = 10

#SHAPLEY Threshold value and threshold level
shap_threshold = .05349
shap_threshold_level = "low"
very_low_percentage = 0.00000000000000000001
low_percentage = 0.001
medium_percentage = 0.01
high_percentage = 0.1