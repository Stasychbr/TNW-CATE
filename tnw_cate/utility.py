from .generators import *
from .models import get_full_model, get_alpha_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

def train_kernel(X_train, y_train, n, m, mlp_coef, epochs_num, validation, seed, batch_size, patience=3, learning_rate=0.01, topology=None, verbose=2):
    model = get_full_model(n, m, topology, seed)

    early_stopping = EarlyStopping(patience = patience, monitor = "val_loss",  mode = "min", restore_best_weights = True)
    adam = Adam(learning_rate=learning_rate) #0.0005; 0.01
    model.compile(optimizer=adam, loss='mean_squared_error')

    train_gen = TrainGenerator(X_train, y_train, n, m, mlp_coef, batch_size)
    hist = model.fit(train_gen, batch_size=batch_size, epochs=epochs_num, callbacks=[early_stopping], verbose=verbose, validation_data=validation).history
    return model, hist

def train_alpha(cnt_x, cnt_y, trt_x, trt_y, n_c, n_t, m, epochs_num, tasks_num, validation, seed, alpha, batch_size, patience=5, learning_rate=0.002, topology=None, verbose=2):
    model = get_alpha_model(n_c, n_t, m, topology, seed)
    output_name = model.output_names[0]
    early_stopping = EarlyStopping(patience = patience, monitor = f"val_{output_name}_loss",  mode = "min", restore_best_weights = True)
    adam = Adam(learning_rate=learning_rate) #0.0005
    model.compile(optimizer=adam, loss=['mse', 'mse'], loss_weights=[1.0, alpha])
    train_gen = AlphaGenerator(cnt_x, cnt_y, trt_x, trt_y, n_c, n_t, m, tasks_num, batch_size)
    hist = model.fit(train_gen, batch_size=batch_size, epochs=epochs_num, callbacks=[early_stopping], verbose=verbose, validation_data=validation).history
    return model, hist