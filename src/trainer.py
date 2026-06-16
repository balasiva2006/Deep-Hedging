import tensorflow as tf
from tqdm.auto import tqdm
from src.quant_lib import calculate_pnl, cvar_loss

def run_mse_bootstrapping(model, S_paths, BS_deltas, K, epochs=7, batch_size=1024):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
    K_tf = tf.constant(K, dtype=tf.float32)
    num_steps = S_paths.shape[1] - 1

    dataset = tf.data.Dataset.from_tensor_slices((S_paths, BS_deltas)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    @tf.function
    def train_step(batch_S, batch_BS):
        curr_batch_size = tf.shape(batch_S)[0]
        with tf.GradientTape() as tape:
            states = model.get_initial_states(curr_batch_size)
            delta_prev = tf.zeros((curr_batch_size, 1))
            preds = tf.TensorArray(tf.float32, size=num_steps)

            for t in tf.range(num_steps):
                S_t_norm = tf.cast(tf.expand_dims(batch_S[:, t], -1), tf.float32) / K_tf
                BS_t = tf.cast(tf.expand_dims(batch_BS[:, t], -1), tf.float32)
                
                inp = tf.concat([S_t_norm, BS_t, delta_prev], axis=-1)
                d, states = model(inp, states)
                preds = preds.write(t, d)
                delta_prev = d

            final_deltas = tf.squeeze(tf.transpose(preds.stack(), [1, 0, 2]), axis=-1)
            loss = tf.reduce_mean(tf.square(final_deltas - tf.cast(batch_BS, tf.float32)))
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    for epoch in range(epochs):
        epoch_loss = tf.keras.metrics.Mean()
        for batch_S, batch_BS in tqdm(dataset, desc=f"MSE Epoch {epoch+1}"):
            loss = train_step(batch_S, batch_BS)
            epoch_loss.update_state(loss)
        print(f"MSE Epoch {epoch+1} Completed. Loss: {epoch_loss.result().numpy():.6f}")


def run_cvar_tuning(model, S_paths, BS_deltas, K, r, epochs=10, tc=0.0015, batch_size=1024):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    K_tf = tf.constant(K, dtype=tf.float32)
    r_tf = tf.constant(r, dtype=tf.float32)
    num_steps = S_paths.shape[1] - 1

    dataset = tf.data.Dataset.from_tensor_slices((S_paths, BS_deltas)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    @tf.function
    def train_step(batch_S, batch_BS):
        curr_batch_size = tf.shape(batch_S)[0]
        with tf.GradientTape() as tape:
            states = model.get_initial_states(curr_batch_size)
            delta_prev = tf.zeros((curr_batch_size, 1))
            preds = tf.TensorArray(tf.float32, size=num_steps)

            for t in tf.range(num_steps):
                S_t_norm = tf.cast(tf.expand_dims(batch_S[:, t], -1), tf.float32) / K_tf
                BS_t = tf.cast(tf.expand_dims(batch_BS[:, t], -1), tf.float32)
                
                inp = tf.concat([S_t_norm, BS_t, delta_prev], axis=-1)
                d, states = model(inp, states)
                preds = preds.write(t, d)
                delta_prev = d

            final_deltas = tf.squeeze(tf.transpose(preds.stack(), [1, 0, 2]), axis=-1)
            
            pnl = calculate_pnl(batch_S, final_deltas, K_tf, r_tf, "call", tc)
            loss = cvar_loss(pnl)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    for epoch in range(epochs):
        epoch_loss = tf.keras.metrics.Mean()
        for batch_S, batch_BS in tqdm(dataset, desc=f"CVaR Epoch {epoch+1}"):
            loss = train_step(batch_S, batch_BS)
            epoch_loss.update_state(loss)
        print(f"CVaR Epoch {epoch+1} Completed. Loss: {epoch_loss.result().numpy():.2f}")