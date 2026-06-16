import tensorflow as tf
from src.quant_lib import calculate_pnl, cvar_loss

def run_mse_bootstrapping(model, S_paths, BS_deltas, K, epochs=7):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
    K_tf = tf.constant(K, dtype=tf.float32)
    num_steps = S_paths.shape[1] - 1

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            batch_size = tf.shape(S_paths)[0]
            states = model.get_initial_states(batch_size)
            delta_prev = tf.zeros((batch_size, 1))
            preds = tf.TensorArray(tf.float32, size=num_steps)

            for t in range(num_steps):
                S_t_norm = tf.expand_dims(S_paths[:, t], -1) / K_tf
                inp = tf.concat([S_t_norm, tf.expand_dims(BS_deltas[:, t], -1), delta_prev], axis=-1)
                d, states = model(inp, states)
                preds = preds.write(t, d)
                delta_prev = d

            final_deltas = tf.squeeze(tf.transpose(preds.stack(), [1,0,2]), axis=-1)
            loss = tf.reduce_mean(tf.square(final_deltas - tf.cast(BS_deltas, tf.float32)))
        
        optimizer.apply_gradients(zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables))
        print(f"MSE Epoch {epoch+1}, Loss: {loss.numpy():.6f}")

def run_cvar_tuning(model, S_paths, BS_deltas, K, r, epochs=10, tc=0.0015):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    K_tf = tf.constant(K, dtype=tf.float32)
    num_steps = S_paths.shape[1] - 1

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            batch_size = tf.shape(S_paths)[0]
            states = model.get_initial_states(batch_size)
            delta_prev = tf.zeros((batch_size, 1))
            preds = tf.TensorArray(tf.float32, size=num_steps)

            for t in range(num_steps):
                S_t_norm = tf.expand_dims(S_paths[:, t], -1) / K_tf
                inp = tf.concat([S_t_norm, tf.expand_dims(BS_deltas[:, t], -1), delta_prev], axis=-1)
                d, states = model(inp, states)
                preds = preds.write(t, d)
                delta_prev = d

            final_deltas = tf.squeeze(tf.transpose(preds.stack(), [1,0,2]), axis=-1)
            pnl = calculate_pnl(S_paths, final_deltas, K_tf, r, "call", tc)
            loss = cvar_loss(pnl)

        optimizer.apply_gradients(zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables))
        print(f"CVaR Epoch {epoch+1}, Loss: {loss.numpy():.2f}")