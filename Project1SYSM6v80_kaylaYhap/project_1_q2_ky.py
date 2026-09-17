import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
import matplotlib.pyplot as plt
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "data.npy")

print("Loading from:", file_path)

with open(file_path, "rb") as f:
    state = jnp.load(f)
    phi_static = jnp.load(f)
    phi_c1 = jnp.load(f)
    phi_c2 = jnp.load(f)
X = state[:, [0,1, 2]]
Y = phi_c1[:, 2:4]


key = jax.random.PRNGKey(0)
N = X.shape[0]
perm = jax.random.permutation(key, N)

X = X[perm]
Y = Y[perm]

n_train = int(0.8 * N)
X_train = X[:n_train]
X_test = X[n_train:]
Y_train = Y[:n_train]
Y_test = Y[n_train:]

x_mean = jnp.mean(X_train, axis=0)
x_std = jnp.std(X_train, axis=0) + 1e-8
y_mean = jnp.mean(Y_train, axis=0)
y_std = jnp.std(Y_train, axis=0) + 1e-8

X_train_n = (X_train - x_mean) / x_std
X_test_n = (X_test - x_mean) / x_std
Y_train_n = (Y_train - y_mean) / y_std
Y_test_n = (Y_test - y_mean) / y_std
class MLP(eqx.Module):
    layers: list

    def __init__(self, in_size, out_size, width, key):
        keys = jax.random.split(key, 4)
        self.layers = [
            eqx.nn.Linear(in_size, width, key=keys[0]),
            eqx.nn.Linear(width, width, key=keys[1]),
            eqx.nn.Linear(width, width, key=keys[2]),
            eqx.nn.Linear(width, out_size, key=keys[3]),
        ]

    def __call__(self, x):
        x = jax.nn.tanh(self.layers[0](x))
        x = jax.nn.tanh(self.layers[1](x))
        x = jax.nn.tanh(self.layers[2](x))
        x = self.layers[3](x)
        return x

model = MLP(in_size=3, out_size=2, width=64, key=jax.random.PRNGKey(42))

def loss_fn(model, x, y):
    pred = jax.vmap(model)(x)
    return jnp.mean((pred - y) ** 2)

def mae_fn(model, x, y):
    pred = jax.vmap(model)(x)
    return jnp.mean(jnp.abs(pred - y))

optimizer = optax.adam(1e-4)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
@eqx.filter_jit
def train_step(model, opt_state, x, y):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss
batch_size = 256
epochs = 400
num_batches = X_train_n.shape[0] // batch_size

train_losses = []
test_losses = []

for epoch in range(epochs):
    perm = jax.random.permutation(jax.random.PRNGKey(epoch + 1), X_train_n.shape[0])
    X_shuff = X_train_n[perm]
    Y_shuff = Y_train_n[perm]

    epoch_loss = 0.0

    for i in range(num_batches):
        xb = X_shuff[i * batch_size:(i + 1) * batch_size]
        yb = Y_shuff[i * batch_size:(i + 1) * batch_size]
        model, opt_state, loss = train_step(model, opt_state, xb, yb)
        epoch_loss += loss

    epoch_loss = epoch_loss / num_batches
    test_loss = loss_fn(model, X_test_n, Y_test_n)

    train_losses.append(float(epoch_loss))
    test_losses.append(float(test_loss))

    if epoch % 25 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:3d} | Train Loss = {epoch_loss:.6f} | Test Loss = {test_loss:.6f}")
Y_pred_test_n = jax.vmap(model)(X_test_n)
Y_pred_test = Y_pred_test_n * y_std + y_mean
Y_true_test = Y_test_n * y_std + y_mean
mae = jnp.mean(jnp.abs(Y_pred_test - Y_true_test), axis=0)
rmse = jnp.sqrt(jnp.mean((Y_pred_test - Y_true_test) ** 2, axis=0))
print("\nFinal metrics:")
print("MAE:", mae)
print("RMSE:", rmse)
# Plot 1: loss curves
plt.figure(figsize=(7, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Question 2 Training Curve")
plt.legend()
plt.grid(True)
plt.show()
# Plot 2: prediction vs true for each output
Y_true_np = np.array(Y_true_test)
Y_pred_np = np.array(Y_pred_test)
labels = ["Force component 1", "Force component 2"]
for i in range(2):
    plt.figure(figsize=(6, 6))
    plt.scatter(Y_true_np[:, i], Y_pred_np[:, i], alpha=0.5)
    min_val = min(Y_true_np[:, i].min(), Y_pred_np[:, i].min())
    max_val = max(Y_true_np[:, i].max(), Y_pred_np[:, i].max())
    plt.plot([min_val, max_val], [min_val, max_val], "--")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(f"Question 2: Predicted vs True ({labels[i]})")
    plt.grid(True)
    plt.show()