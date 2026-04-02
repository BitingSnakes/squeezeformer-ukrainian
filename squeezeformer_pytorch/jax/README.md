# JAX Squeezeformer

This folder contains a Flax/Optax implementation of the encoder and a CTC head
that mirrors the existing PyTorch model closely enough to train on TPU in Colab.

Install the JAX stack in Colab before importing it:

```bash
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install -U flax optax orbax-checkpoint
```

Verify that JAX actually sees the TPU before running the model:

```python
import jax
print(jax.default_backend())
print(jax.devices())
```

You should see `tpu` as the backend. If you see `cpu` together with a message like
`A Google TPU may be present ... Falling back to cpu`, your JAX install is not
using the TPU runtime. In Colab, the most reliable fix is:

```bash
pip uninstall -y jax jaxlib libtpu-nightly
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install -U flax optax orbax-checkpoint
```

If TPU initialization fails with an error like
`open(/dev/vfio/0): Device or resource busy`, the TPU VM is usually in a bad
state or already attached elsewhere. The fix is normally:

1. Switch Colab to a TPU runtime.
2. Run `Runtime -> Factory reset runtime`.
3. Reinstall the JAX TPU packages in the fresh session.
4. Re-run the backend check above before importing repo code.

The `transparent hugepages` warning is not the blocker here; it is only a
performance/startup warning.

The JAX training script intentionally uses single-process data loading. PyTorch
`DataLoader` worker forking is not a good match for JAX's multithreaded runtime
and can trigger warnings like `os.fork() is incompatible with multithreaded code`
or hang the process. If you pass `--num-workers`, `--pin-memory`, or
`--persistent-workers`, the JAX script will override them to safe values.

Minimal usage:

```python
import jax
import jax.numpy as jnp
from functools import partial

from squeezeformer_pytorch.jax import (
    SqueezeformerCTC,
    create_adamw,
    create_train_state,
    replicate_state,
    shard_batch,
    squeezeformer_variant,
    train_step,
)

model = SqueezeformerCTC(
    encoder_config=squeezeformer_variant("xs"),
    vocab_size=256,
)
tx = create_adamw(learning_rate=3e-4)
rng = jax.random.PRNGKey(0)

example_features = jnp.zeros((8, 320, 80), dtype=jnp.float32)
example_feature_lengths = jnp.full((8,), 320, dtype=jnp.int32)
state = create_train_state(
    rng,
    model,
    example_features=example_features,
    example_feature_lengths=example_feature_lengths,
    tx=tx,
)

state = replicate_state(state)

p_train_step = jax.pmap(
    partial(train_step, blank_id=0, axis_name="batch"),
    axis_name="batch",
)
batch = {
    "features": example_features,
    "feature_lengths": example_feature_lengths,
    "labels": jnp.ones((8, 64), dtype=jnp.int32),
    "label_lengths": jnp.full((8,), 64, dtype=jnp.int32),
}
sharded_batch = shard_batch(batch)
dropout_rng = jax.random.split(rng, jax.local_device_count())
state, metrics = p_train_step(state, sharded_batch, dropout_rng=dropout_rng)
```

Training entrypoint:

```bash
python -m squeezeformer_pytorch.jax.train --variant xs --batch-size 16 --epochs 10
```

If you just want to test the code path without TPU, force CPU before importing
JAX:

```python
import os
os.environ["JAX_PLATFORMS"] = "cpu"
import jax
```

The training helpers expect precomputed feature tensors of shape
`[batch, time, mel_bins]`. That keeps the TPU path simple and avoids mixing the
PyTorch audio frontend into the JAX graph.
