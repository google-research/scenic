<a name="base_model"></a>
## Scenic `BaseModel`
A solution usually has several parts: data/task pipeline, model architecture,
losses and metrics, training and evaluation, etc. Given that much of research
done in Scenic is trying out different architectures, Scenic introduces the
concept of `model`, to facilitate plug-in/plug-out experiments. A Scenic model
is defined as the network architecture plus the losses that are used to update
the weights of the network as well as metrics that are used to evaluate the
output of the network. This is implemented as `BaseModel`.

`BaseModel` is an abstract class with three members: `get_metrics_fn`,
`loss_fn`, and a `build_flax_model`.

`get_metrics_fn` returns a callable function, `metric_fn`, that calculates the
metrics and returns a dictionary. The metric function computes `f(x_i, y_i)` on
a mini-batch, it has API:

```python
metric_fn(logits, label, weights)
```

The trainer will then aggregate and compute the mean across all samples
evaluated.

`loss_fn` is a function of API:

```python
loss = loss_fn(logits, batch,
model_params=None)
```

And finally a `flax_model` is returned from the `build_flax_model` function. A
typical usage pattern will be:

```python
model_cls = model_lib.models.get_model_cls('fully_connected_classification')
model = model_cls(config, dataset.meta_data)
flax_model = model.build_flax_model
dummy_input = jnp.zeros(input_shape, model_input_dtype)
model_state, params = flax_model.init(
    rng, dummy_input, train=False).pop('params')
```

And this is how to call the model:

```python
variables = {'params': params, **model_state} logits,
new_model_state = flax_model.apply(variables, inputs, ...)
```

The abstract classes defining Scenic models, including `BaseModel` that defines
the Scenic model as well as `ClassificationModel`,
`MultiLabelClassificationModel`, `EncoderDecoderModel`, `SegmentationModelthat`
that define losses and metrics for classification, seq2seq, and segmentation
tasks are defined in `model_lib/base_models`. A Scenic project can define a new
base-class based on the task, metrics or overwrite the existing one when it is
needed.

Also, it is important to say that this design pattern, although recommended, is
not forced and there is no issue deviating from such structure, as some projects
in Scenic already do that.
