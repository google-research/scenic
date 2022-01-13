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

## Implementing loss and metrics with data parallelism
In Scenic, all the default training loops are designed to support data
parallelism. To do so, we have to be careful about our loss
and metrics calculations.

When training on multiple devices on multiple hosts, the gradient calculations
are handled locally on each device, given the examples in the device batch. So,
in the loss function, we simply "average" over the loss of all examples in that
device and return a **scalar** value, indicating the loss in that device (Check
out `weighted_softmax_cross_entropy` loss in the [base_models/model_lib.py](./base_models/model_lib.py)
as an example). Then, in the training loop,  we compute the gradient on each
device given the loss on that device. Then we **average** over the gradient from
all devices in all hosts:

```python
grad = jax.lax.pmean(grad, axis_name='batch')
```

Note that the `pmean` operation is synchronised across all hosts.

For metrics, however, the averaging is not done locally to make sure that we
account for actual number of examples in the partial last batch of
test/validation sets.
So each device returns two items: (1) the sum of the "per-example" value of that
metric and (2) number of actual examples processed by that device (to be used
for normalizing the value of that metric). Then, we **sum** over these two items
over all devices in all hosts (check out `psum_metric_normalizer` function
in [base_models/model_lib.py](./base_models/model_lib.py) and pass a tuple of
two scalars for each metric `<sum_of_value:float, normalizer:int>`.
Then, the summary writer uses the sum and the normalizer to compute the final
value of the metric.
So if you implement a new metric, you should be careful of returning the sum
and normalizer instead of the average of metric value over the examples in the
device (local) batch to guarantee the correctness of metrics' calculation.

This might seem a bit complicated, however, this is necessary as this carefully
accounts for the potential partial last batch in the test/validation splits and
guarantees correct computation of metrics. More precisely, if we average
locally and compute the mean of local averages, the batches with less example
would contribute to the final mean as much as full batches.

Note that there are metrics that do not decompose across different examples,
and cannot be computed as `sum(metric_val)/N`, like Mean Average
Precision. For such metrics, we need a special procedure to bring all the
`<predictions, targets>` pairs to the host and then compute the metrics we want.
You can look at [DETR implementation](../projects/baselines/detr) to learn more
about how this can be implemented using `lax.all_gather`.
