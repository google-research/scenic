# Copyright 2024 The Scenic Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines different learning_rate schedules."""

import jax.numpy as jnp
import ml_collections


def polynomial_lr_scheduler(step, decay_steps, end_factor, power):
  """Same behavior as tf.train.polynomial_decay.

  This is the original formula for this learning rate scheduler:
    ```
    end_learning_rate = config['base_learning_rate'] * config['end_factor']
    step = min(config['decay_steps'], step)
    decayed_learning_rate = (config['base_learning_rate'] -
                             end_learning_rate) * (
                                 1 - step / config['decay_steps'])**(
                                     config['power']) + end_learning_rate
    ```
  We rewrite this as a multiplicative factor for the initial learning rate.
  Args:
    step: int; Current step.
    decay_steps: int; Parameter of the decay function.
    end_factor: float; Final lr is: initial lr x end_factor.
    power: int; Parameter of the decay function.

  Returns:
    Scaling factor applied to the learning rate on the given step.
  """

  decay = step <= decay_steps
  decayed_learning_rate = (1 - end_factor) * (
      decay * (1 - step / decay_steps))**(power) + end_factor
  return decayed_learning_rate


def piecewise_constant_scheduler(step, decay_events, decay_factors):
  """Gives a scaling factor based on Piecewise Constant scheduling.

  Args:
    step: int; Current step.
    decay_events: list(int); List of steps in which a decay is applied.
    decay_factors: list(int); List containing the absolute ratio of the decay
      applied on the decay events. Note that each element of decay_factors is
      absolute (not relative). For example, to decay the learning rate to 0.5 of
      its initial value after 100 steps, followed by 0.1 of its *initial value*
      after 200 steps, with a plateau of 0.1 of its initial value thereafter,
      use decay_events = [100, 200] and decay_factors = [0.5, 0.1].

  Returns:
    Scaling factor applied to the learning rate on the given step.
  """
  boundaries = jnp.array(decay_events)
  factors = jnp.array([1.0] + decay_factors)
  index = jnp.sum(boundaries < step)
  ratio = jnp.take(factors, index)
  return ratio


def piecewise_linear_scheduler(step, decay_events, decay_factors):
  """Gives a scaling factor based on Piecewise Linear scheduling.

  Args:
    step: int; Current step.
    decay_events: list(int); List of steps in which a decay is applied.
    decay_factors: list(int); List containing the absolute ratio of the decay
      applied on the decay events.  Note that each element of decay_factors is
      absolute (not relative). For example, to decay the learning rate to 0.5 of
      its initial value after 100 steps, followed by 0.1 of its *initial value*
      after 200 steps, with a plateau of 0.1 of its initial value thereafter,
      use decay_events = [100, 200] and decay_factors = [0.5, 0.1].

  Returns:
    Scaling factor applied to the learning rate on the given step.
  """
  boundaries = jnp.array([0] + decay_events + [step])
  factors = jnp.array([1.0] + decay_factors + [decay_factors[-1]])
  index = jnp.sum(boundaries[1:] < step)
  m = jnp.take(factors, index + 1) - jnp.take(factors, index)
  n = jnp.take(boundaries, index + 1) - jnp.take(boundaries, index)
  a = m / jnp.clip(n, 1)
  interpolated_factor = (
      a * (step - jnp.take(boundaries, index)) + jnp.take(factors, index))
  return interpolated_factor


def linear_warmup_scheduler(step, warmup_steps, alpha=0.):
  """Gives a scaling factor based on scheduling with a Linear Warmup.

  Args:
    step: int; Current step.
    warmup_steps: int; How many steps to warm up for in the warmup schedule.
    alpha: float: The minimum value as a fraction of the initial value.

  Returns:
    Scaling factor applied to the learning rate on the given step.
  """
  if warmup_steps > 0:
    return jnp.minimum(1.0, alpha + step * (1.0 - alpha) / warmup_steps)
  else:
    return 1.0


def decay_every_scheduler(step, steps_per_decay, decay_factor):
  """Gives a scaling factor based on scheduling with a decay every n-steps.

  Args:
    step: int; Current step.
    steps_per_decay: int; How often to decay.
    decay_factor: float; The amount to decay.

  Returns:
    Scaling factor applied to the learning rate on the given step.
  """
  return decay_factor**(step // steps_per_decay)


def exponential_decay_scheduler(step, decay_steps, decay_rate, staircase=False):
  """Gives a scaling factor based on scheduling with an exponential decay.

  Args:
    step: int; Current step.
    decay_steps: int; Number of steps to decay over.
    decay_rate: float; Rate of exponential decay.
    staircase: bool; If True, use integer division in scale-computation.

  Returns:
    Scaling factor applied to the learning rate on the given step.
  """
  progress = step / float(decay_steps)
  if staircase:
    progress = jnp.floor(progress)
  return jnp.power(decay_rate, progress)


def cosine_decay_scheduler(step, steps_per_cycle, t_mul=1, m_mul=1., alpha=0.):
  """Gives a scaling factor based on scheduling with a cosine decay.

  Args:
    step: int; Current step.
    steps_per_cycle: int; Number of steps to reset the decay cycle.
    t_mul: int; Used to derive the number of iterations in the i-th period.
    m_mul: float; Used to derive the initial learning rate of the i-th period.
    alpha: float; The minimum value as a fraction of the initial value.

  Returns:
    Scaling factor applied to the learning rate on the given step.
  """
  if steps_per_cycle <= 0:
    raise ValueError(f'steps_per_cycle must be > 0. Got {steps_per_cycle}.')
  progress = step / float(steps_per_cycle)
  if t_mul == 1.0:
    i_restart = jnp.floor(progress)
    progress -= i_restart
  else:
    i_restart = jnp.floor(
        jnp.log(1.0 - progress * (1.0 - t_mul)) / jnp.log(t_mul))
    sum_r = (1.0 - t_mul**i_restart) / (1.0 - t_mul)
    progress = (progress - sum_r) / t_mul**i_restart
  m_fac = m_mul**i_restart
  cosine_decay = jnp.maximum(
      0.0, 0.5 * m_fac * (1.0 + jnp.cos(jnp.pi * (progress % 1.0))))
  return (1 - alpha) * cosine_decay + alpha


def compound_lr_scheduler(config):
  """Creates a learning rate scheduler by combining multiple factors.

  Interprets factors in the factors string which can consist of:
  * constant: interpreted as the constant value,
  * linear_warmup: interpreted as linear warmup until warmup_steps,
  * rsqrt_decay: divide by square root of max(step, warmup_steps)
  * decay_every: Every k steps decay the learning rate by decay_factor.
  * cosine_decay: Cyclic cosine decay.

  For instance, `config['factors'] = 'constant*linear_warmup'` combines the
  constant learning rate schedule with a linear warmup. This requires one to
  have the following configuration entries:
  config['warmup_steps'] and config['base_learning_rate'].

  Args:
    config: Relevant config based on the chosen factors.

  Returns:
    lr_fn: A function mapping global_step to lr.
  """

  ratio_factors = [n.strip() for n in config['factors'].split('*')]

  def lr_fn(step):
    """Step to learning rate function."""
    ratio = 1.0
    for name in ratio_factors:
      if name == 'constant':
        ratio *= config['base_learning_rate']
      elif name == 'polynomial':
        decay_steps = config['decay_steps']
        end_factor = config['end_factor']
        power = config['power']
        ratio *= polynomial_lr_scheduler(step, decay_steps, end_factor, power)
      elif name == 'piecewise_constant':
        decay_events = config['decay_events']
        decay_factors = config['decay_factors']
        ratio *= piecewise_constant_scheduler(step, decay_events, decay_factors)

      elif name == 'piecewise_linear':
        decay_events = config['decay_events']
        decay_factors = config['decay_factors']
        ratio *= piecewise_linear_scheduler(step, decay_events, decay_factors)

      elif name == 'linear_warmup':
        warmup_steps = config['warmup_steps']
        warmup_alpha = config.get('warmup_alpha', 0)
        ratio *= linear_warmup_scheduler(step, warmup_steps, warmup_alpha)

      elif name == 'rsqrt_decay':
        warmup_steps = config.get('warmup_steps', 0.)
        timescale = config.get('timescale', 10_000)
        shift = timescale - warmup_steps
        ratio *= jnp.where(warmup_steps < step,
                           jnp.sqrt(timescale) / jnp.sqrt(step + shift), 1.)

      elif name == 'decay_every':
        steps_per_decay = config['steps_per_decay']
        decay_factor = config['decay_factor']
        ratio *= decay_every_scheduler(step, steps_per_decay, decay_factor)

      elif name == 'exponential_decay':
        decay_steps = config['decay_steps']
        decay_rate = config['decay_rate']
        staircase = config.get('staircase', False)
        ratio *= exponential_decay_scheduler(
            step, decay_steps, decay_rate, staircase=staircase)

      elif name == 'cosine_decay':
        steps_per_cycle = config['steps_per_cycle']
        t_mul = config.get('t_mul', 1.)
        m_mul = config.get('m_mul', 1.)
        alpha = config.get('alpha', 0.0)
        warmup_steps = config.get('warmup_steps', 0.)
        adjusted_step = jnp.maximum(
            0.0, (step - (warmup_steps + config.get('start_decay_step', 0.))))
        total_steps = config.get('total_steps', steps_per_cycle)

        # We make the cos equal and subtract warmup steps for each cycle. If
        # there are fewer steps than warmup steps, cosine can be skipped.
        steps_per_cycle = steps_per_cycle - int(
            warmup_steps / (total_steps / steps_per_cycle))
        if steps_per_cycle > 0:
          ratio *= cosine_decay_scheduler(
              adjusted_step,
              steps_per_cycle,
              t_mul=t_mul,
              m_mul=m_mul,
              alpha=alpha)
      elif name == 'linear_decay':
        warmup_steps = config.get('warmup_steps', 0.)
        total_steps = config.get('total_steps')
        assert total_steps > warmup_steps, (
            'With linear decay, total_steps should be higher than warmup_steps.'
        )
        progress = jnp.maximum(0.0, (step - warmup_steps) /
                               float(total_steps - warmup_steps))
        ratio -= config.get('end_learning_rate', 0.)
        ratio *= jnp.maximum(1.0 - progress, 0.0)
        ratio += config.get('end_learning_rate', 0.)

      elif name == 'linear_cooldown':
        adjusted_step = jnp.maximum(step, config.get('warmup_steps', 0.))
        ratio *= jnp.minimum(1., (config.total_steps - adjusted_step) /
                             config.cooldown_steps)

      else:
        raise ValueError('Unknown factor %s.' % name)

    return jnp.asarray(ratio, dtype=jnp.float32)

  return lr_fn


lr_fn_dict = {
    'compound': compound_lr_scheduler,
}


def get_learning_rate_fn(config: ml_collections.ConfigDict):
  """Looks up for the learning rate scheduler and return lr_fn.

  Args:
    config: ConfigDict that has configuration ofthe learning rate function.

  Returns:
    An learning rate or a function learning_rate(step):  float ->
    {'learning_rate': float}, the step-dependent lr.

  """
  if 'base_learning_rate' not in config.lr_configs:
    raise ValueError(
        '`base_learning_rate` has to be defined in the lr_config.')
  if not config.lr_configs.base_learning_rate:
    # raise ValueError(  # raised for {0, False, None, [], (), {}}
    #   f'`base_learning_rate = {config.lr_configs.base_learning_rate}` is not '
    #   'allowed for training parameters. If your intention was to freeze '
    #   'parameters, use Scenic optax and `config.lr_configs = None` instead.')
    pass
    # Circumvent failing of config.lr_configs.base_learning_rate in {0, False,
    # None, [], (), {}} here as a short-term solution. This case is for now
    # handled in optax.make to handle edge cases.
  if 'learning_rate_schedule' in config.lr_configs:
    # A function that given the current step, returns the LR.
    return lr_fn_dict[config.lr_configs['learning_rate_schedule']](
        config.lr_configs)
  else:
    # LR as a scalar value.
    lr = jnp.asarray(config.lr_configs.base_learning_rate, dtype=jnp.float32)
    return lambda step: lr
