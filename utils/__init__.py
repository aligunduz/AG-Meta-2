import os
import shutil
import time
import torch

import numpy as np
import scipy.stats as stats


_log_path = None

def set_log_path(path):
  global _log_path
  _log_path = path


def log(obj, filename='log.txt'):
  print(obj)
  if _log_path is not None:
    with open(os.path.join(_log_path, filename), 'a') as f:
      print(obj, file=f)


class AverageMeter(object):
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0.
    self.avg = 0.
    self.sum = 0.
    self.count = 0.

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

  def item(self):
    return self.avg


class Timer(object):
  def __init__(self):
    self.start()

  def start(self):
    self.v = time.time()

  def end(self):
    return time.time() - self.v


def set_gpu(gpu: str):
    """Safely set GPU device if available, otherwise stay on CPU."""
    if not torch.cuda.is_available() or gpu == '-1':
        print("⚠️ CUDA not available or GPU disabled, running on CPU.")
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        return

    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        torch.cuda.set_device(0)
        print(f"✅ Using CUDA device(s): {gpu}")
    except Exception as e:
        print(f"⚠️ Could not set CUDA device, fallback to CPU. ({e})")
        os.environ['CUDA_VISIBLE_DEVICES'] = ''


def ensure_path(path, remove=True):
  basename = os.path.basename(path.rstrip('/'))
  if os.path.exists(path):
    if remove and (basename.startswith('_')
      or input('{} exists, remove? ([y]/n): '.format(path)) != 'n'):
      shutil.rmtree(path)
      os.makedirs(path)
  else:
    os.makedirs(path)


def time_str(t):
  if t >= 3600:
    return '{:.1f}h'.format(t / 3600)
  if t >= 60:
    return '{:.1f}m'.format(t / 60)
  return '{:.1f}s'.format(t)


def compute_acc(pred, label, reduction='mean'):
  result = (pred == label).float()
  if reduction == 'none':
    return result.detach()
  elif reduction == 'mean':
    return result.mean().item()


def compute_n_params(model, return_str=True):
  n_params = 0
  for p in model.parameters():
    n_params += p.numel()
  if return_str:
    if n_params >= 1e6:
      return '{:.1f}M'.format(n_params / 1e6)
    else:
      return '{:.1f}K'.format(n_params / 1e3)
  else:
    return n_params


def mean_confidence_interval(data, confidence=0.95):
  a = 1.0 * np.array(data)
  stderr = stats.sem(a)
  h = stderr * stats.t.ppf((1 + confidence) / 2., len(a) - 1)
  return h


def config_inner_args(inner_args):
  if inner_args is None: 
    inner_args = dict()

  inner_args['reset_classifier'] = inner_args.get('reset_classifier') or False
  inner_args['n_step'] = inner_args.get('n_step') or 5
  inner_args['encoder_lr'] = inner_args.get('encoder_lr') or 0.01
  inner_args['classifier_lr'] = inner_args.get('classifier_lr') or 0.01
  inner_args['momentum'] = inner_args.get('momentum') or 0.
  inner_args['weight_decay'] = inner_args.get('weight_decay') or 0.
  inner_args['first_order'] = inner_args.get('first_order') or False
  inner_args['frozen'] = inner_args.get('frozen') or []

  return inner_args


def config_task_gate_args(config):
  task_gate_args = dict(config.get('task_gate_args') or {})

  mode = config.get('gradient_transport_mode')
  if mode is not None:
    if mode == 'scalar':
      task_gate_args['enabled'] = False
    elif mode == 'task_conditioned_gate_norm':
      task_gate_args['enabled'] = True
      task_gate_args['signal'] = 'grad_norm'
    else:
      raise ValueError('invalid gradient_transport_mode: {}'.format(mode))

  if 'use_task_conditioned_gate' in config:
    task_gate_args['enabled'] = config['use_task_conditioned_gate']
  task_gate_args['enabled'] = task_gate_args.get('enabled', False)

  if 'task_gate_signal' in config:
    task_gate_args['signal'] = config['task_gate_signal']
  task_gate_args['signal'] = task_gate_args.get('signal', 'grad_norm')

  if 'task_gate_detach_signal' in config:
    task_gate_args['detach_signal'] = config['task_gate_detach_signal']
  task_gate_args['detach_signal'] = task_gate_args.get('detach_signal', True)

  if 'task_gate_normalize_by_numel' in config:
    task_gate_args['normalize_by_numel'] = \
      config['task_gate_normalize_by_numel']
  task_gate_args['normalize_by_numel'] = task_gate_args.get(
    'normalize_by_numel', True)

  if 'task_gate_gamma_scale' in config:
    task_gate_args['gamma_scale'] = config['task_gate_gamma_scale']
  task_gate_args['gamma_scale'] = task_gate_args.get('gamma_scale', 1.0)

  if 'task_gate_gamma_l2_weight' in config:
    task_gate_args['gamma_l2_weight'] = \
      config['task_gate_gamma_l2_weight']
  task_gate_args['gamma_l2_weight'] = task_gate_args.get(
    'gamma_l2_weight', 0.0)

  if 'task_gate_collect_stats' in config:
    task_gate_args['collect_stats'] = config['task_gate_collect_stats']
  task_gate_args['collect_stats'] = task_gate_args.get('collect_stats', True)

  return task_gate_args
