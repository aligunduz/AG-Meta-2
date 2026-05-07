import argparse
import os
import random
from collections import OrderedDict

import yaml
import torch
#from torch.cuda.amp import autocast, GradScaler
from torch import amp
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchviz import make_dot
import datasets
import models
import utils
import utils.optimizers as optimizers


class NullSummaryWriter(object):
  def add_scalar(self, *args, **kwargs):
    pass

  def add_scalars(self, *args, **kwargs):
    pass

  def flush(self):
    pass

  def close(self):
    pass


def main(config):
  random.seed(0)
  np.random.seed(0)
  torch.manual_seed(0)
  torch.cuda.manual_seed(0)
  # torch.backends.cudnn.deterministic = True
  # torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.benchmark = True
  ckpt_name = args.name
  if ckpt_name is None:
    ckpt_name = config['encoder']
    ckpt_name += '_' + config['dataset'].replace('meta-', '')
    ckpt_name += '_{}_way_{}_shot'.format(
      config['train']['n_way'], config['train']['n_shot'])
  if args.tag is not None:
    ckpt_name += '_' + args.tag

  ckpt_path = os.path.join('./save', ckpt_name)
  utils.ensure_path(ckpt_path)
  utils.set_log_path(ckpt_path)
  if config.get('use_tensorboard', True):
    writer = SummaryWriter(os.path.join(ckpt_path, 'tensorboard'))
  else:
    writer = NullSummaryWriter()
    utils.log('tensorboard: disabled')
  yaml.dump(config, open(os.path.join(ckpt_path, 'config.yaml'), 'w'))
  # Alignment log açık mı?
  log_alignment = config.get('log_alignment', False)

  # Pre-alignment loss kullanılacak mı?
  use_alignment_pre_loss = config.get('use_alignment_pre_loss', False)

  # Post-alignment loss kullanılacak mı?
  use_alignment_post_loss = config.get('use_alignment_post_loss', False)

  # Pre-alignment loss ağırlığı (eta)
  alignment_pre_weight = config.get('alignment_pre_weight', 0.0)

  # Post-alignment loss ağırlığı (eta)
  alignment_post_weight = config.get('alignment_post_weight', 0.0)
  # Modelden alignment çıktısı almamız gerekiyor mu?
  # Log açık olabilir, pre-loss açık olabilir, post-loss açık olabilir.
  need_alignment_outputs = (
      log_alignment or
      use_alignment_pre_loss or
      use_alignment_post_loss
  )

  use_gradient_transport = config.get('use_gradient_transport', False)
  task_gate_args = utils.config_task_gate_args(config)
  if task_gate_args.get('enabled', False) and not use_gradient_transport:
    utils.log('warning: task-conditioned gate is enabled but gradient transport is disabled')
  ##### Dataset #####

  # meta-train
  train_set = datasets.make(config['dataset'], **config['train'])
  utils.log('meta-train set: {} (x{}), {}'.format(
    train_set[0][0].shape, len(train_set), train_set.n_classes))
  train_loader = DataLoader(
    train_set, config['train']['n_episode'],
    collate_fn=datasets.collate_fn, num_workers=8, pin_memory=True,prefetch_factor=4,persistent_workers=True)

  # meta-val
  eval_val = False
  if config.get('val'):
    eval_val = True
    val_set = datasets.make(config['dataset'], **config['val'])
    utils.log('meta-val set: {} (x{}), {}'.format(
      val_set[0][0].shape, len(val_set), val_set.n_classes))
    val_loader = DataLoader(
      val_set, config['val']['n_episode'],
      collate_fn=datasets.collate_fn, num_workers=4, pin_memory=True,prefetch_factor=4,persistent_workers=True)
  
  ##### Model and Optimizer #####

  inner_args = utils.config_inner_args(config.get('inner_args'))
  if config.get('load'):
    ckpt = torch.load(config['load'])
    config['encoder'] = ckpt['encoder']
    config['encoder_args'] = ckpt['encoder_args']
    config['classifier'] = ckpt['classifier']
    config['classifier_args'] = ckpt['classifier_args']
    model = models.load(ckpt, load_clf=(not inner_args['reset_classifier']))
    optimizer, lr_scheduler = optimizers.load(ckpt, model.parameters())
    start_epoch = ckpt['training']['epoch'] + 1
    max_va = ckpt['training']['max_va']
  else:
    config['encoder_args'] = config.get('encoder_args') or dict()
    config['classifier_args'] = config.get('classifier_args') or dict()
    config['encoder_args']['bn_args']['n_episode'] = config['train']['n_episode']
    config['classifier_args']['n_way'] = config['train']['n_way']
    model = models.make(config['encoder'], config['encoder_args'],
                        config['classifier'], config['classifier_args'])
    optimizer, lr_scheduler = optimizers.make(
      config['optimizer'], model.parameters(), **config['optimizer_args'])
    start_epoch = 1
    max_va = 0.

  if args.efficient:
    model.go_efficient()

  if config.get('_parallel'):
    model = nn.DataParallel(model)

  utils.log('num params: {}'.format(utils.compute_n_params(model)))
  utils.log('gradient transport: {}'.format(
    'enabled' if use_gradient_transport else 'disabled'))
  utils.log('task-conditioned gate: {}'.format(
    task_gate_args if task_gate_args.get('enabled', False) else 'disabled'))
  timer_elapsed, timer_epoch = utils.Timer(), utils.Timer()

  scaler = amp.GradScaler('cuda')  # NEW (AMP scaler)
  ##### Training and evaluation #####
    
  # 'tl': meta-train loss
  # 'ta': meta-train accuracy
  # 'vl': meta-val loss
  # 'va': meta-val accuracy
  aves_keys = ['tl', 'ta', 'vl', 'va']
  if log_alignment:
      aves_keys += ['align_pre', 'align_post']
  if task_gate_args.get('enabled', False):
      aves_keys += ['eff_gate', 'eff_gate_min', 'eff_gate_max', 'task_signal']
      if task_gate_args.get('gamma_l2_weight', 0.0) > 0:
          aves_keys += ['gamma_l2']
  trlog = dict()
  for k in aves_keys:
    trlog[k] = []

  for epoch in range(start_epoch, config['epoch'] + 1):
    timer_epoch.start()
    aves = {k: utils.AverageMeter() for k in aves_keys}

    # meta-train
    model.train()
    writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
    np.random.seed(epoch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    did_viz = False
    for data in tqdm(train_loader, desc='meta-train', leave=False):
      x_shot, x_query, y_shot, y_query = data
      x_shot = x_shot.cuda(non_blocking=True)
      y_shot = y_shot.cuda(non_blocking=True)
      x_query = x_query.cuda(non_blocking=True)
      y_query = y_query.cuda(non_blocking=True)

      if inner_args['reset_classifier']:
        if config.get('_parallel'):
          model.module.reset_classifier()
        else:
          model.reset_classifier()

      optimizer.zero_grad(set_to_none=True)  # NEW (daha verimli)
      gamma_l2_loss = None
      with amp.autocast('cuda'):  # NEW
          # Alignment ile ilgili herhangi bir şey açıksa
          # modeli metrics dönecek şekilde çağırıyoruz.
          if need_alignment_outputs:
              logits, metrics = model(
                  x_shot,
                  x_query,
                  y_shot,
                  inner_args,
                  meta_train=True,
                  y_query=y_query,
                  return_metrics=True,
                  use_alignment_pre_loss=use_alignment_pre_loss,
                  use_alignment_post_loss=use_alignment_post_loss,
                  alignment_pre_weight=alignment_pre_weight,
                  alignment_post_weight=alignment_post_weight,
                  use_gradient_transport=use_gradient_transport,
                  task_gate_args=task_gate_args
              )
          else:
              logits = model(
                  x_shot,
                  x_query,
                  y_shot,
                  inner_args,
                  meta_train=True,
                  use_gradient_transport=use_gradient_transport,
                  task_gate_args=task_gate_args
              )
              metrics = None
          if task_gate_args.get('enabled', False):
              model_for_stats = model.module if config.get('_parallel') else model
              task_gate_stats = model_for_stats.get_task_gate_stats()
              if task_gate_stats is not None:
                  aves['eff_gate'].update(task_gate_stats['effective_gate_mean'], 1)
                  aves['eff_gate_min'].update(task_gate_stats['effective_gate_min'], 1)
                  aves['eff_gate_max'].update(task_gate_stats['effective_gate_max'], 1)
                  aves['task_signal'].update(task_gate_stats['task_signal_mean'], 1)
          if log_alignment and metrics is not None:
              if metrics['align_pre_mean'] is not None:
                  aves['align_pre'].update(metrics['align_pre_mean'], 1)
              if metrics['align_post_mean'] is not None:
                  aves['align_post'].update(metrics['align_post_mean'], 1)
          logits = logits.flatten(0, 1)
          labels = y_query.flatten()

          pred = torch.argmax(logits, dim=-1)
          acc = utils.compute_acc(pred, labels)
          loss = F.cross_entropy(logits, labels)
          # Toplam alignment cezasını burada ana loss'a ekliyoruz.
          # Başlangıçta sıfır kabul ediyoruz.
          align_loss_total = 0.0

          # Pre-alignment loss aktifse ekle
          if metrics is not None and metrics['align_pre_loss_mean'] is not None:
              align_loss_total = align_loss_total + metrics['align_pre_loss_mean']

          # Post-alignment loss aktifse ekle
          if metrics is not None and metrics['align_post_loss_mean'] is not None:
              align_loss_total = align_loss_total + metrics['align_post_loss_mean']

          # Final outer loss = query CE loss + alignment cezası
          loss = loss + align_loss_total

          if task_gate_args.get('enabled', False):
              gamma_l2_weight = task_gate_args.get('gamma_l2_weight', 0.0)
              if gamma_l2_weight > 0:
                  model_for_loss = model.module if config.get('_parallel') else model
                  gamma_l2_loss = gamma_l2_weight * model_for_loss.task_gate_gamma_l2()
                  loss = loss + gamma_l2_loss

      # if not did_viz:
      #     from torchviz import make_dot
      #     # AMP bloğunun DIŞINDA çağır
      #     dot = make_dot(loss, params=dict(model.named_parameters()))
      #     dot.save("maml_graph.dot")
      #     did_viz = True

      ##AGAG Burada theta'yı güncelliyoruz.
      ##AGAG backward ile varolan önceki güncellemeleri de hesaba katarak theta güncellenir.
      scaler.scale(loss).backward()  # NEW
      scaler.unscale_(optimizer)  # NEW (clip öncesi gerekli)
      for param in optimizer.param_groups[0]['params']:
          if param.grad is not None:
              nn.utils.clip_grad_value_(param, 10)
      scaler.step(optimizer)  # NEW
      scaler.update()  # NEW


      aves['tl'].update(loss.item(), 1)
      aves['ta'].update(acc, 1)
      if gamma_l2_loss is not None:
          aves['gamma_l2'].update(gamma_l2_loss.detach().item(), 1)
      

    # meta-val
    if eval_val:
      model.eval()
      np.random.seed(0)

      for data in tqdm(val_loader, desc='meta-val', leave=False):
        x_shot, x_query, y_shot, y_query = data
        x_shot, y_shot = x_shot.cuda(), y_shot.cuda()
        x_query, y_query = x_query.cuda(), y_query.cuda()

        if inner_args['reset_classifier']:
          if config.get('_parallel'):
            model.module.reset_classifier()
          else:
            model.reset_classifier()

        with torch.no_grad(), amp.autocast('cuda'):  # NEW (val’de de AMP aç)
            logits = model(
                x_shot,
                x_query,
                y_shot,
                inner_args,
                meta_train=False,
                use_gradient_transport=use_gradient_transport,
                task_gate_args=task_gate_args)
            logits = logits.flatten(0, 1)
            labels = y_query.flatten()

            pred = torch.argmax(logits, dim=-1)
            acc = utils.compute_acc(pred, labels)
            loss = F.cross_entropy(logits, labels)
        aves['vl'].update(loss.item(), 1)
        aves['va'].update(acc, 1)

    if lr_scheduler is not None:
      lr_scheduler.step()

    for k, avg in aves.items():
      aves[k] = avg.item()
      trlog[k].append(aves[k])

    gate_mean = None
    gamma_abs_mean = None
    if use_gradient_transport:
        model_for_log = model.module if config.get('_parallel') else model
        gates = model_for_log.get_gradient_transport_gates()
        gate_mean = sum(gates.values()) / len(gates)

        writer.add_scalar('gradient_transport/gate_mean', gate_mean, epoch)

        for gate_name, gate_value in gates.items():
            writer.add_scalar(f'gradient_transport/{gate_name}', gate_value, epoch)

        if task_gate_args.get('enabled', False):
            gammas = model_for_log.get_task_gate_gammas()
            gamma_mean = sum(gammas.values()) / len(gammas)
            gamma_abs_mean = sum(abs(v) for v in gammas.values()) / len(gammas)

            writer.add_scalar('task_gate/gamma_mean', gamma_mean, epoch)
            writer.add_scalar('task_gate/gamma_abs_mean', gamma_abs_mean, epoch)
            writer.add_scalar('task_gate/effective_gate_mean', aves['eff_gate'], epoch)
            writer.add_scalar('task_gate/effective_gate_min', aves['eff_gate_min'], epoch)
            writer.add_scalar('task_gate/effective_gate_max', aves['eff_gate_max'], epoch)
            writer.add_scalar('task_gate/task_signal_mean', aves['task_signal'], epoch)
            if 'gamma_l2' in aves:
                writer.add_scalar('task_gate/gamma_l2_loss', aves['gamma_l2'], epoch)

            for gamma_name, gamma_value in gammas.items():
                writer.add_scalar(f'task_gate/{gamma_name}', gamma_value, epoch)

    t_epoch = utils.time_str(timer_epoch.end())
    t_elapsed = utils.time_str(timer_elapsed.end())
    t_estimate = utils.time_str(timer_elapsed.end() / 
      (epoch - start_epoch + 1) * (config['epoch'] - start_epoch + 1))

    # formats output
    log_str = 'epoch {}, meta-train {:.4f}|{:.4f}'.format(
      str(epoch), aves['tl'], aves['ta'])
    if use_gradient_transport and gate_mean is not None:
        log_str += ', gate_mean {:.4f}'.format(gate_mean)
    if gamma_abs_mean is not None:
        log_str += ', gamma_abs_mean {:.4f}'.format(gamma_abs_mean)
    if task_gate_args.get('enabled', False):
        log_str += ', eff_gate {:.4f}|{:.4f}|{:.4f}, signal {:.4f}'.format(
            aves['eff_gate'],
            aves['eff_gate_min'],
            aves['eff_gate_max'],
            aves['task_signal'])
        if 'gamma_l2' in aves:
            log_str += ', gamma_l2 {:.6f}'.format(aves['gamma_l2'])
    writer.add_scalars('loss', {'meta-train': aves['tl']}, epoch)
    writer.add_scalars('acc', {'meta-train': aves['ta']}, epoch)
    if log_alignment:
        writer.add_scalar('alignment/align_pre', aves['align_pre'], epoch)
        writer.add_scalar('alignment/align_post', aves['align_post'], epoch)
    if eval_val:
      if log_alignment:
          log_str += ', meta-val {:.4f}|{:.4f}, align {:.4f}|{:.4f}'.format(aves['vl'], aves['va'], aves['align_pre'], aves['align_post'])
      else:
          log_str += ', meta-val {:.4f}|{:.4f}'.format(aves['vl'], aves['va'])
      writer.add_scalars('loss', {'meta-val': aves['vl']}, epoch)
      writer.add_scalars('acc', {'meta-val': aves['va']}, epoch)

    log_str += ', {} {}/{}'.format(t_epoch, t_elapsed, t_estimate)
    utils.log(log_str)

    # saves model and meta-data
    if config.get('_parallel'):
      model_ = model.module
    else:
      model_ = model

    training = {
      'epoch': epoch,
      'max_va': max(max_va, aves['va']),

      'optimizer': config['optimizer'],
      'optimizer_args': config['optimizer_args'],
      'optimizer_state_dict': optimizer.state_dict(),
      'lr_scheduler_state_dict': lr_scheduler.state_dict() 
        if lr_scheduler is not None else None,
    }
    ckpt = {
      'file': __file__,
      'config': config,

      'encoder': config['encoder'],
      'encoder_args': config['encoder_args'],
      'encoder_state_dict': model_.encoder.state_dict(),

      'classifier': config['classifier'],
      'classifier_args': config['classifier_args'],
      'classifier_state_dict': model_.classifier.state_dict(),
      'gradient_transport_state_dict': model_.gradient_transport_logits.state_dict(),
      'task_gate_gamma_state_dict': model_.task_gate_gammas.state_dict(),
      'training': training,
    }

    # 'epoch-last.pth': saved at the latest epoch
    # 'max-va.pth': saved when validation accuracy is at its maximum
    torch.save(ckpt, os.path.join(ckpt_path, 'epoch-last.pth'))
    torch.save(trlog, os.path.join(ckpt_path, 'trlog.pth'))

    if aves['va'] > max_va:
      max_va = aves['va']
      torch.save(ckpt, os.path.join(ckpt_path, 'max-va.pth'))

    writer.flush()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', 
                      help='configuration file')
  parser.add_argument('--name', 
                      help='model name', 
                      type=str, default=None)
  parser.add_argument('--tag', 
                      help='auxiliary information', 
                      type=str, default=None)
  parser.add_argument('--gpu', 
                      help='gpu device number', 
                      type=str, default='0')
  parser.add_argument('--efficient', 
                      help='if True, enables gradient checkpointing',
                      action='store_true')
  args = parser.parse_args()
  config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

  if len(args.gpu.split(',')) > 1:
    config['_parallel'] = True
    config['_gpu'] = args.gpu

    # ✅ GPU varsa kullan, yoksa CPU'da devam et
    if torch.cuda.is_available() and args.gpu != '-1':
        utils.set_gpu(args.gpu)
    else:
        print("⚠️ CUDA not available — running on CPU mode.")
  main(config)
