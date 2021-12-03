import time
import datetime
import torch

import torch
from benchmark.data import build_xbm_dataloader
from benchmark.models.xbmv1 import XBMv1
from benchmark.models.xbmv2 import XBMv2

from benchmark.utils.metric_logger import MetricLogger
from benchmark.utils.extract_fn import extract_batch_feature


def flush_log(log, writer, step):
    for key, value in log.items():
        writer.add_scalar(key, value, step)
    writer.flush()
    del log


def do_train(cfg, dataloader, model, criterion, optimizer, scheduler,
             evaluators, checkpointer, writer, device, logger):

    max_step = cfg.SOLVER.MAX_STEP
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    test_period = cfg.TEST.TEST_PERIOD
    
    if not isinstance(evaluators, list):
        evaluators = [evaluators]

    logger.info("start training")

    log = {}
    for evaluator in evaluators:
       msg, res = evaluator.evaluate(model, device)
       logger.info(msg)
       log.update(res)
    flush_log(log, writer, 0)

    time_meters = MetricLogger("  ")
    loss_meters = MetricLogger("  ")

    if cfg.XBM.ENABLE:
        if cfg.XBM.VERSION == "v1":
            xbm = XBMv1(cfg)
        elif cfg.XBM.VERSION == "v2":
            xbm = XBMv2(cfg)
            xbm.init(model, build_xbm_dataloader(cfg), device)
        else:
            raise ValueError(f"undefined XBM version: {cfg.XBM.VERSION}")
                 
    start_training_time = time.time()
    end = time.time()
    for step, batch in enumerate(dataloader, 1):
        log = {}
        data_time = time.time() - end

        anchor_inputs = batch["input"]
        anchor_indices = batch["idx"]
        anchor_labels = batch["label"]

        neighbor_inputs = batch["neighbor_inputs"] 
        neighbor_indices = batch["neighbor_ind"]
        neighbor_labels = batch["neighbor_labels"]

        # cat
        inputs = torch.cat((anchor_inputs, neighbor_inputs), dim=0)
        indices = torch.cat((anchor_indices, neighbor_indices), dim=0)
        labels = torch.cat((anchor_labels, neighbor_labels), dim=0).long()

        if cfg.NOAUG.ENABLE:
            noaug_anchor_inputs = batch["noaug_input"].to(device)
            noaug_neighbor_inputs = batch["noaug_neighbor_inputs"].to(device)
            noaug_inputs = torch.cat((noaug_anchor_inputs, noaug_neighbor_inputs), dim=0)
            model = model.eval()
            noaug_features = extract_batch_feature(model, noaug_inputs, cfg.NOAUG.SCALES)

        model.train()
        inputs = inputs.to(device)
        features, *_ = model(inputs)

        if cfg.SELECTION.ENABLE:
            # for an anchor, select its pesudo positive neighbors
            # compute the similarity between an anchor and its neighbors
            if cfg.SELECTION.USE_NOAUG_SIMILARITY:
                global_features = noaug_features
            else:    
                global_features = features.detach().cpu()

            anchor_features, neighbor_features = \
                torch.split(global_features, [len(anchor_inputs), len(neighbor_inputs)], dim=0)
            anchor_features = anchor_features.unsqueeze(dim=1)
            neighbor_features = neighbor_features.reshape(len(anchor_features), -1, global_features.shape[1])
            neighbor_features = torch.transpose(neighbor_features, 1, 2)
            batch_similarities = torch.matmul(anchor_features, neighbor_features).squeeze(dim=1)

            # for each anchor, split its neighbors into positive/negative
            knn_cnt = 0
            num_anchors = len(anchor_inputs)
            for anchor_label, similarities in zip(anchor_labels, batch_similarities):
                # for an anchor, the init positive neighbors in the batch 
                init_positive_ind = \
                    torch.nonzero(anchor_label == neighbor_labels, as_tuple=True)[0] + num_anchors
                if len(init_positive_ind) == 0:
                    # no neighbors for this anchor, seems impossible
                    continue

                # True means positive
                if cfg.SELECTION.TYPE == "fixed":
                    mask = similarities >= cfg.SELECTION.THRESH
                elif cfg.SELECTION.TYPE == "dynamic":
                    mask = (similarities / torch.max(similarities)) >= cfg.SELECTION.THRESH
                else:
                    raise RuntimeError(f"invalid selection type: {cfg.SELECTION.TYPE}")

                if cfg.SELECTION.ALWAYS_TRUE > 0:
                    mask[:cfg.SELECTION.ALWAYS_TRUE] = True

                knn_cnt += sum(mask)
                pred_negative_ind = init_positive_ind[~mask]
                # set labels for predicted negative
                labels[pred_negative_ind] = -anchor_label

        labels = labels.to(device)
        batch_loss, _log = criterion(features, labels, features, labels)
        log["batch/loss"] = batch_loss.detach().item()
        log.update({f"batch/{k}": v for k, v in _log.items()})

        if cfg.XBM.ENABLE and step > cfg.XBM.START_STEP:
            xbm_features = features
            if cfg.XBM.VERSION == "v2":
                noaug_features = noaug_features
            else:
                noaug_features = None
            
            xbm.dequeue_and_enqueue(indices=indices, features=xbm_features, noaug_features=noaug_features)
        
        if cfg.XBM.ENABLE and step > cfg.XBM.GET_STEP:
            memory_features, memory_labels, _log = xbm.get(
                size=cfg.XBM.GET_SIZE, indices=indices, labels=labels
            )

            log.update({f"xbm{cfg.XBM.VERSION}" + f"_{k}" if "/" in k else f"/{k}": v \
                for k, v in _log.items()})

            memory_features = memory_features.to(device)
            memory_labels = memory_labels.to(device)

            memory_loss, _log = criterion(features, labels, memory_features, memory_labels)
            log["memory/loss"] = memory_loss.detach().item()
            log.update({f"memory/{k}": v for k, v in _log.items()})
            loss = batch_loss + cfg.XBM.MEMORY_LOSS_WEIGHT * memory_loss
        else:
            memory_loss = torch.tensor(0.0, requires_grad=True, dtype=torch.float32)
            loss = batch_loss

        log["loss"] = loss.detach().item()
        log["lr"] = optimizer.param_groups[0]["lr"]

        optimizer.zero_grad()
        loss.backward()

        # for log purpose
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.SOLVER.GRAD_MAX_NORM, norm_type=2.0)
        log["grad_norm"] = grad_norm

        optimizer.step()
        scheduler.step()
        
        batch_time = time.time() - end
        end = time.time()
        time_meters.update(data_time=data_time, batch_time=batch_time)
        loss_meters.update(batch_loss=batch_loss)
        loss_meters.update(memory_loss=memory_loss)
        loss_meters.update(loss=loss)
        eta_seconds = (max_step - step) * time_meters.batch_time.avg
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if step % 20 == 0 or step == max_step:
            lr_msg = optimizer.param_groups[0]["lr"]
            loss_msg = str(loss_meters)
            time_msg = str(time_meters)
            msg = f"eta: {eta_string}  step: {step}  {loss_msg}  {time_msg}  lr: {lr_msg:.0e}"
            logger.info(msg)
    
        if step % checkpoint_period == 0:
            checkpointer.save(f"model_{step:07d}")
        if step == max_step:
            checkpointer.save(f"model_final")
        if step % test_period == 0:
            for evaluator in evaluators:
                msg, res = evaluator.evaluate(model, device)
                logger.info(msg)
                log.update(res)

        flush_log(log, writer, step)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(f"total training time: {total_time_str} " + \
        f"({total_training_time / max_step:.4f} s / it)")
