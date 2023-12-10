import logging

import torch
from tqdm import tqdm

from open_clip import get_input_dtype, get_tokenizer, build_zero_shot_classifier, \
    IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES
from .precision import get_autocast
from open_clip import tokenize
import torch.nn.functional as F
import torchmetrics
import numpy as np

import pdb
from .pjdataset_zeroshot_data import pj_classnames
from .voc_zeroshot_data import voc_classnames
from .coco_zeroshot_data import coco_classnames
from sklearn.metrics import average_precision_score

def zero_shot_classifier(model, classnames, templates, args):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            # print(f'classname: {classname}')
            texts = [template(classname) for template in templates]  # format with class
            texts = tokenize(texts).to(args.device)  # tokenize
            if args.distributed and not args.horovod:
                class_embeddings = model.module.encode_text(texts)
            else:
                class_embeddings = model.encode_text(texts)
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
            # print(f'zeroshot_weights: {zeroshot_weights}')
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(args.device)
        # print(zeroshot_weights.shape)
    return zeroshot_weights


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def run(model, classifier, dataloader, args):
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)
    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for images, target in tqdm(dataloader, unit_scale=args.batch_size):
            images = images.to(device=args.device, dtype=input_dtype)
            target = target.to(args.device)
            with autocast():
                # predict
                output = model(image=images)
                image_features = output['image_features'] if isinstance(output, dict) else output[0]
                logits = 100. * image_features @ classifier

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = (top1 / n)
    top5 = (top5 / n)
    return top1, top5

def run_multilabels(model, classifier, dataloader, args):
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)
    num_labels = 0
    if args.pj_dataset_val is not None:
        num_labels = 10
    if args.voc_val is not None:
        num_labels = 20
    if args.coco_val is not None:
        num_labels = 80
    macc = torchmetrics.classification.MultilabelAccuracy(num_labels=num_labels).to(args.device)
    probs = []
    targets = []
    with torch.no_grad():
        mAP, mAcc, n = 0., 0., 0.
        for images, target in tqdm(dataloader, unit_scale=args.batch_size):
            images = images.to(device=args.device, dtype=input_dtype)
            target = target.to(args.device)
            with autocast():
                # predict
                logits = model(images, classifier)
                # image_features = output['image_features'] if isinstance(output, dict) else output[0]
                # logit_scale = output['logit_scale'] if isinstance(output, dict) else output[2]
                # logits = 100. * image_features @ classifier / logit_scale

            # FIXME: a naive implementation
            # pdb.set_trace()
            pred = torch.zeros_like(logits, dtype=int)
            prob = torch.sigmoid(logits)
            pred = pred.to(args.device)
            pred[torch.arange(pred.shape[0]) ,torch.argmax(logits, 1)] = 1
            pred[prob > 0.56] = 1
            probs.append(prob)
            targets.append(target)
            acc = macc(pred, target)
            n += images.size(0)
    
    probs = torch.cat(probs).to('cpu').numpy()
    targets = torch.cat(targets).to('cpu').numpy()
    # pdb.set_trace()
    aps = []
    for i in range(targets.shape[1]):
        ap = average_precision_score(targets[:, i], probs[:, i])
        aps.append(ap)
    mAP = np.mean(aps)
    
    mAcc = macc.compute()
    # print(f'mAcc: {mAcc}')
    return mAP, mAcc.item()


def zero_shot_eval(model, data, epoch,  args, classifier=None, tokenizer=None):
    if 'imagenet-val' not in data and 'imagenet-v2' not in data and 'pj_dataset_val' not in data and 'voc_val' not in data and 'coco_val' not in data:
        return {}
    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}
    if args.distributed and not args.horovod:
        model = model.module

    logging.info('Starting zero-shot.')
    if tokenizer is None:
        tokenizer = get_tokenizer(args.model)

    logging.info('Building zero-shot classifier')
    # autocast = get_autocast(args.precision)
    # with autocast():
    #     classifier = build_zero_shot_classifier(
    #         model,
    #         tokenizer=tokenizer,
    #         classnames=IMAGENET_CLASSNAMES,
    #         templates=OPENAI_IMAGENET_TEMPLATES,
    #         num_classes_per_batch=10,
    #         device=args.device,
    #         use_tqdm=True,
    #     )

    if classifier is None:
        if 'pj_dataset_val' in data:
            classifier = zero_shot_classifier(model, pj_classnames, OPENAI_IMAGENET_TEMPLATES, args)
        elif 'voc_val' in data:
            classifier = zero_shot_classifier(model, voc_classnames, OPENAI_IMAGENET_TEMPLATES, args)
        elif 'coco_val' in data:
            classifier = zero_shot_classifier(model, coco_classnames, OPENAI_IMAGENET_TEMPLATES, args)
        else:
            classifier = zero_shot_classifier(model, IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES, args)


    logging.info('Using classifier')
    results = {}
    if 'imagenet-val' in data:
        top1, top5 = run(model, classifier, data['imagenet-val'].dataloader, args)
        results['imagenet-zeroshot-val-top1'] = top1
        results['imagenet-zeroshot-val-top5'] = top5
    if 'imagenet-v2' in data:
        top1, top5 = run(model, classifier, data['imagenet-v2'].dataloader, args)
        results['imagenetv2-zeroshot-val-top1'] = top1
        results['imagenetv2-zeroshot-val-top5'] = top5
    if 'pj_dataset_val' in data:
        mAP, mAcc = run_multilabels(model, classifier, data['pj_dataset_val'].dataloader, args)
        results['pj_dataset_val-mAP'] = mAP
        results['pj_dataset_val-mAcc'] = mAcc
    if 'voc_val' in data:
        mAP, mAcc = run_multilabels(model, classifier, data['voc_val'].dataloader, args)
        results['voc_val-mAP'] = mAP
        results['voc_val-mAcc'] = mAcc
    if 'coco_val' in data:
        mAP, mAcc = run_multilabels(model, classifier, data['coco_val'].dataloader, args)
        results['coco_val-mAP'] = mAP
        results['coco_val-mAcc'] = mAcc
    logging.info('Finished zero-shot.')

    return results
