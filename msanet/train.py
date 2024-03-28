import os
import time

import torch

from engine import trainer

from options import get_arguments

if __name__ == "__main__":
    args = get_arguments()
    engine = trainer(args)

    print(f'using device: {engine.device}')

    project_dir = f'{args.output_dir}/{args.project_name}'
    ckpt_dir = f'{project_dir}/ckpt'


    best_mae = engine.best_mae
    ## Start Training ##
    # Main loop
    for _ in range(engine.epoch, engine.epoch+args.epochs):
        # always run evaluation first (to check what model has been loaded !!!)
        # if engine.epoch==0 or args.resume!='':
            # t1 = time.time()
            # r2, mae, eval_loss = engine.evaluation()
            # t2 = time.time()
            # # print the evaluation results
            # print('=======================================Evaluation=======================================')
            # print(f"[Epoch {engine.epoch}] R2: {r2:.4f}, mae: {mae:.4f}, time: {(t2-t1):.2f}s, best mae: {engine.best_mae:.4f}, at epoch: {engine.best_mae_epoch}")
            # print('=======================================Evaluation=======================================')

            # # print the evaluation results
            # print('=======================================Evaluation=======================================')
            # print(f"[Epoch {engine.epoch}] R2: {r2:.4f}, mae: {mae:.4f}, time: {(t2-t1):.2f}s, best eval loss: {engine.best_eval_loss:.4f}, at epoch: {engine.best_eval_loss_epoch}")
            # print('=======================================Evaluation=======================================')
        # train
        t1 = time.time()
        avg_loss = engine.train_one_epoch()
        t2 = time.time()
        lr = engine.optim.param_groups[0]['lr']
        print(f'[epoch {engine.epoch}][lr {lr:.7f}][{(t2-t1):.2f}s]')
        
        # evaluation
        if engine.epoch%args.eval_freq==0:
            t1 = time.time()
            r2, mae, eval_loss = engine.evaluation()
            t2 = time.time()
            # print the evaluation results
            print('=======================================Evaluation=======================================')
            print(f"[Epoch {engine.epoch}] R2: {r2:.4f}, mae: {mae:.4f}, time: {(t2-t1):.2f}s, best mae: {engine.best_mae:.4f}, at epoch: {engine.best_mae_epoch}")
            print('=======================================Evaluation=======================================')

        # visualization
        if engine.epoch%args.vis_freq==0:
            engine.visualization()

        # save model checkpoints
        if engine.epoch%args.save_freq==0:
            print(f'saving checkpoints @epoch {engine.epoch}')
            torch.save(engine.get_checkpoint(), f'{ckpt_dir}/epoch_{engine.epoch}.pth')

        torch.save(engine.best_model, f'{ckpt_dir}/epoch_{engine.best_mae_epoch:.4f}_mae_{engine.best_mae}.pth')
    print(f'Training process finished, best mae; {engine.best_mae:.4f}')
    torch.save(engine.best_model, f'{ckpt_dir}/best_mae.pth')
    engine.writer.close()