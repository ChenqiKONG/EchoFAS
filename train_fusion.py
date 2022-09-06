from __future__ import print_function, division
import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from arch.fusion import Fusion_model
from tqdm import tqdm
import random
import pandas as pd
import  torch.nn.functional as F
from utils.dataloader_fusion import Intra_fusion_datareader, my_transforms
from utils.metrics import get_metrics

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device: ', device)

def parse_args():
    parser = argparse.ArgumentParser(description='chenqi_echofas')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--save_model_epoch', default=1, type=int)
    parser.add_argument('--disp_step', default=200, type=int)
    parser.add_argument('--warm_start_epoch', default=0, type=int)

    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--batch_size_train', default=1024, type=int)
    parser.add_argument('--batch_size_val', default=1024, type=int)
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--weight_decay', default=0.00001, type=float)
    parser.add_argument('--thre', default=0.5, type=float)

    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--save_root', default='./Training_results/Cross/Cross_5id/fusion/', type=str)
    parser.add_argument('--root_path', default='./', type=str)
    
    parser.add_argument('--model_name', default="note", type=str)
    parser.add_argument('--train_csv', default="./csv_files/cross_5id/train/cross_5id_note_train.csv", type=str)
    parser.add_argument('--val_csv', default="./csv_files/cross_5id/val/cross_5id_note_val.csv", type=str)
    parser.add_argument('--test_csv1', default="./csv_files/cross_5id/test/cross_5id_id_25_note_test.csv", type=str)
    parser.add_argument('--test_csv2', default="./csv_files/cross_5id/test/cross_5id_id_26_note_test.csv", type=str)
    parser.add_argument('--test_csv3', default="./csv_files/cross_5id/test/cross_5id_id_27_note_test.csv", type=str)
    parser.add_argument('--test_csv4', default="./csv_files/cross_5id/test/cross_5id_id_28_note_test.csv", type=str)
    parser.add_argument('--test_csv5', default="./csv_files/cross_5id/test/cross_5id_id_29_note_test.csv", type=str)
    return parser.parse_args()

def fix_seed(seed):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def Validation(model, dataloader, args, thre, facenum):
    print('Validating...')
    model.eval()
    length = len(dataloader)
    batch_val_losses = []
    GT = np.zeros((facenum,), np.int)
    PRED = np.zeros((facenum,), np.float)
    for num, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            ffts = data['ffts'].to(device)
            sepcts = data['spects'].to(device)
            labels = data['labels'].to(device)
            logits = model(ffts, sepcts)
            val_loss = F.cross_entropy(logits, labels)
            pred_score = torch.nn.functional.softmax(logits, dim=1)
            index = torch.max(pred_score, 1)[1]
            batch_val_losses.append(val_loss.item())
            if num != length-1:
                GT[num*args.batch_size_val:(num+1)*args.batch_size_val] = labels.cpu().numpy()
                PRED[num*args.batch_size_val:(num+1)*args.batch_size_val] = pred_score[:,1].cpu().numpy()
            else:
                GT[(length-1)*args.batch_size_val:] = labels.cpu().numpy()
                PRED[(length-1)*args.batch_size_val:] = pred_score[:,1].cpu().numpy()
                
    acc, auc, hter, eer = get_metrics(PRED, GT, thre)        
    avg_val_loss = round(sum(batch_val_losses) / (len(batch_val_losses)), 5)
    return avg_val_loss, auc, hter, eer, acc

def train(args, model):
    avg_train_loss_list = np.array([])
    train_dataset = Intra_fusion_datareader(csv_file=args.train_csv, transform=my_transforms())
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size_train, shuffle=True, num_workers=args.num_workers, drop_last=False)
    val_dataset = Intra_fusion_datareader(csv_file=args.val_csv, transform=my_transforms())
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size_val, shuffle=True, num_workers=args.num_workers, drop_last=False)

    test_dataset1 = Intra_fusion_datareader(csv_file=args.test_csv1, transform=my_transforms())
    test_dataloader1 = DataLoader(test_dataset1, batch_size=args.batch_size_val, shuffle=True, num_workers=args.num_workers, drop_last=False)
    test_dataset2 = Intra_fusion_datareader(csv_file=args.test_csv2, transform=my_transforms())
    test_dataloader2 = DataLoader(test_dataset2, batch_size=args.batch_size_val, shuffle=True, num_workers=args.num_workers, drop_last=False)
    test_dataset3 = Intra_fusion_datareader(csv_file=args.test_csv3, transform=my_transforms())
    test_dataloader3 = DataLoader(test_dataset3, batch_size=args.batch_size_val, shuffle=True, num_workers=args.num_workers, drop_last=False)
    test_dataset4 = Intra_fusion_datareader(csv_file=args.test_csv4, transform=my_transforms())
    test_dataloader4 = DataLoader(test_dataset4, batch_size=args.batch_size_val, shuffle=True, num_workers=args.num_workers, drop_last=False)
    test_dataset5 = Intra_fusion_datareader(csv_file=args.test_csv5, transform=my_transforms())
    test_dataloader5 = DataLoader(test_dataset5, batch_size=args.batch_size_val, shuffle=True, num_workers=args.num_workers, drop_last=False)
    Num_val_ffts = len(pd.read_csv(args.val_csv, header=None))
    Num_test_ffts1 = len(pd.read_csv(args.test_csv1, header=None))
    Num_test_ffts2 = len(pd.read_csv(args.test_csv2, header=None))
    Num_test_ffts3 = len(pd.read_csv(args.test_csv3, header=None))
    Num_test_ffts4 = len(pd.read_csv(args.test_csv4, header=None))
    Num_test_ffts5 = len(pd.read_csv(args.test_csv5, header=None))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # result folder
    res_folder_name = args.save_root + args.model_name
    if not os.path.exists(res_folder_name):
        os.makedirs(res_folder_name)
        os.mkdir(res_folder_name + '/ckpt/')
    else:
        print("WARNING: RESULT PATH ALREADY EXISTED -> " + res_folder_name)
    print('find models here: ', res_folder_name)
    writer = SummaryWriter(res_folder_name)
    f1 = open(res_folder_name + "/training_log.csv", 'a+')

    # training
    steps_per_epoch = len(train_dataloader)
    Best_AUC = 0.0 
    
    for epoch in range(args.warm_start_epoch, args.epochs):
        batch_train_losses = []
        step_loss = np.zeros(steps_per_epoch, dtype=np.float)

        # scheduler.step()
        for step, data in enumerate(tqdm(train_dataloader)):
            model.train()
            optimizer.zero_grad()
            ffts = data['ffts'].to(device)
            sepcts = data['spects'].to(device)
            labels = data['labels'].to(device)
            logits = model(ffts, sepcts)
            loss = F.cross_entropy(logits, labels)

            step_loss[step] = loss
            batch_train_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            Global_step = epoch * steps_per_epoch + (step + 1) 

        if (epoch+1) % args.save_model_epoch == 0:
            now_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            avg_train_loss = np.mean(step_loss[(step + 1) - args.disp_step: (step + 1)])
            avg_train_loss_list = np.append(avg_train_loss_list, avg_train_loss)
            log_msg = '[%s] Epoch: %d/%d | 1/10 average epoch loss: %f' % (
                now_time, epoch + 1, args.epochs, avg_train_loss)
            print('\n', log_msg)
            f1.write(log_msg)
            f1.write('\n')
  
            # validation
            Avg_val_loss, AUC, HTER, EER, ACC = Validation(model, val_dataloader, args, args.thre, Num_val_ffts)
            val_msg = '[%s] Epoch: %d/%d | Global_step: %d | average validation loss: %f | AUC: %f | HTER: %f | EER: %f | ACC: %f' % (
                                                                                        now_time, epoch + 1, args.epochs, Global_step, Avg_val_loss, AUC, HTER, EER, ACC)
            print('\n', val_msg)
            f1.write(val_msg)
            f1.write('\n')

            if AUC > Best_AUC:
                Best_AUC = AUC
                torch.save(model.state_dict(), res_folder_name + '/ckpt/' + 'best.pth')
                np.save(res_folder_name + '/avg_train_loss_list.np', avg_train_loss_list)
                cur_learning_rate = [param_group['lr'] for param_group in optimizer.param_groups]
                print('Saved model. lr %f' % cur_learning_rate[0])
                f1.write('Saved model. lr %f' % cur_learning_rate[0])
                f1.write('\n')

                Avg_test_loss, AUC1, HTER1, EER1, ACC1 = Validation(model, test_dataloader1, args, args.thre, Num_test_ffts1)
                test_msg = '[%s] Epoch: %d/%d | Global_step: %d | average testing loss: %f | id: %s | AUC: %f | HTER: %f | EER: %f | ACC: %f' % (
                                                                                        now_time, epoch + 1, args.epochs, Global_step, Avg_test_loss, 'id_25', AUC1, HTER1, EER1, ACC1)
                print('\n', test_msg)
                f1.write(test_msg)
                f1.write('\n')

                Avg_test_loss, AUC2, HTER2, EER2, ACC2 = Validation(model, test_dataloader2, args, args.thre, Num_test_ffts2)
                test_msg = '[%s] Epoch: %d/%d | Global_step: %d | average testing loss: %f | id: %s | AUC: %f | HTER: %f | EER: %f | ACC: %f' % (
                                                                                        now_time, epoch + 1, args.epochs, Global_step, Avg_test_loss, 'id_26', AUC2, HTER2, EER2, ACC2)
                print('\n', test_msg)
                f1.write(test_msg)
                f1.write('\n')

                Avg_test_loss, AUC3, HTER3, EER3, ACC3 = Validation(model, test_dataloader3, args, args.thre, Num_test_ffts3)
                test_msg = '[%s] Epoch: %d/%d | Global_step: %d | average testing loss: %f | id: %s | AUC: %f | HTER: %f | EER: %f | ACC: %f' % (
                                                                                        now_time, epoch + 1, args.epochs, Global_step, Avg_test_loss, 'id_27', AUC3, HTER3, EER3, ACC3)
                print('\n', test_msg)
                f1.write(test_msg)
                f1.write('\n')

                Avg_test_loss, AUC4, HTER4, EER4, ACC4 = Validation(model, test_dataloader4, args, args.thre, Num_test_ffts4)
                test_msg = '[%s] Epoch: %d/%d | Global_step: %d | average testing loss: %f | id: %s | AUC: %f | HTER: %f | EER: %f | ACC: %f' % (
                                                                                        now_time, epoch + 1, args.epochs, Global_step, Avg_test_loss, 'id_28', AUC4, HTER4, EER4, ACC4)
                print('\n', test_msg)
                f1.write(test_msg)
                f1.write('\n')

                Avg_test_loss, AUC5, HTER5, EER5, ACC5 = Validation(model, test_dataloader5, args, args.thre, Num_test_ffts5)
                test_msg = '[%s] Epoch: %d/%d | Global_step: %d | average testing loss: %f | id: %s | AUC: %f | HTER: %f | EER: %f | ACC: %f' % (
                                                                                        now_time, epoch + 1, args.epochs, Global_step, Avg_test_loss, 'id_29', AUC5, HTER, EER5, ACC5)
                print('\n', test_msg)
                f1.write(test_msg)
                f1.write('\n')

                AVG_AUC = (AUC1 + AUC2 + AUC3 + AUC4 + AUC5) / 5.0
                AVG_ACC = (ACC1 + ACC2 + ACC3 + ACC4 + ACC5) / 5.0
                AVG_HTER = (HTER1 + HTER2 + HTER3 + HTER4 + HTER5) / 5.0
                AVG_EER = (EER1+EER2+EER3+EER4+EER5)/5.0
                test_msg = 'AVG_AUC: %f | AVG_ACC: %f| AVG_HTER: %f| AVG_EER: %f' % (AVG_AUC, AVG_ACC,AVG_HTER,AVG_EER)
                print('\n', test_msg)
                f1.write(test_msg)
                f1.write('\n')
    f1.close()


if __name__ == '__main__':
    args = parse_args()
    print(args)
    if args.random_seed is not None:
        fix_seed(args.random_seed)
    model = Fusion_model(2)
    model = model.to(device)
    train(args, model)

