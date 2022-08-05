from __future__ import print_function, division
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from arch.fusion import Fusion_model
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F
from utils.dataloader_fusion import Intra_fusion_datareader, my_transforms
from utils.metrics import get_metrics

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device: ', device)

def parse_args():
    parser = argparse.ArgumentParser(description='cq_test')
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--batch_size_val', default=100, type=int)
    parser.add_argument('--thre', default=0.5, type=float)
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--model_path', default="./trained_models/Cross_5id/fusion_new_ca/note/ckpt/best.pth", type=str)

    parser.add_argument('--test_csv1', default="./csv_files/cross_5id/test/cross_5id_id_25_note_test.csv", type=str)
    parser.add_argument('--test_csv2', default="./csv_files/cross_5id/test/cross_5id_id_26_note_test.csv", type=str)
    parser.add_argument('--test_csv3', default="./csv_files/cross_5id/test/cross_5id_id_27_note_test.csv", type=str)
    parser.add_argument('--test_csv4', default="./csv_files/cross_5id/test/cross_5id_id_28_note_test.csv", type=str)
    parser.add_argument('--test_csv5', default="./csv_files/cross_5id/test/cross_5id_id_29_note_test.csv", type=str)
    return parser.parse_args()

def Testing(model, dataloader, args, thre, facenum):
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

def test(args, model):
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

    Num_test_ffts1 = len(pd.read_csv(args.test_csv1, header=None))
    Num_test_ffts2 = len(pd.read_csv(args.test_csv2, header=None))
    Num_test_ffts3 = len(pd.read_csv(args.test_csv3, header=None))
    Num_test_ffts4 = len(pd.read_csv(args.test_csv4, header=None))
    Num_test_ffts5 = len(pd.read_csv(args.test_csv5, header=None))

    Avg_test_loss, AUC1, HTER1, EER1, ACC1 = Testing(model, test_dataloader1, args, args.thre, Num_test_ffts1)
    test_msg = ' average testing loss: %f | id: %s | AUC: %f | HTER: %f | EER: %f | ACC: %f' % (Avg_test_loss, 'id_25', AUC1, HTER1, EER1, ACC1)
    print('\n', test_msg)

    Avg_test_loss, AUC2, HTER2, EER2, ACC2 = Testing(model, test_dataloader2, args, args.thre, Num_test_ffts2)
    test_msg = ' average testing loss: %f | id: %s | AUC: %f | HTER: %f | EER: %f | ACC: %f' % (Avg_test_loss, 'id_26', AUC2, HTER2, EER2, ACC2)
    print('\n', test_msg)

    Avg_test_loss, AUC3, HTER3, EER3, ACC3 = Testing(model, test_dataloader3, args, args.thre, Num_test_ffts3)
    test_msg = ' average testing loss: %f | id: %s | AUC: %f | HTER: %f | EER: %f | ACC: %f' % (Avg_test_loss, 'id_27', AUC3, HTER3, EER3, ACC3)
    print('\n', test_msg)

    Avg_test_loss, AUC4, HTER4, EER4, ACC4 = Testing(model, test_dataloader4, args, args.thre, Num_test_ffts4)
    test_msg = ' average testing loss: %f | id: %s | AUC: %f | HTER: %f | EER: %f | ACC: %f' % (Avg_test_loss, 'id_28', AUC4, HTER4, EER4, ACC4)
    print('\n', test_msg)

    Avg_test_loss, AUC5, HTER5, EER5, ACC5 = Testing(model, test_dataloader5, args, args.thre, Num_test_ffts5)
    test_msg = ' average testing loss: %f | id: %s | AUC: %f | HTER: %f | EER: %f | ACC: %f' % (Avg_test_loss, 'id_29', AUC5, HTER5, EER5, ACC5)
    print('\n', test_msg)

    AVG_AUC = (AUC1 + AUC2 + AUC3 + AUC4 + AUC5) / 5.0
    AVG_ACC = (ACC1 + ACC2 + ACC3 + ACC4 + ACC5) / 5.0
    AVG_HTER = (HTER1 + HTER2 + HTER3 + HTER4 + HTER5) / 5.0
    AVG_EER = (EER1+EER2+EER3+EER4+EER5)/5.0
    test_msg = 'AVG_AUC: %f | AVG_ACC: %f| AVG_HTER: %f| AVG_EER: %f' % (AVG_AUC, AVG_ACC,AVG_HTER,AVG_EER)
    print('\n', test_msg)


def main(args):
    model = Fusion_model(2)
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)
    test(args, model)

if __name__ == '__main__':
    args = parse_args()
    main(args)

