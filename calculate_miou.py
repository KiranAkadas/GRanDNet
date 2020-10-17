import glob
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

def log_out(out_str, f_out):
    f_out.write(out_str + '\n')
    f_out.flush()
    print(out_str)


gt_classes = [0 for _ in range(5)]
positive_classes = [0 for _ in range(5)]
true_positive_classes = [0 for _ in range(5)]
val_total_correct = 0
val_total_seen = 0
#a=np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]])
res_path="./test/Log_*_*_*/predictions/*.labels"

for filename in glob.glob(res_path):
    label_pd = pd.read_csv(filename, header=None, delim_whitespace=True, dtype=np.uint8)
    pred_labels = label_pd.values
    #print(cloud_labels.shape)
    fil=file_name = filename.split('/')[-1][:-11]
    orig_file="./data_test/original_data/"+fil+".txt"
    pc_pd = pd.read_csv(orig_file, header=None, delim_whitespace=True, dtype=np.float16)
    pc = pc_pd.values
    lab=pc[:, 3:4].astype(np.uint8)
    #print(lab.shape)
    invalid_idx = np.where(lab == [0])[0]
    #print(lab.shape)
    labels_valid = np.delete(lab, invalid_idx)
    #print(labels_valid,labels_valid.shape)
    labels_valid = labels_valid# - 1
    #print(labels_valid)
    pred_valid = np.delete(pred_labels, invalid_idx)
    #print(pred_labels.shape,pred_valid.shape)
    correct = np.sum(pred_valid == labels_valid)
    val_total_correct += correct
    val_total_seen += len(labels_valid)
    #print(val_total_correct,val_total_seen)
    #print(np.unique(pred_valid))
    #print(np.unique(labels_valid))
    conf_matrix = confusion_matrix(labels_valid, pred_valid, np.arange(1, 6, 1))
    gt_classes += np.sum(conf_matrix, axis=1)
    positive_classes += np.sum(conf_matrix, axis=0)
    true_positive_classes += np.diagonal(conf_matrix)
    

iou_list = []
for n in range(0, 5, 1):
    iou = true_positive_classes[n] / float(gt_classes[n] + positive_classes[n] - true_positive_classes[n])
    iou_list.append(iou)
mean_iou = sum(iou_list) / float(5)


log_file=open("./log_MIOU.txt", 'a')

log_out('eval accuracy: {}'.format(val_total_correct / float(val_total_seen)),log_file)
log_out('mean IOU:{}'.format(mean_iou), log_file)

mean_iou = 100 * mean_iou
log_out('Mean IoU = {:.1f}%'.format(mean_iou), log_file)
s = '{:5.2f} | '.format(mean_iou)
for IoU in iou_list:
    s += '{:5.2f} '.format(100 * IoU)
log_out('-' * len(s), log_file)
log_out(s, log_file)
log_out('-' * len(s) + '\n', log_file)
print(mean_iou)
