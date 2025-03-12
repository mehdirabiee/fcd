import os
import nibabel as nib
import numpy as np
import random
import torch
from scipy import ndimage as nd
from scipy.ndimage import binary_opening
from sklearn.metrics import cohen_kappa_score
from monai.metrics import DiceMetric, compute_average_surface_distance
from sklearn.metrics import jaccard_score, accuracy_score, roc_auc_score, f1_score, roc_curve
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix, classification_report


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.deterministic = False #overfitting if true
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

   

class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def write_image(img, save_dir, name, affine=None):
    os.makedirs(save_dir, exist_ok= True)
    output_name = os.path.join(save_dir, name)
    nib.save(nib.Nifti1Image(img.astype(np.float64), affine), output_name)

def gradient_penalty(real_images, fake_images, D, device):
   # compute gradient penalty
    alpha = torch.rand(real_images.size(0), 1, 1, 1, 1).cuda().expand_as(real_images)
    # (*, 1, 64, 64)
    interpolated = (alpha * real_images.data + ((1 - alpha) * fake_images.data)).requires_grad_(True)
    # (*,)
    out = D(interpolated)[-1]
    # get gradient w,r,t. interpolates
    grad = torch.autograd.grad(
        outputs=out,
        inputs = interpolated,
        grad_outputs = torch.ones(out.size()).cuda(),
        retain_graph = True,
        create_graph = True,
        only_inputs = True
    )[0]

    grad_l2norm = grad.norm(2, dim=[1,2,3,4])
    gp = torch.mean((grad_l2norm - 1) ** 2)

    return gp


def get_sliding_windows(input_shape, patch_size, overlap=0.25):
    step0 = int(patch_size[0]*(1-overlap))
    step1 = int(patch_size[1]*(1-overlap))
    step2 = int(patch_size[2]*(1-overlap))

    s0 = [0]
    while(1):
        p = s0[-1] + step0
        if p + patch_size[0] < input_shape[0]:
            s0.append(p)
        elif p + patch_size[0] == input_shape[0]:
            s0.append(p)
            break
        else:
            s0.append(input_shape[0]-patch_size[0])
            break

    s1 = [0]
    while(1):
        p = s1[-1] + step1
        if p + patch_size[1] < input_shape[1]:
            s1.append(p)
        elif p + patch_size[1] == input_shape[1]:
            s1.append(p)
            break
        else:
            s1.append(input_shape[1]-patch_size[1])
            break
            
    s2 = [0]
    while(1):
        p = s2[-1] + step2
        if p + patch_size[2] <= input_shape[2]:
            s2.append(p)
        elif p + patch_size[2] == input_shape[2]:
            s2.append(p)
            break
        else:
            s2.append(input_shape[2]-patch_size[2])
            break
    return s0, s1, s2

#mask: binary mask
def post_process_segment(mask, l_min):

    output_msk = np.zeros_like(mask)
    output_lab = np.zeros_like(mask)

    morphed = binary_opening(mask, iterations=1)
    morphed = nd.binary_fill_holes(morphed, structure=np.ones((5, 5, 5))).astype(int)
    lab_img, _ = nd.label(morphed, structure=np.ones((3, 3, 3)))
    lab_val = np.unique(lab_img)
    num_elements_by_lesion = nd.labeled_comprehension(
        morphed, lab_img, lab_val, np.sum, float, 0
    )

    # filter candidates by size and store those > l_min
    count = 0
    for l in range(len(num_elements_by_lesion)):
        if num_elements_by_lesion[l] > l_min:
            count = count + 1
            # assign voxels to output
            current_voxels = np.stack(np.where(lab_img == l), axis=1)
            output_msk[current_voxels[:, 0], current_voxels[:, 1], current_voxels[:, 2]] = 1
            output_lab[current_voxels[:, 0], current_voxels[:, 1], current_voxels[:, 2]] = count

    return output_msk, output_lab


def evaluate_dice(im1, im2):
    """
    dice coefficient 2nt/na + nb.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return 0

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    dc = 2.0 * intersection.sum() / im_sum

    return dc

#cc: label image of connected component analysis
def evaluate_fp(cc, label):

    regs_idx = np.unique(cc.flatten())
    dc_sub = []
    for l in regs_idx:
        if l>0:
            mask = cc == l
            dc_sub.append( evaluate_dice(mask, label) )
    return np.sum(np.array(dc_sub)==0)


def evaluate_classifcation(pred, prob, label, save_dir):
    #sensitivuty = tp/(tp+fn)
    #specificity = tn/(tn+fp)
    
    pred = (pred > 0).astype(np.int32).flatten()
    prob = prob.flatten()
    label = label.flatten()
    cr = classification_report(label, pred, output_dict=True)

    cm = confusion_matrix(label, pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp/(tp+fn+1e-6)
    spec = tn/(tn+fp+1e-6)

    prob = (prob*100).astype(np.int)/100.0
    #fpr, tpr, thresholds = roc_curve(label, prob, pos_label=1)
    auc = roc_auc_score(label, prob)

    metric = dict()
    metric['prec']   = cr['1.0']['precision']
    metric['recall'] = cr['1.0']['recall']
    metric['f1']     = cr['1.0']['f1-score']
    metric['sens'] = sens
    metric['spec'] = spec
    #metric['fpr'] = fpr
    #metric['tpr'] = tpr
    metric['auc']  = auc

    '''
    os.makedirs(save_dir, exist_ok=True)
    metric_file = os.path.join(save_dir, 'roc.txt')
    with open(metric_file, "w") as file:
        for p, q in zip(fpr, tpr):
            file.write("{:.3f} \t {:.3f}\n".format(p, q))
        file.write("{:.3f} \t 0.000\n".format(auc))

    metric_file = os.path.join(save_dir, 'metric_voxel.txt')
    with open(metric_file, "w") as file:
        file.write("precision   : {:.3f}\n".format(metric['prec']))
        file.write("recall      : {:.3f}\n".format(metric['recall']))
        file.write("f1-score    : {:.3f}\n".format(metric['f1']))
        file.write("sensitivity : {:.3f}\n".format(metric['sens']))
        file.write("specificity : {:.3f}\n".format(metric['spec']))
        file.write("auc         : {:.3f}\n".format(metric['auc']))
    '''

    return metric



