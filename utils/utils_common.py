import numpy as np
from scipy import ndimage as nd
from scipy.ndimage import binary_opening





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
    if l_min == -1:
        l_min = np.max(num_elements_by_lesion)
    # filter candidates by size and store those > l_min
    count = 0
    for l in range(len(num_elements_by_lesion)):
        if num_elements_by_lesion[l] >= l_min:
            count = count + 1
            # assign voxels to output
            current_voxels = np.stack(np.where(lab_img == l), axis=1)
            output_msk[current_voxels[:, 0], current_voxels[:, 1], current_voxels[:, 2]] = 1
            output_lab[current_voxels[:, 0], current_voxels[:, 1], current_voxels[:, 2]] = count

    return output_msk, output_lab



def evaluate_fp(cc, label):
    """
    Evaluate the number of false positives in a connected component analysis.

    Parameters:
    cc (numpy.ndarray): Label image of connected component analysis.
    label (numpy.ndarray): Ground truth label image.

    Returns:
    int: Number of false positives, where a false positive is defined as a connected component 
         that does not overlap with the true label (i.e., Dice coefficient of 0).
    """
    # Get unique labels from the connected component image
    regs_idx = np.unique(cc.flatten())
    dc_sub = []
    # Iterate over each label
    for l in regs_idx:
        if l > 0:  # Ignore the background label
            # Create a mask for the current label
            mask = (cc == l)
            # Check if there is any overlap between the mask and the true label
            dc_sub.append(np.logical_and(mask, label).sum() > 0)
    # Count the number of false positives (no overlap with the true label)
    return np.sum(np.array(dc_sub) == 0)

