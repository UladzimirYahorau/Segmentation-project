import nibabel as nib
import os
import numpy as np
from random import randint
import random

modalities = ["Flair", "T1", "T2", "DWI"]


def get_affine():
    """
    This function returns one of the input parameters which is required when we save image to a .nii format in 
    save_segmentation_to_file.
    """
    folder = "/home/uladzimir/Segmentation project/SISS2015/Training/01"
    for file in os.listdir(folder):
        if "OT" in file:
            filefile = file + '/' + file +'.nii'
            file_name = os.path.join(folder, filefile)
            image = nib.load(file_name)
    return image.affine

    

def load_data(data_path = "/home/uladzimir/Segmentation project/SISS2015/Training", modalities = ["Flair", "T1", "T2", "DWI"]):
    """
    Parameters:
    data_path: path to the folder "Training" with brain images 
    modalities: list of modalities of the image
    
    Returns: by default, list of 28 4D numpy arrays of brain images in 4 different modalities (stacked along the 0-axis)
    """
    data = []
    for brain_id in range(1, 29):
        
        images = []
        
        if brain_id < 10:
                brain_id = '0' + str(brain_id)
        
        for modality in modalities:
            
            folder = os.path.join(data_path, str(brain_id))
            for file in os.listdir(folder):
                if modality in file:
                    filefile = file + '/' + file +'.nii'
                    file_name = os.path.join(folder, filefile)
                    image = nib.load(file_name)
                    np_image = image.get_data()
                    images.append(np_image)
            normalized = standard_normalization(images)
            stacked = np.stack(normalized, axis = 0)
        data.append(stacked)
    
    return data

     
    
    
def load_labels(data_path = "/home/uladzimir/Segmentation project/SISS2015/Training"):
    """
    Parameters: 
    data_path: path to the folder "Training" with brain images
    
    Returns: a list of 28 3D numpy arrays of 0/1 labels. 
    """
    labels = []
    for brain_id in range(1, 29):
        if brain_id < 10:
            brain_id = '0' + str(brain_id)
        folder = os.path.join(data_path, str(brain_id))
        for file in os.listdir(folder):
            if "OT" in file:
                filefile = file + '/' + file +'.nii'
                file_name = os.path.join(folder, filefile)
                image = nib.load(file_name)
                np_image = image.get_data()
        labels.append(np_image)
    
    return labels


def standard_normalization(data):
    """
    Parameters:
    data: a list of numpy arrays
    
    Returns: a list of normalized numpy arrays with overall mean 0 and standard deviation 1.
    """
    normalized = []
    for image in data:
        m = np.mean(image)
        image = image - m
        st_dev = np.std(image)
        image = image/st_dev
        normalized.append(image)
    return normalized


# Previous version of one_hot_encode function, working for 1-dimensional arrays
#def one_hot_encode_old(labels, number_of_labels=2):
#    """
#    One hot encode a list of sample labels.
#    Parameters:
#    number_of_labels: an integer greater than 1
#    labels: a list of integers between 0 and number_of_labels-1
#    
#    Returns: a numpy array of one-hot encoded labels
#    """
#    labels = np.array(labels)
#    one_hot = np.zeros((labels.size, number_of_labels), dtype = np.int)
#    one_hot[np.arange(labels.size), labels] = 1 
#    return one_hot

def one_hot_encode(labels, number_of_labels=2):
    """
    Parameters:
    number_of_labels: a positive integer
    labels: a list of integers between 0 and number_of_labels-1
    
    Returns: a numpy array of one-hot encoded labels
    """
    labels = np.array(labels)
    class_encodings = []
    for label in range(0, number_of_labels):
        class_encodings.append(np.equal(labels, label).astype(int))
    one_hot = np.stack(class_encodings, axis = -1)
    return one_hot



def get_samples(data, labels, brain_ids = range(28), size = 100, dim = 33):
    """
    Parameters:
    data: list of 3D numpy arrays representing an image of the brain
    labels: list of 3D numpy array representing a voxelwise segmentation of the brain in data
    brain_ids: list of brain indices to sample from
    size: number of extracted patches 
    dim: patch has a shape (dim, dim)

    Returns: tuple, namely, (array of patches, array of labels of the central voxels in the patches).
    
    Samples are taken from the uniform distribution.
    """
    batch_data = []
    batch_labels = []
    while len(batch_data) < size:
        index = randint(0,len(brain_ids)-1)
        x = randint(int(dim/2)+1, data[index].shape[1] - int(dim/2)-1)
        y = randint(int(dim/2)+1, data[index].shape[2] - int(dim/2)-1)
        z = randint(0, data[index].shape[3]-1)
        if data[index][0, x, y, z] != 0:
            patch = data[index][: , x-int(dim/2):x+int(dim/2)+1, y-int(dim/2):y+int(dim/2)+1, z]
           # patch.reshape((dim, dim, 1))
            batch_data.append(patch)
            label = labels[index][x][y][z]
            batch_labels.append(label)
    batch_data = np.array(batch_data)
    batch_labels = np.array(one_hot_encode(batch_labels))
    
    return batch_data, batch_labels
    


def get_uniform_samples(data, labels, brain_ids = range(28), size = 100, dim = 33):
    """
    Parameters:
    data: list of 3D numpy arrays representing an image of the brain
    labels: list of 3D numpy array representing a voxelwise segmentation of the brain in data
    brain_ids: list of brain indices to sample from 
    size: number of extracted patches 
    dim: patch has a shape (dim, dim)

    Returns: tuple, namely, (array of patches, array of labels of the central voxels in the patches).
    
    There are equal number of samples labeled 1 and 0.
    """
    batch_data = []
    batch_labels = []
    while len(batch_data) < size/2:
        index = randint(0,len(brain_ids)-1)
        x = randint(int(dim/2)+1, data[index].shape[1] - int(dim/2)-1)
        y = randint(int(dim/2)+1, data[index].shape[2] - int(dim/2)-1)
        z = randint(0, data[index].shape[3]-1)
        if data[index][0,x,y,z] != 0:
            patch = data[index][:, x-int(dim/2):x+int(dim/2)+1, y-int(dim/2):y+int(dim/2)+1, z]
            batch_data.append(patch)
            label = labels[index][x][y][z]
            batch_labels.append(label)
    while len(batch_data) < size:
        x = randint(int(dim/2)+1, data[index].shape[1] - int(dim/2)-1)
        y = randint(int(dim/2)+1, data[index].shape[2] - int(dim/2)-1)
        z = randint(0, data[index].shape[3]-1)
        if data[index][0, x, y, z] != 0 and labels[index][x][y][z] == 1:
            patch = data[index][:, x-int(dim/2):x+int(dim/2)+1, y-int(dim/2):y+int(dim/2)+1, z]
            batch_data.append(patch)
            label = labels[index][x][y][z]
            batch_labels.append(label)
    batch_data = np.array(batch_data)
    batch_labels = np.array(one_hot_encode(batch_labels))
 
    return batch_data, batch_labels

def brain_mask(t):
    """
    Given a 4D prediction of a brain segmentation by segm_model_1nn.py, convert it into a 3D-numpy array 
    """
    labeled = np.argmax(t, axis = 3)
    labeled = labeled.transpose([1,2,0])
    labeled = np.concatenate((np.zeros([16,198,153]),labeled, np.zeros([16,198,153])), axis = 0)
    labeled = np.concatenate((np.zeros([230, 16 , 153]), labeled, np.zeros([230, 16 , 153])), axis = 1)
                             
    return labeled
    
    
def save_segmentation_to_file(labeled, index, path = "/home/uladzimir/Segmentation project/predicted segmentations"):
    affine = get_affine()
    new_image = nib.Nifti1Image(labeled, affine)
    file = str(index) + ".nii"
    file_name = os.path.join(path, file)
    nib.save(new_image, file_name)
    
    
                          
def get_validation_samples(data, labels, brain_ids = range(28), dim = 33):
    """
    Parameters:
        data: a list of 4D-numpy arrays of brain images 
        labels: a list of 3D-numpy arrays of 0/1 labels
    Returns:
        a tuple, namely, (a numpy array of horizontal slices in 4 modalities,
                          a numpy array of one-hot encoded labels for those slices)
    """
    z_dim = 153 # each brain contains either 153 or 154 horizontal slices and the bottom ones are irrelevant.
    
    validation_data = []
    validation_labels = []
    for index in brain_ids:
        for i in range(z_dim):
            patch = data[index][:,:,:,i]
            validation_data.append(patch)
            label = labels[index][dim//2:-(dim//2), dim//2:-(dim//2),i]
            validation_labels.append(label)
    validation_data = np.array(validation_data)
    validation_labels = np.array(one_hot_encode(validation_labels))
    return validation_data, validation_labels
    
def crop_samples(data, dim_data = 65, dim_crop = 33):
    """
    Parameters:
    data: 3D numpy array of patches
    Returns:
    3D numpy array of central subpatches 
    """
    crop = []
    for patch in data:
        subpatch = patch[:,dim_data//2 - int(dim_crop/2):dim_data//2+int(dim_crop/2)+1,
                        dim_data//2 - int(dim_crop/2):dim_data//2+int(dim_crop/2)+1]
        crop.append(subpatch)   
    crop = np.array(crop) 
    return crop


