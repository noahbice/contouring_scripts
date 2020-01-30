"""
Retrieve images and masks from DICOM data and store as NumPy arrays.
Data organized as master -> patients -> CT or structs -> DICOM files
Structure directory names should contain an indicator structure_directory_indicator.
CTs are stored as arrays of form (Z, X, Y).
Corresponding structs are stored as (Z, structure, X, Y)
"""

import numpy as np
from skimage import draw
from tqdm import tqdm
import pydicom
import os


def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask

def undicom(master, structure_directory_indicator, to_keep=[], print_structures=False):
    patient = 0
    print('De-dicom-ing...')
    for dir in tqdm(os.listdir(master)):
        sub_dir = os.listdir(master + dir)[0]
        direcs = os.listdir(master + dir + '/' + sub_dir)
        for direc in direcs:
            if structure_directory_indicator in direc:
                struct_dir_path = master + dir + '/' + sub_dir + '/' + direc + '/'
            else:
                image_dir_path = master + dir + '/' +sub_dir + '/' + direc + '/'
                
        #get contour_data
        struct_file = struct_dir_path + os.listdir(struct_dir_path)[0]
        dcmstruct = pydicom.dcmread(struct_file)
        contours = np.zeros((len(to_keep)), dtype='object')
        if print_structures:
            for struct in range(len(dcmstruct.StructureSetROISequence)):
                name = dcmstruct.StructureSetROISequence[struct].ROIName
                print(name)
            print()
            quit()
        contours_present = {}
        present_index = 0
        for struct in range(len(dcmstruct.StructureSetROISequence)):
            name = dcmstruct.StructureSetROISequence[struct].ROIName
            contours_present[name] = present_index
            present_index += 1
        for i, name in enumerate(to_keep):
            if name in contours_present.keys():
                contour = []
                for slice in range(len(dcmstruct.ROIContourSequence[contours_present[name]].ContourSequence)):
                    slice_contour = dcmstruct.ROIContourSequence[contours_present[name]].ContourSequence[slice].ContourData
                    contour.append(slice_contour)
                contours[i] = contour
            else:
                contours[i] = []
        contours = np.array(contours)
        
        #store images
        im_file_list = os.listdir(image_dir_path)
        num_slices = len(im_file_list)
        volume = np.zeros((num_slices,512,512), dtype = 'float32')
        ipp = np.zeros((3))
        highest = np.inf
        for image_file in im_file_list:
            dcmimage = pydicom.dcmread(image_dir_path + image_file)
            slice = dcmimage.InstanceNumber
            slice = num_slices - slice
            IPP = dcmimage.ImagePositionPatient
            if IPP[2] < highest:
                ipp = IPP
                highest = IPP[2]
                spacing = dcmimage.PixelSpacing
                spacing.append(dcmimage.SliceThickness)
                spacing = np.array(spacing)
            img = dcmimage.pixel_array
            volume[slice,:,:] = img
        np.save('./npydata/cts/' + str(patient) + '.npy', volume)
        
        #make masks
        num_structs = len(to_keep)
        mask = np.zeros((num_slices, num_structs, 512,512), dtype='int')
        for struct in range(num_structs):
            if contours[struct] != 0:
                for slice in range(len(contours[struct])):
                    X = ((np.array(contours[struct][slice][::3]) - ipp[0]) / (spacing[0])).astype(np.int)
                    Y = ((np.array(contours[struct][slice][1::3]) - ipp[1]) / (spacing[1])).astype(np.int)
                    Z = ((np.array(contours[struct][slice][2]) - ipp[2]) / (spacing[2])).astype(np.int)
                    mask[Z,struct,:,:] += poly2mask(Y,X,(512,512)).astype(int)
        mask %= 2
        mask = mask.astype('bool')
        np.save('./npydata/contours/' + str(patient) + '.npy', mask)
        patient += 1
        
if __name__ == '__main__':

    #-----------------------------------
    master = './dicom/' #master directory
    to_keep = ['Spinal-Cord', 'Neck-Right', 'Neck-Left', \
        'Submandibular-Gland-Right', 'Submandibular-Gland-Left', 'Parotid-Right', \
        'Parotid-Left', 'Oral-Cavity', 'Medulla-Oblongata', 'Brain']
    structure_directory_indicator = '-' #how structure directories are different from image directories
    #-----------------------------------
    
    #undicom(master, structure_directory_indicator, print_structures=True) #print available structs in first patient
    undicom(master, structure_directory_indicator, to_keep=to_keep, print_structures=False)

    