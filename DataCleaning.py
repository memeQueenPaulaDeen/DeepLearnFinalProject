import os
from pathlib import Path
import keras as k
from Generators import SyntheticDataSet
import numpy as np
from scipy import ndimage as nd
import cv2 as cv




if __name__ == "__main__":

    targetData2Clean = "generatedData1"

    UnityOutDataFolder = os.path.join("E:", "UnitySegOutPut", targetData2Clean )
    UnityOutDataFolderX = os.path.join("E:", "UnitySegOutPut", targetData2Clean, 'x' )
    UnityOutDataFolderY = os.path.join("E:", "UnitySegOutPut", targetData2Clean, 'y' )


    cleansedOutPath = os.path.join("E:","UnitySegOutPut" , targetData2Clean, "y_clean")
    Path(cleansedOutPath).mkdir(parents=True, exist_ok=True)

    img_shape = (480, 480, 3)
    num_cat = 6
    dataset = SyntheticDataSet(targetData2Clean, "x", "y", img_shape, num_cat)

    for seg_mask_loc in os.listdir(UnityOutDataFolderY):
        seg_mask_path = os.path.join(UnityOutDataFolderY,seg_mask_loc)

        seg_mask = k.preprocessing.image.load_img(seg_mask_path)
        seg_mask = k.preprocessing.image.img_to_array(seg_mask)



        # there is some noise in the unity data so need to infer bad labels
        # trying to fill by the closest

        foobarMask = None
        for mask_val in dataset.mask2class_encoding:
            if foobarMask is None:
                foobarMask = np.any(seg_mask[:, :] != mask_val, axis=2)
            else:
                foobarMask = np.logical_and(foobarMask, np.any(seg_mask[:, :] != mask_val, axis=2))

        for mask_val, enc in dataset.mask2class_encoding.items():
            seg_mask[np.all(seg_mask == mask_val, axis=2)] = enc

        # ideally we need to figure out how to make unity behave not shadding flat??
        indices = nd.distance_transform_edt(foobarMask, return_distances=False, return_indices=True)
        seg_mask = seg_mask[tuple(indices)]

        y_clean = np.empty(seg_mask.shape)

        for classNum, mask_val in dataset.class_encoding2Mask.items():
            y_clean[seg_mask[:,:,0]==classNum] = mask_val



        k.preprocessing.image.save_img(os.path.join(cleansedOutPath,seg_mask_loc),y_clean)
        # cv.imwrite(os.path.join(cleansedOutPath,seg_mask_loc),cv.cvtColor(y_clean,cv.COLOR_RGB2BGR))