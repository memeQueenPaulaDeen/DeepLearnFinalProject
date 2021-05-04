import keras
import numpy as np
import matplotlib.pyplot as plt

num = 123
file_x = ''
file_y = ''

train_x = np.load('./train/npy/'+file_x+'.npy')
train_y = np.load('./train/npy/'+file_y+'.npy')
model = keras.models.load_model('./output/test/seg_model.h5')

input = train_x[num]
mask = train_y[num]

input = np.expand_dims(input, axis=0)
output = model.predict(input, verbose=0)[0]

print(input.shape)
print(output.shape)
print(mask.shape)
print(np.amax(mask))

mask_new = None
output_new = None
if seg_model_load:
    mask_new = convert_mask(mask, 0)
    output_new = convert_mask(output, 1)

plt.imshow(input[0])
plt.show()
if seg_model_load:
    plt.imshow(output_new, cmap='Set1')
    plt.show()
    plt.imshow(mask_new, cmap='Set1')
    plt.show()
else:
    plt.imshow(output[:,:, 0])
    plt.show()
    plt.imshow(mask)
    plt.show()
