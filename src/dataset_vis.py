import numpy as np
from pysteps.visualization import plot_precip_field
import matplotlib.pyplot as plt
all_data_test = np.load('/space/ankushroy/Data/all_data_test.npy')
print('Testing data shape',all_data_test.shape)
image_seq = all_data_test[600]
for i in range (image_seq.shape[0]):
    image = image_seq[i,:,:]*40 #    Important post-processing step
    plt.figure(figsize=(12, 4))
    plot_precip_field(image, title="Input")
    plt.tight_layout()
    plt.show()

all_data_train = np.load('/space/ankushroy/Data/all_data_train.npy')
print('Training data shape',all_data_train.shape)
image_seq_t = all_data_train[10000]
for i in range (image_seq_t.shape[0]):
    image_t = image_seq_t[i,:,:]*40 #    Important post-processing step
    plt.figure(figsize=(12, 4))
    plot_precip_field(image_t, title="Input")
    plt.tight_layout()
    plt.show()

all_data_vali = np.load('/space/ankushroy/Data/all_data_vali.npy')
print('Validation data shape',all_data_vali.shape)
image_seq_v = all_data_vali[927]
for i in range (image_seq_v.shape[0]):
    image_v = image_seq_v[i,:,:]*40 #    Important post-processing step
    plt.figure(figsize=(12, 4))
    plot_precip_field(image_v, title="Input")
    plt.tight_layout()
    plt.show()