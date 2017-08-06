import os, struct
from array import array as pyarray
import sys
import random
import numpy as np
import h5py

def load_mnist_complex(fname_img, fname_lbl):
    digits=np.arange(10)

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = size

    data_shape = [N, 1, rows, cols]
    images = np.zeros(data_shape, dtype=np.uint8)
    labels = np.zeros((N, 1), dtype=np.int8)
    
    for i in range(N):
        images[i, 0, :, :] = np.array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    labels = [label[0] for label in labels]

    data_shape_ri = data_shape + [2]
    data_ri = np.zeros(data_shape_ri, dtype=np.float32)
    data_ri[:,:,:,:,0] = images.astype(np.float32)/255.0
    data_ri[:,:,:,:,1] = np.zeros(data_shape, dtype=np.float32)

    return(data_ri, np.array(labels).astype(np.float32))

def test_write():
    # Example complex data
    data_shape = [1000, 1, 128, 128]
    data_complex = np.random.randn(*data_shape) + 1j*np.random.randn(*data_shape)
    labels = np.floor(np.random.uniform(0, 10, data_shape[0])).astype(np.float32)

    data_shape_ri = data_shape + [2]
    data_ri = np.zeros(data_shape_ri, dtype=np.float32)
    data_ri[:,:,:,:,0] = np.real(data_complex)
    data_ri[:,:,:,:,1] = np.imag(data_complex)

    return(data_ri, labels)

def main(argv):
    if len(argv) != 4:
        print("")
        sys.exit()

    fimg = argv[1]
    flbl = argv[2]
    fout = argv[3]

    img_z, lbls = load_mnist_complex(fimg, flbl)

    filename = fout + ".h5"
    hfile = h5py.File(filename, 'w')
    hfile.create_dataset('data', data=img_z)
    hfile.create_dataset('labels', data=lbls)
    hfile.close()

    with open(fout + ".txt", "w") as file:
        file.write(filename)

if __name__=="__main__": main(sys.argv)
