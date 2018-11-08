#!/usr/bin/python
#coding:utf-8
import os 
import matplotlib.pyplot as plt 
import struct 
import numpy as np 
def load_mnist(path, kind='t10k'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
            '%s-labels.idx1-ubyte' % kind) 
    images_path = os.path.join(path,
            '%s-images.idx3-ubyte' % kind) 
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                lbpath.read(8))
        labels = np.fromfile(lbpath,
                dtype=np.uint8)
        
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                imgpath.read(16)) 
        images = np.fromfile(imgpath,
                dtype=np.uint8).reshape(len(labels), 784)
        
        
    return images, labels
def main():
    x_train,y_train=load_mnist("/home/carl/Program/minst")
    fig, ax = plt.subplots( 
            nrows=10, ncols=10, sharex=True, sharey=True, ) 

    ax = ax.flatten() 
    for i in range(100): 
        img = x_train[i].reshape(28, 28) 
        print(y_train[i],end="")
        if((i+1)%10!=0):
            print(" ",end="")
        else:
            print()

        ax[i].imshow(img, cmap='Greys', interpolation='nearest') 
    
    ax[0].set_xticks([]) 
    ax[0].set_yticks([]) 
    plt.tight_layout() 
    plt.show()
    return;
if __name__ == "__main__":
    main()
