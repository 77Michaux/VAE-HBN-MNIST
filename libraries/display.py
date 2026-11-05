import numpy as np
from matplotlib import pyplot as plt


import torch
import torch.nn as nn

def display_from_net(net: nn.Module, X: torch.Tensor, F: int = 2) : 
    pred = net.pred(X)
    hat_X = pred[0]
    del pred
    fig, axes = plt.subplots(2, 2)
    axes[0, 0].imshow(X.detach().numpy()[0, 0])
    axes[0, 1].imshow(hat_X.detach().numpy()[0, 0])
    if F == 2 :
        axes[1, 0].imshow(X.detach().numpy()[0, 1], cmap = "jet")
        axes[1, 1].imshow(hat_X.detach().numpy()[0, 1], cmap = "jet")
    
    plt.show()


def display(X: torch.Tensor, hat_X: torch.Tensor, F: int = 2) : 
    fig, axes = plt.subplots(2, 2)
    axes[0, 0].imshow(X.detach().numpy()[0, 0])
    axes[0, 1].imshow(hat_X.detach().numpy()[0, 0])
    if F == 2 :
        axes[1, 0].imshow(X.detach().numpy()[0, 1], cmap = "jet")
        axes[1, 1].imshow(hat_X.detach().numpy()[0, 1], cmap = "jet")
    
    plt.show()


def group_images(X: np.array, cols: int = 10, has_filter = True) -> np.array :
    images = None
    if has_filter : 
        images = np.array(X.mean(axis = 1), dtype = np.float32)
    else : 
        images = np.array(X, dtype = np.float32)
    # images = exp_softmax(images)
    index = len(images)
    if index%cols == 0 :
        new_images = images.reshape((cols, index//cols,) + X.shape[-2:])
        new_images = np.concatenate(new_images, axis = 2)
        new_images = np.concatenate(new_images, axis = 0)
        return new_images
    print("cols must divide X.shape[0] !")
    return None

def display_group_images(X: np.array, cols: int = 10, figsize = None, cmap = 'viridis', has_filter = True) :
    images = group_images(X, cols, has_filter)
    if images is not None :
        if figsize is None :
            figsize = (20, 2 *len(X) //cols)
        
        fig = plt.figure(figsize = figsize)
        plt.imshow(images, cmap = cmap)

def display_group_images_3d(X: np.array, cols: int = 10, cmap = 'viridis', has_filter = False, K = None, axis = 0) :
    X = np.array([np.take(X, k, axis = axis) for k in K])
    images = group_images(X, cols, has_filter)
    if images is not None :
        plt.imshow(images, cmap = cmap)


def display_error(x, hat_x, k = 10, axis = 0) :
    plt.figure(figsize = (12, 6)) 
    x = np.take(x, k, axis = axis+1)
    hat_x = np.take(hat_x, k, axis = axis+1)
    y = np.abs(x[0]*x[3] -x[2])
    hat_y = np.abs(hat_x[0]*hat_x[3] -hat_x[2])
    
    true_x  = np.concatenate(list(x)+[y], axis = 1)
    false_x = np.concatenate(list(hat_x)+[hat_y], axis = 1)
    diff = np.abs(true_x -false_x)
    
    img = np.concatenate([true_x, false_x, diff], axis = 0)
    plt.imshow(img, cmap = "jet")
    plt.colorbar()
    plt.show()



def display_epoch(net: nn.Module, X: torch.Tensor, epoch: int = 0, mode = None) : 
    pred = net.pred(X)
    hat_X = pred[0]
    del pred
    F = X.shape[1]
    if len(X.shape[2:]) == 2 :
        fig, axes = plt.subplots(F, 2)
        if F == 1 :
            axes[0].imshow(X.detach().numpy()[0, 0])
            axes[1].imshow(hat_X.detach().numpy()[0, 0])
            axes[0].set_axis_off()
            axes[1].set_axis_off()
        if F == 2 :
            axes[0, 0].imshow(X.detach().numpy()[0, 0])
            axes[0, 1].imshow(hat_X.detach().numpy()[0, 0])
            axes[0, 0].set_axis_off()
            axes[0, 1].set_axis_off()
            axes[1, 0].imshow(X.detach().numpy()[0, 1], cmap = "jet")
            axes[1, 1].imshow(hat_X.detach().numpy()[0, 1], cmap = "jet")
            axes[1, 0].set_axis_off()
            axes[1, 1].set_axis_off()
            
    elif len(X.shape[2:]) == 3 :
        coords = np.array(X.shape[2:])
        k, k2, k3 = coords//2
        
        fig, axes = plt.subplots(2, 3*F, figsize = (18, 6)) # h, w
        if F == 1 :
            axes[0, 0].imshow(X.detach().numpy()[0, 0, k], cmap = "jet")
            axes[1, 0].imshow(hat_X.detach().numpy()[0, 0, k], cmap = "jet")
            #
            axes[0, 1].imshow(X.detach().numpy()[0, 0, :, k2 +3], cmap = "jet")
            axes[1, 1].imshow(hat_X.detach().numpy()[0, 0, :, k2 +3], cmap = "jet")
            # 
            axes[0, 2].imshow(X.detach().numpy()[0, 0, :, :, k3], cmap = "jet")
            axes[1, 2].imshow(hat_X.detach().numpy()[0, 0, :, :, k3], cmap = "jet")
            
        if F >= 2 :
            axes[0, 0].imshow(X.detach().numpy()[0, 0, k], cmap = "jet")
            axes[0, 1].imshow(hat_X.detach().numpy()[0, 0, k], cmap = "jet")
            
            axes[1, 0].imshow(X.detach().numpy()[0, 1, k], cmap = "gray")
            axes[1, 1].imshow(hat_X.detach().numpy()[0, 1, k], cmap = "gray")
            #
            axes[0, 2].imshow(X.detach().numpy()[0, 0, :, k2 +3], cmap = "jet")
            axes[0, 3].imshow(hat_X.detach().numpy()[0, 0, :, k2 +3], cmap = "jet")
            
            axes[1, 2].imshow(X.detach().numpy()[0, 1, :, k2 +3], cmap = "gray")
            axes[1, 3].imshow(hat_X.detach().numpy()[0, 1, :, k2 +3], cmap = "gray")
            # 
            axes[0, 4].imshow(X.detach().numpy()[0, 0, :, :, k3], cmap = "jet")
            axes[0, 5].imshow(hat_X.detach().numpy()[0, 0, :, :, k3], cmap = "jet")
            
            axes[1, 4].imshow(X.detach().numpy()[0, 1, :, :, k3], cmap = "gray")
            axes[1, 5].imshow(hat_X.detach().numpy()[0, 1, :, :, k3], cmap = "gray")
        
        for a in range(axes.shape[0]) :
            for b in range(axes.shape[1]) :
                axes[a, b].set_axis_off()
            
    
    # plt.show()
    if mode is not None : 
        print(mode)
        plt.savefig(f"IMG/{mode}/{epoch:03d}.pdf")
        plt.show()
        plt.close()



        
