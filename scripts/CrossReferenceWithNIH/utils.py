# Utilities for the cross reference task
import matplotlib.pyplot as plt
import numpy as np

def draw_confusion_matrix(cm, row_labels, col_labels, xlabel='Column Categories',ylabel='Row Categories',title="",figsize=(14,14)):
    assert(len(row_labels)==cm.shape[0])
    assert(len(col_labels)==cm.shape[1])
    fig=plt.figure(figsize=figsize) #/len(col_labels)*len(row_labels)+5))
    cm = np.asarray(cm)
    normalized=np.abs(np.sum(cm)-1)<1.e-4

    #po = np.get_printoptions()
    #np.set_printoptions(precision=2)
    ax=plt.gca()
    ax.imshow(cm, cmap='Oranges',vmin=0.0)

    x_ticks = np.arange(len(col_labels))
    y_ticks = np.arange(len(row_labels))

    ax.set_xlabel(xlabel)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.set_xlim(reversed(ax.get_xlim()))

    ax.set_ylabel(ylabel)
    ax.set_yticks(y_ticks)
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()
    ax.set_ylim([-0.50,len(row_labels)-0.5])

    #ax.set_xticklabels(col_labels, rotation=-90,  ha='center')
    ax.set_xticklabels(col_labels, ha='center')
    ax.set_yticklabels(row_labels, va ='center')

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            if cm[i,j] == 0:
                txt = '.'
            elif normalized:            
                txt = '{:.2f}'.format(cm[i,j])
            else:
                txt = '{}'.format(int(cm[i,j]))
            ax.text(j, i, txt, horizontalalignment="center", verticalalignment='center', color= "black", fontsize=14)

    #np.set_printoptions(**po)
    if len(title)>0:
        plt.title(title)
    plt.show()
    plt.close(fig)


