import pandas as pd
import numpy as np
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import sklearn

def plot_roc(name, labels, predictions, **kwargs):
  """ROC/AUC plot"""
  fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)
  mpl.rcParams['figure.figsize'] = (12, 10)
  colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
  plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
  plt.xlabel('False positives [%]')
  plt.ylabel('True positives [%]')
  plt.xlim([-0.5,100])
  plt.ylim([0,100.5])
  plt.grid(True)
  ax = plt.gca()
  ax.set_aspect('equal')


def line_chart(history):
    """Line chart showing model evaulation on different metrics"""
    history_df = pd.DataFrame(history.history).reset_index()
    history_df = history_df.rename(columns={'index':'epoch'})
    accuracy_df = pd.melt(history_df, id_vars=['epoch'], value_vars=['accuracy', 'precision','recall','auc'])
    loss_df = pd.melt(history_df, id_vars=['epoch'], value_vars=['loss'])
    history_df = pd.melt(history_df, id_vars=['epoch'], value_vars=['accuracy', 'precision','recall', 'auc', 'loss'])
    display(history_df.head())

    performanceChart = alt.Chart(history_df,title='Model performance').mark_line(size=3).encode(
        x=alt.X('epoch',axis=alt.Axis(title='Epoch', grid=False,tickCount=10)),
        y=alt.Y('value',axis=alt.Axis(title='Accuracy/Loss',format='%')),
        color=alt.Color('variable',scale=alt.Scale(range=['#6f0000','#696969','#ff7b7b','#CC5500','#458205']),
        legend=alt.Legend(title=None,labelFont='Arial',labelColor='#696969',labelFontSize=14))
        ).properties(
        width=800,
        height=300
        ).configure_title(fontSize=30,color='#232b2b',font='Arial',anchor='start',offset=20
        ).configure_axis(labelColor='#696969',labelFont='Arial',labelFontSize=14,titleFont='Arial',titleFontSize=16,titleColor='#696969'
        )#.configure_legend(labelFontStyle='Cambria',labelFontSize=12)
    
    #save as png

    return performanceChart

def plot_confusion_matrix_2(cm, normalize=True):
    """Confusion Matrix Plot"""
    group_names = ['True Negative (no fire)','False Positive (fire)','False Negative (no fire)','True Positive (fire)']
    group_counts = ["{0:0.0f}".format(value) for value in
                    cm.flatten()]
<<<<<<< HEAD

    TN = int(group_counts[0])
    FP = int(group_counts[1])
    FN = int(group_counts[2])
    TP = int(group_counts[3])
    
    # TNrate = (TN/((TN+FP)*100))
    # FPrate = (FP/((TN+FP)*100))
    # FNrate = (FN/((FN+TP)*100))
    # TPrate = (TP/((FN+TP)*100))

    # truth = (TNrate + TPrate) * 100

    TNrate = round(TN/((TN+FP)*100),2)
    FPrate = round(FP/((TN+FP)*100),2)
    FNrate = round(FN/((FN+TP)*100),2)
    TPrate = round(TP/((FN+TP)*100),2)

    group_percentages = [str(TN/(TN+FP)*100),str(FP/(TN+FP)*100),str(FN/(FN+TP)*100),str(TP/(FN+TP)*100)]
=======
    group_percentages = ["{0:.2%}".format(value) for value in
                         cm.flatten()/np.sum(cm)]
>>>>>>> 40254885c06ee55ca434d2df906bef5d3017cb3e

    if normalize:
        labels = [f"{v1}\n\n{v2}" for v1, v2 in
              zip(group_names,group_percentages)]
    else:
        labels = [f"{v1}\n\n{v2}" for v1, v2 in
              zip(group_names,group_counts)]

    # print(group_counts)

    labels = np.asarray(labels).reshape(2,2)

    # print(labels[0][0])

    ax = sns.heatmap(cm, annot=labels, fmt='', cmap='Reds')
    
    sns.set_style('whitegrid', {'font.family':'arial', 'font.arial':'Arial Narrow'})

    ax.set_title('Image confusion matrix\n',fontdict = { 'fontsize': 16});
    ax.set_xlabel('\nPredicted Values',fontdict = { 'fontsize': 12})
    ax.set_ylabel('Actual Values ',fontdict = { 'fontsize': 12});

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])
    
    ax.figure.tight_layout()

    plt.ylim(2, 0) # update the ylim(bottom, top) values
    plt.show() # ta-da!

<<<<<<< HEAD
    return TNrate, FPrate, FNrate, TPrate

# def plot_roc(name, labels, predictions, **kwargs):
#   fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)

#   plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
#   plt.xlabel('False positives [%]')
#   plt.ylabel('True positives [%]')
#   plt.xlim([-0.5,50])
#   plt.ylim([80,100.5])
#   plt.grid(True)
#   ax = plt.gca()
#   ax.set_aspect('equal')
=======
>>>>>>> 40254885c06ee55ca434d2df906bef5d3017cb3e

def plot_metrics(history):
  """plot of different metrics"""
  metrics = ['loss', 'prc', 'precision', 'recall']
  for n, metric in enumerate(metrics):
    name = metric.replace("_"," ").capitalize()
    plt.subplot(2,2,n+1)
    plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
    plt.plot(history.epoch, history.history['val_'+metric],
             color=colors[0], linestyle="--", label='Val')
    plt.xlabel('Epoch')
    plt.ylabel(name)
    if metric == 'loss':
      plt.ylim([0, plt.ylim()[1]])
    elif metric == 'auc':
      plt.ylim([0.8,1])
    else:
      plt.ylim([0,1])

    plt.legend()

def plot_images(images, labels):
    """plot images for display"""
    import math
    p_size = 5
    class_names = ['no_fire','fire']
    plt.figure(figsize=(10,6))
    for i in range(p_size**2):
        plt.subplot(p_size,p_size,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i])
        plt.xlabel(class_names[int(labels[i][0])])

def visualize(original, augmented):
  """viualize augmentation plot"""
  p_size = 4

  fig = plt.figure(figsize=(15,15))
  plt.subplot(1,2,1)
  plt.title('Original image')
  plt.imshow(original)

  plt.subplot(1,2,2)
  plt.title('Augmented image')
  plt.imshow(augmented)



