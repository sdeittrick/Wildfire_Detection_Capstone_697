import pandas as pd
import numpy as np
import altair as alt


def line_chart(history):

    history_df = pd.DataFrame(history.history).reset_index()
    history_df = history_df.rename(columns={'index':'epoch'})
    accuracy_df = pd.melt(history_df, id_vars=['epoch'], value_vars=['accuracy', 'val_accuracy'])
    loss_df = pd.melt(history_df, id_vars=['epoch'], value_vars=['loss', 'val_loss'])
    history_df = pd.melt(history_df, id_vars=['epoch'], value_vars=['accuracy', 'val_accuracy', 'loss', 'val_loss'])

    performanceChart = alt.Chart(history_df,title='Model performance').mark_line(size=3).encode(
        x=alt.X('epoch',axis=alt.Axis(title='Epoch', grid=False,tickCount=10)),
        y=alt.Y('value',axis=alt.Axis(title='Accuracy/Loss',format='%')),
        color=alt.Color('variable',scale=alt.Scale(range=['#6f0000','#696969','#ff7b7b','#999999']),
        legend=alt.Legend(title=None,labelFont='Cambria',labelColor='#696969',labelFontSize=14))
        ).properties(
        width=800,
        height=300
        ).configure_title(fontSize=30,color='#232b2b',font='Cambria',anchor='start',offset=20
        ).configure_axis(labelColor='#696969',labelFont='Cambria',labelFontSize=14,titleFont='Cambria',titleFontSize=16,titleColor='#696969'
        )#.configure_legend(labelFontStyle='Cambria',labelFontSize=12)
    
    #save as png

    return performanceChart

def plot_confusion_matrix_2(cm, normalize=True):
    
    group_names = ['True Negative (no fire)','False Positive (fire)','False Negative (no fire)','True Positive (fire)']

    group_counts = ["{0:0.0f}".format(value) for value in
                    cm.flatten()]

    group_percentages = ["{0:.2%}".format(value) for value in
                         cm.flatten()/np.sum(cm)]

#     labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
#               zip(group_names,group_counts,group_percentages)]

    if normalize:
        labels = [f"{v1}\n\n{v2}" for v1, v2 in
              zip(group_names,group_percentages)]
    else:
        labels = [f"{v1}\n\n{v2}" for v1, v2 in
              zip(group_names,group_counts)]

    labels = np.asarray(labels).reshape(2,2)

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