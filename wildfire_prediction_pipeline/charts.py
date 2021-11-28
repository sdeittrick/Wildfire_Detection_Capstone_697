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
