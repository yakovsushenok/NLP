"""
    file : utils
    authors : 21112254, 16008937, 20175911, 21180859

"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_class_im(df):
    """
    Process dataframe to balance classes
    
    Parameters
    ----------
    df : DataFrame
        data

    Returns
    -------
    df: DataFrame with instances of classes removed

    """
    index_names = df[ df['genre'] == "pop" ][:3042].index
    df.drop(index_names, inplace = True)
    df = df.reset_index(drop=True)
    index_names = df[ df['genre'] == "blues" ][:604].index
    df.drop(index_names, inplace = True)
    df = df.reset_index(drop=True)
    index_names = df[ df['genre'] == "country" ][:1445].index
    df.drop(index_names, inplace = True)
    df = df.reset_index(drop=True)
    index_names = df[ df['genre'] == "rock" ][:34].index
    df.drop(index_names, inplace = True)
    df = df.reset_index(drop=True)
    index_names = df[ df['genre'] == "hip hop" ].index
    df.drop(index_names, inplace = True)
    df = df.reset_index(drop=True)
    index_names = df[ df['genre'] == "reggae" ].index
    df.drop(index_names, inplace = True)
    df = df.reset_index(drop=True)
    return df

def update_results(results,experiment_results,i):
    """
    Update results dict
    
    Parameters
    ----------
    results : Dict
        Dictionary of current results
    experiment_results : Dict
        Dictionary of most recent epoch results
    i: Integer
        Epoch index

    Returns
    -------
    dict: Updated results dict

    """
    if results == None:
        results = {}
        for key, value in experiment_results.items():
            results[key] = experiment_results[key] - experiment_results[key]
        return {'mean':experiment_results,'var': results}
    else:
        for key, value in results['var'].items():
            results['var'][key] = ((i+1)/(i+2)) * (results['var'][key] + (((results['mean'][key]-experiment_results[key])**2)/(i+2)))
        for key, value in results['mean'].items():
            results['mean'][key] = (1/(i+1))*( (i)*results['mean'][key] + experiment_results[key])
    return results


def save_class_breakdown(df):
    """
    Save image of class breakdown for df

    """
    df['genre'].value_counts().plot.bar(color='orchid')
    plt.xlabel('Genre',fontsize=12)
    plt.ylabel('Count',fontsize=12)
    plt.xticks(rotation = 0,fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig("post_classes.png")

def save_conf_mat(config,dict_,mapping):
    """
    Save image of confusion matrix
    
    """
    ax = plt.axes()
    sns.heatmap(dict_, xticklabels = mapping['genre'].keys(),yticklabels = mapping['genre'].keys(),ax=ax)
    ax.set_title(config['Tasks'])
    plt.xlabel ("Target")
    plt.ylabel ("Predicted")
    plt.savefig("_base_confusion_mat.png") 