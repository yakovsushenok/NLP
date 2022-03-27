import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_class_im(df):
    index_names = df[ df['genre'] == "pop" ][:2900].index
    df.drop(index_names, inplace = True)
    df = df.reset_index(drop=True)
    index_names = df[ df['genre'] == "blues" ][:500].index
    df.drop(index_names, inplace = True)
    df = df.reset_index(drop=True)
    index_names = df[ df['genre'] == "country" ][:1400].index
    df.drop(index_names, inplace = True)
    df = df.reset_index(drop=True)
    return df

def update_results(results,experiment_results,i):
    if results == None:
        return experiment_results
    else:
        for key, value in results.items():
            results[key] = (1/(i+1))*( (i)*results[key] + experiment_results[key])
    return results


def save_class_breakdown(df):
    df['genre'].value_counts().plot.bar()
    df = df.reset_index(drop=True)
    plt.savefig("classes.png")

def save_conf_mat(config,dict_,mapping):
    ax = plt.axes()
    sns.heatmap(dict_, xticklabels = mapping['genre'].keys(),yticklabels = mapping['genre'].keys(),ax=ax)
    ax.set_title(config['Tasks'])
    plt.xlabel ("Target")
    plt.ylabel ("Predicted")
    plt.savefig("confusion_mat.png") 