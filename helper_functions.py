import matplotlib.pyplot as plt 

def add_labels_top_of_barchart(x, y):
    """
    Fügt Beschriftungen oben an den Balken eines Balkendiagramms hinzu.

    Parameter:
    -----------
    x : list oder array
        Die x-Werte (Kategorien) der Balken im Diagramm.
    y : list oder array
        Die y-Werte (Höhen) der Balken, die beschriftet werden sollen.
    """
    # Rundet die y-Werte auf zwei Dezimalstellen
    y = y.round(2)
    # Fügt für jeden Balken eine Textbeschriftung hinzu
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha='center')  # 'ha' steht für horizontale Ausrichtung ('center' zentriert den Text über dem Balken)

def add_label_stacked_bar_chart(axis): 
    """
    Fügt Beschriftungen zu einem gestapelten Balkendiagramm hinzu.

    Parameter:
    -----------
    axis : matplotlib.axis.Axis
        Die Achse, die das gestapelte Balkendiagramm enthält.
    """
    # Iteriert über alle Balkensegmente im Diagramm
    for c in axis.containers:

        # Optional: Wenn das Segment klein oder 0 ist, wird das Label angepasst
        labels = [v.get_height() if v.get_height() > 0 else '' for v in c]
        
        # Fügt die Labels in der Mitte jedes Balkensegments hinzu
        axis.bar_label(c, labels=labels, label_type='center')


# Source: https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
import numpy as np 
from scipy.cluster.hierarchy import dendrogram

def plot_dendrogram(model, labels=None, **kwargs):
    """
    Erstellt und visualisiert ein Dendrogramm basierend auf einem hierarchischen Clustering-Modell.

    Diese Funktion generiert eine Verknüpfungsmatrix (linkage matrix) aus einem gegebenen 
    hierarchischen Clustering-Modell und verwendet diese, um ein Dendrogramm zu plotten, das die 
    hierarchische Struktur der Cluster visualisiert.

    Parameter:
    -----------
    model : sklearn.cluster._agglomerative.AgglomerativeClustering
        Das trainierte hierarchische Clustering-Modell, das die Informationen über die 
        Clusterzusammenführungen enthält.
    
    labels : list, optional
        Eine Liste von Beschriftungen, die den einzelnen Datenpunkten entsprechen. Diese werden im 
        Dendrogramm angezeigt. Standardmäßig werden die Indizes der Datenpunkte verwendet, wenn keine 
        Labels angegeben sind.

    **kwargs : dict
        Zusätzliche Argumente, die an die `dendrogram`-Funktion von SciPy weitergegeben werden. 
        Diese können verwendet werden, um das Aussehen und Verhalten des Dendrogramms anzupassen.

    Returns:
    --------
    None
        This function plots the dendrogram directly using matplotlib and does not return any value.
    """
    
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, labels=labels, **kwargs)
