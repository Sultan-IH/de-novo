import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.pyplot import figure
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit import DataStructs
import numpy as np

from tqdm import tqdm

def visualise_activity(activities):
    """
    Draws a distribution of activities, also shows values for mean and max
    :param activities:
    :return:
    """
    # Create a figure instance
    fig = plt.figure()

    # Create an axes instance
    ax = fig.add_axes([0, 0, 1, 1])

    # Create the boxplot
    bp = ax.violinplot(activities)

    s = f'mean={activities.mean()}\n max={max(activities)}'
    plt.text(0.2, 0.9, s, ha='center', va='center',
             transform=ax.transAxes,
             bbox=dict(facecolor='grey', alpha=0.5))
    ax.set_title('Activity plot')

    plt.xlabel('variation')
    plt.ylabel('probability of activity')

    plt.show()


def visualise_similarity(mol, rep_set, title):
    """
    visualises a distribution of Tanimoto distances from a representatives set
    molecules are considered similar if Tanimoto distance > 0.85

    :param mol:
    :param rep_set:
    :return: matplotlib figure
    """
    # get tanimoto distances
    generated_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    tan_distances = []
    for rep_mol in tqdm(rep_set):
        if rep_mol is None:
            print('got None molecule, skipping ... ')
            continue
        repfp = AllChem.GetMorganFingerprintAsBitVect(rep_mol, 2, nBits=2048)
        pair_tmp = DataStructs.TanimotoSimilarity(repfp, generated_fp)
        tan_distances.append(pair_tmp)



    ax = sns.distplot(tan_distances, hist=True, kde=True,
                 kde_kws={'shade': True, 'linewidth': 3})
    ax.set(xlim=(0,1))
    plt.title(title)


def visualise_mols(mols, m_per_row=2, imgsize=(300, 280), labels=[]):
    """
    visualises a set of molecules as a grid
    :param mols: set of molecules
    :return: PIL img
    """
    img = Draw.MolsToGridImage(mols, molsPerRow=m_per_row,
                               subImgSize=imgsize, legends=labels)
    return img
