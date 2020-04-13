## by Giovanni Bottegoni - g.bottegoni@bham.ac.uk

## This is a simple script to evaluate molecules generated as putatively active vs. GSK-3beta
## from two different angles:
##
## 1) a prediction of activity at the target produced by a Bayesian model for activity
##    More info about the model can be found here:
##    http://chembl.blogspot.com/2014/04/ligand-based-target-predictions-in.html
##
##    There actually are two  models for each target: i) probability of activity at 1 micromolar and ii) at 10 micromolar
##    Here I have only considered the more conservative 1uM model but it would be easy to include the other one
##
## 2) an assessment of the chemical distance (Tanimoto distance between molecular fingerprints)
##    between each generated molecule and a representative set of 100 GSK-3beta known inhibitors
##    extracted from the training set through a MaxMin procedure based on chemical features.
##    In principle, unless the calculation is not exceedingly slow, the representative set should be dropped
##    and the same assessment performed on the entire training set
##   
## Together these two values return a classic 'four quadrants' situation:
##
## A) [High chemical Distance, Low Predicted Activity Probability] this is somewhat expected. These molecules do NOT resemble 
##    known GSK-3beta inhibitors and the model assigns them a low probability
##
## B) [Low chemical distance, High Predicted Activity Probability] this is somewhat expected. These molecules are trivially similar
##    to those included in the training set and the Bayesian model assigns them high probability. Even though there are
##    exceptions, one of the general tenets of medicinal chemistry is that similar structures will display similar activities 
##
## C) [Low chemical Distance, Low Predicted Activity Probability] I would ignore these for the time being. These results 
##    are most likely due to the limits of Bayesian models in managing outliers in their training set.
## 
## D) [High chemical distance, High Predicted Activity Probability] This is the quadrant we are interested in. The ML strategy that
##    more efficiently populates this quadrant is the more interesting one. Basically, the 'chemical space' is extremely vast
##    and there are many possibilities to combine atoms in ways that are different from the structures in the training set 
##    yet returning compounds that are recognised as putative active molecules by the Bayesian model. 
##    NOTE:  This is why it would be important to perform the chemical similarity assessment of the entire set and not just 
##           on representatives (see (2) above)
##


## Install the Bayesian models - ChEMBL25 being the most recent version of the database
##    http://ftp.ebi.ac.uk/pub/databases/chembl/target_predictions/ 

model_path = '/home/giova/COMPCHEM/SMILES/chembl_25/models/1uM/mNB_1uM_all.pkl'

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import PandasTools
from rdkit import DataStructs
import pandas as pd
from pandas import concat
from collections import OrderedDict
import requests
import numpy
from sklearn.externals import joblib
from rdkit import rdBase


def calc_scores(classes):
    p = []
    for c in classes:
        p.append(pred_score(c))
    return p


def pred_score(trgt):
    diff = morgan_nb.estimators_[classes.index(trgt)].feature_log_prob_[1] - \
           morgan_nb.estimators_[classes.index(trgt)].feature_log_prob_[0]
    return sum(diff * fp)


## Install somewhere the list of SMILES from the GSK-3beta representatives
## or change the reference to a list of smiles encompassing the entire training set

gsk_reps = Chem.SmilesMolSupplier('gsk_reps.smi', delimiter='\t', titleLine=False)

## The Bayesian assessment is largely taken from:
## https://nbviewer.jupyter.org/gist/madgpap/10457778
## Please note that to make it work, several changes had to be introduced.
## See also:
## https://iwatobipen.wordpress.com/2017/04/07/target-prediction-using-chembl/

morgan_nb = joblib.load(model_path)

classes = list(morgan_nb.targets)

gsk3b_probability = []
gsk3b_maxdistances = []

## I have only been testing the script on small files and not on the entire 1000-ish set
## I would not expect any issue but you never know
## At the moment the code does NOT handle exceptions, so any issue with a specific molecule
## will cause the code to hang  

# suppl = Chem.SmilesMolSupplier('molecules.smi',delimiter='\t',titleLine=False)
suppl = Chem.SmilesMolSupplier('small_set.smi', delimiter='\t', titleLine=False)

for mol in suppl:

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    res = numpy.zeros(len(fp), numpy.int32)
    DataStructs.ConvertToNumpyArray(fp, res)

    probas = list(morgan_nb.predict_proba(res.reshape(1, -1))[0])

    predictions = pd.DataFrame(list(zip(classes, calc_scores(classes), probas)), columns=['id', 'score', 'proba'])

    xxx = predictions.loc[predictions['id'] == "CHEMBL262"]
    myvalue = float(xxx['proba'])
    gsk3b_probability.append(myvalue)

    tmp_distance = []

    for rep in gsk_reps:
        repfp = AllChem.GetMorganFingerprintAsBitVect(rep, 2, nBits=2048)
        pair_tmp = DataStructs.TanimotoSimilarity(repfp, fp)

        tmp_distance.append(1 - pair_tmp)

    gsk3b_maxdistances.append(min(tmp_distance))
    tmp_distance = []

outcome = pd.DataFrame(list(zip(gsk3b_probability, gsk3b_maxdistances)), columns=['Probability', 'Distances'])

outcome.to_csv("evaluation_new_molecule.csv", sep='\t')

## This code has NOT being debugged in any possible way. I am actually surprised it works at all!
