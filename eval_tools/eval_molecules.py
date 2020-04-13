# Sultan Kenjeyev
import argparse
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

from joblib import Parallel, delayed
from tqdm import tqdm


def calc_mol_activity(mol, eval_model, target_idx=430):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    res = np.zeros(len(fp), np.int32)
    DataStructs.ConvertToNumpyArray(fp, res)

    probabilities = list(eval_model.predict_proba(res.reshape(1, -1))[0])
    smiles = Chem.MolToSmiles(mol)

    return smiles, probabilities[target_idx]


def predict_probability(new_mols, eval_model, n_jobs=-1, target_id='CHEMBL262'):
    """
    parallel function to predict the probability that a molecule will be active on a target.
    :param new_mols: list of rdkit.Mol
    :param eval_model: numpy bayesian model to predict probability of activity
    :param n_jobs: how many processes to spawn, n_jobs=-2 meaning leave 1 cpu alone
    :param target_id: ChemBL id of target, i.e. CHEMBL262 is GCK3Beta
    :return:
    """

    targets = list(eval_model.targets)
    target_idx = targets.index(target_id)

    output = Parallel(n_jobs=n_jobs)(delayed(calc_mol_activity)(mol, eval_model, target_idx) for mol in tqdm(new_mols))

    results = {}

    for i, p in output:
        results[i] = p

    return results


def get_args():
    parser = argparse.ArgumentParser(
        description='Evaluating molecules activity and Tanimotot distance suing a bayesian model')
    parser.add_argument('--model_path', default='./chembl_25/models/1uM/mNB_1uM_all.pkl',
                        help='path to bayesian model to be used for evaluation')
    parser.add_argument('--rep_smiles', default='./gsk_reps.smi', help='molecules that are active against your target')
    parser.add_argument('--output_file', default='./eval_results.csv', help='file for evaluation output')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
