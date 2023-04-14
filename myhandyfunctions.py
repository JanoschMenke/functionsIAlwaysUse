from matplotlib import pyplot as plt
from matplotlib import rc

from rdkit.Chem import AllChem as Chem
from tqdm import tqdm 
import pandas as pd
import multiprocessing as mp
from rdkit.Chem import Descriptors
import numpy as np


def setPlotSettings():
    RED = "#e4897b"
    BLUE = "#5aa8c8"
    GREEN = "#82C163"
    YELLOW = "#f4bb4d"

    myPalette = [
        RED,
        BLUE,
        GREEN,
        YELLOW,
        "#ffffff00",
        "#757575ff",
    ]
    sns.set_palette(sns.color_palette(myPalette))
    plt.rc("axes.spines", top=False, right=False)
    rc("font", **{"family": "monospace"})
    rc("text", usetex=False)
    return RED, BLUE, GREEN, YELLOW

class compute_ecfp():
    """
    Quick and Dirty class, used to calculate the ECFP on multiple cores.
    """
    
    def __init__(self, bitSize=2048, radius=2):        
        self.bitSize = bitSize
        self.radius = radius
        
    def get_single_fp(self,smile):
        """Computes a single fingerprint"""
        try:
            fp = Chem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smile),self.radius,nBits=self.bitSize)
            return np.array(fp, dtype = np.int8)
        except:
            return np.zeros(self.bitSize, np.int8) #zero vector is return of invalid smiles
    def get_fingerprints(self,smiles_list,ncores=1):
        """Computes the fingerprint  for a list of smiles"""
        pool = mp.Pool(processes=ncores)
        fingerprints = np.stack(pool.map(self.get_single_fp,smiles_list))
        pool.close()
        pool.join()
        return fingerprints



def saveFigure(path,**kwargs):
  """quickly save plots in the most important file formats"""
  plt.savefig(f"{path}.png",format = "png", **kwargs)
  plt.savefig(f"{path}.pdf",format = "pdf", **kwargs)
  plt.savefig(f"{path}.svg",format = "svg", **kwargs)  
 

def canonicalizeSmiles(mol, stereoInformation = False):
  return Chem.MolToSmiles(Chem.MolFromSmiles(mol), isomericSmiles=stereoInformation)
