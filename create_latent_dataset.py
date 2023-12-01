#!/usr/bin/env python
# coding: utf-8

# In[5]:


# load modules
import os
import json
import numpy as np
import esm
from tqdm import tqdm


# In[2]:


# import model
model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
model = model.eval()


# In[3]:


# import labels
f = open("DeepTMHMM.partitions.json")
labels = json.load(f)


# In[6]:


# get representation structure for all proteins
for cv in tqdm(labels.keys()):
    for protein in tqdm(labels[cv]):

        path = f"/zhome/7a/e/167745/deeplearning/proteins/{cv}/{protein['id']}.pdb"
        encoder_path = f"/zhome/7a/e/167745/deeplearning/encoder_proteins/{cv}/{protein['id']}"
        
        # if protein does not exists in alphafold-db, skip over
        if not os.path.exists(path) or os.path.exists(encoder_path + ".npy"):
            continue
        
        data = {}
        
        # receive 3-D structure (Atom array (of amino acids) including 3-D coordinates and other info)        
        structure = esm.inverse_folding.util.load_structure(path)
        
        # get coordinates for each amino acid's N-terminal, alpha-carbon and C-terminal (first three in pdb)
        coords, native_seq = esm.inverse_folding.util.extract_coords_from_structure(structure)

        # get encoder output as structure representation shape: (amino acid, encoder dimension)
        rep = esm.inverse_folding.util.get_encoder_output(model, alphabet, coords)
        
        data["data"] = rep.detach().numpy()
        data["labels"] = protein["labels"]
        
        # create directory and save data as .npy file
        os.makedirs(os.path.dirname(encoder_path), exist_ok=True)
        np.save(encoder_path, data)
        #print("saved data for: ", protein['id'])


# In[ ]:


# # example of loaded dataset
# read_dictionary = np.load(encoder_path + ".npy", allow_pickle='TRUE').item()
# print(read_dictionary) # displays "world"

