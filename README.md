# Predicting protein transmembrane topology from 3D structure
This project aims to train a neural network to predict transmembrane protein topologies. The neural network will be benchmarked against the existing best performing model DeepTMHMM. 

## Data
The data has been gathered from the DeepTMHMM project. This protein sequence data can be found in the DeepTMHMM.partitions.json.
We have done a lookup in the AlphaFold database to obtain the 3D structures. The 3D structures have then been fed through an encoder ESM-IF1 which transforms them into a latent representation with dimensions (Lx512) where L is the length of the protein sequence.

### Test data set
As the data set is too large to be uploaded to github a small sample can be found in encoder_proteins_test where the cvX is the Xth split of the data.

## Important files
- final_hand_in.ipynb - This is a jupyter notebook that has been used for training the neural network
- accuracy_b5.py - Used to calculate the accuracy of our model with regards to the labels and topology labels. This is the same accuracy calculations as used in DeepTMHMM
- DeepTMHMM.partitions.json - Is the data gathered from the DeepTMHMM project
- data_collect_felix_ver.py - The code uses the AlphaFold API to download the pdb files utilizing the DeepTMHMM data to get the same protein data. Given to us by our supervisor Felix Teufel
- create_latent_dataset.ipynb - After the data_collect_felix_ver.py has been run the proteins pdb files can be used to create the latent representations using this file
- encoder_proteins_test - Is a small sample of the encoded data which has been used for training