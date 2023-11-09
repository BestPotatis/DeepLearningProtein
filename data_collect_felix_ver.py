import os
import requests

def get_alphafold_db_pdb(protein_id: str, out_path: str) -> bool:

    """
    With the uniprot id, get the AF PDB from the DB.
    """

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    requestURL = f"https://alphafold.ebi.ac.uk/files/AF-{protein_id}-F1-model_v4.pdb"
    r = requests.get(requestURL)

    if r.status_code == 200:
        with open(out_path, "wb") as f:
            f.write(r.content)
            return True
    else:
        return False
    
if __name__ == "__main__":
    get_alphafold_db_pdb("P0DTU3", "proteins/P0DTU3")