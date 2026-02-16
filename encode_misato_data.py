"""
Standalone script to perform atom one-hot encoding for ligands and backbone encoding for proteins
from MISATO MD.hdf5 data files.

Usage:
    python encode_misato_data.py --root_dir path/to/MISATO_100 --mode train --pdb_id 1ABC
    python encode_misato_data.py --root_dir path/to/MISATO_100 --mode train --process_all
"""

import os
import sys
import argparse
import numpy as np
import h5py
import pickle
import torch
import pandas as pd
import importlib.resources

# Add NeuralMD to path if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'NeuralMD'))

from NeuralMD.datasets.MISATO.common import extract_backbone
import NeuralMD.datasets


def load_utility_files():
    """Load all required utility files for encoding"""
    # Get utils directory
    utils_dir = os.path.join(
        os.path.dirname(__file__), 
        'NeuralMD', 'datasets', 'MISATO', 'utils'
    )
    
    if not os.path.exists(utils_dir):
        raise FileNotFoundError(
            f"Utils directory not found: {utils_dir}\n"
            "Make sure you're running from the NeuralMD root directory."
        )
    
    # Load pickle files
    residue_index2name_dict = pickle.load(
        open(os.path.join(utils_dir, 'atoms_residue_map.pickle'), 'rb')
    )
    protein_atom_index2standard_name_dict = pickle.load(
        open(os.path.join(utils_dir, 'atoms_type_map.pickle'), 'rb')
    )
    atom_reisdue2standard_atom_name_dict = pickle.load(
        open(os.path.join(utils_dir, 'atoms_name_map_for_pdb.pickle'), 'rb')
    )
    
    # Load periodic table
    with importlib.resources.path(NeuralMD.datasets, 'periodic_table.csv') as file_name:
        periodic_table_data = pd.read_csv(file_name)
    
    atom_num2atom_mass = {}
    for i in range(1, 119):
        atom_mass = periodic_table_data.loc[i-1]['AtomicMass']
        atom_num2atom_mass[i] = atom_mass
    
    # Atomic number to name mapping (for ligands)
    atom_index2name_dict = {
        1: 'H', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 
        11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 
        16: 'S', 17: 'Cl', 19: 'K', 20: 'Ca', 34: 'Se', 
        35: 'Br', 53: 'I'
    }
    
    return {
        'residue_index2name_dict': residue_index2name_dict,
        'protein_atom_index2standard_name_dict': protein_atom_index2standard_name_dict,
        'atom_reisdue2standard_atom_name_dict': atom_reisdue2standard_atom_name_dict,
        'atom_num2atom_mass': atom_num2atom_mass,
        'atom_index2name_dict': atom_index2name_dict,
    }


def encode_single_pdb_entry(root_dir, pdb_id, utils_dict):
    """
    Encode a single PDB entry from MD.hdf5
    
    Args:
        root_dir: Path to MISATO_100 directory
        pdb_id: PDB ID string (e.g., "1ABC")
        utils_dict: Dictionary containing utility mappings
    
    Returns:
        Dictionary containing encoded ligand and protein data
    """
    # Open MD.hdf5
    MD_file_path = os.path.join(root_dir, "raw", "MD.hdf5")
    if not os.path.exists(MD_file_path):
        raise FileNotFoundError(f"MD.hdf5 not found at {MD_file_path}")
    
    MD_data = h5py.File(MD_file_path, "r")
    
    if pdb_id not in MD_data:
        MD_data.close()
        raise KeyError(f"PDB ID {pdb_id} not found in MD.hdf5")
    
    misato_data = MD_data.get(pdb_id)
    
    # Get ligand start index
    ligand_begin_index = misato_data["molecules_begin_atom_index"][:][-1]
    
    # ========== LIGAND ENCODING (One-Hot) ==========
    atom_index = misato_data["atoms_number"][:]
    ligand_atom_index = atom_index[ligand_begin_index:]
    
    # Filter out hydrogen atoms (atomic number = 1)
    valid_ligand_atoms_mask = ligand_atom_index != 1
    ligand_atom_index = ligand_atom_index[valid_ligand_atoms_mask]
    
    # Convert to 0-indexed (for one-hot encoding: 0-117)
    ligand_atom_index -= 1
    
    # Get ligand masses (use numpy array values, not tensor)
    # ligand_atom_index is still a numpy array at this point
    ligand_atoms_mass = torch.tensor(
        [utils_dict['atom_num2atom_mass'][int(atom_num) + 1] for atom_num in ligand_atom_index],
        dtype=torch.float32
    )
    
    # Convert to torch tensor
    ligand_atom_index = torch.tensor(ligand_atom_index, dtype=torch.int64)
    
    # ========== PROTEIN BACKBONE ENCODING ==========
    protein_atom_index = misato_data["atoms_type"][:ligand_begin_index]
    residue_index = misato_data["atoms_residue"][:ligand_begin_index]
    atom_index_protein = misato_data["atoms_number"][:ligand_begin_index]
    molecules_begin_atom_index = misato_data["molecules_begin_atom_index"][:][-1:]
    
    # Extract backbone using the exact function from the codebase
    try:
        mask_backbone, mask_ca, mask_c, mask_n = extract_backbone(
            protein_atom_index, 
            residue_index, 
            atom_index_protein,
            utils_dict['protein_atom_index2standard_name_dict'],
            utils_dict['residue_index2name_dict'],
            utils_dict['atom_reisdue2standard_atom_name_dict'],
            utils_dict['atom_index2name_dict'],
            molecules_begin_atom_index
        )
    except AssertionError as e:
        # Check what the actual counts are for better error message
        from NeuralMD.datasets.MISATO.common import get_atom_name
        try:
            protein_atom_index_names = get_atom_name(
                protein_atom_index, residue_index, atom_index_protein,
                utils_dict['protein_atom_index2standard_name_dict'],
                utils_dict['residue_index2name_dict'],
                utils_dict['atom_reisdue2standard_atom_name_dict'],
                utils_dict['atom_index2name_dict'],
                molecules_begin_atom_index
            )
            mask_backbone_temp = (protein_atom_index_names == "CA") | (protein_atom_index_names == "C") | (protein_atom_index_names == "N")
            protein_backbone_atom_type = protein_atom_index_names[mask_backbone_temp]
            num_ca = np.sum(protein_backbone_atom_type == "CA")
            num_c = np.sum(protein_backbone_atom_type == "C")
            num_n = np.sum(protein_backbone_atom_type == "N")
            raise AssertionError(
                f"Backbone atom counts don't match for {pdb_id}: "
                f"CA={num_ca}, C={num_c}, N={num_n}. "
                f"All should be equal. This may indicate incomplete protein structure."
            ) from e
        except:
            # If we can't get detailed info, just re-raise with pdb_id
            raise AssertionError(f"Backbone extraction failed for {pdb_id}: {str(e)}") from e
    
    # Get coordinates
    trajectory_coordinates = misato_data["trajectory_coordinates"][:]  # (100, num_atom, 3)
    trajectory_coordinates = np.transpose(trajectory_coordinates, (1, 0, 2))  # (num_atom, 100, 3)
    
    # Center coordinates
    trajectory_coordinates_flattened = trajectory_coordinates.reshape(-1, 3)
    trajectory_pos_center = np.sum(trajectory_coordinates_flattened, axis=0) / trajectory_coordinates_flattened.shape[0]
    trajectory_coordinates = trajectory_coordinates - trajectory_pos_center
    
    # Get first frame protein coordinates
    protein_coordinates = trajectory_coordinates[:ligand_begin_index, 0]
    
    # Extract backbone coordinates
    protein_backbone_coordinates = protein_coordinates[mask_backbone, :]
    protein_backbone_coordinates = torch.tensor(protein_backbone_coordinates, dtype=torch.float32)
    
    # Get residue indices for CA atoms only (one per residue)
    protein_backbone_residue = residue_index[mask_backbone][mask_ca]
    protein_backbone_residue = torch.tensor(protein_backbone_residue, dtype=torch.int64)
    
    # Check residue index assertion (from original code)
    if protein_backbone_residue.min() < 1:
        raise ValueError(
            f"Residue indices must be >= 1 before conversion. "
            f"Found min={protein_backbone_residue.min().item()} for {pdb_id}"
        )
    
    protein_backbone_residue -= 1  # Convert to 0-indexed
    
    # Store masks
    mask_ca = torch.tensor(mask_ca, dtype=torch.bool)
    mask_c = torch.tensor(mask_c, dtype=torch.bool)
    mask_n = torch.tensor(mask_n, dtype=torch.bool)
    
    # Get ligand trajectory
    ligand_trajectory_coordinates = trajectory_coordinates[ligand_begin_index:][valid_ligand_atoms_mask]
    ligand_trajectory_coordinates = torch.tensor(ligand_trajectory_coordinates, dtype=torch.float32)
    
    # Get energy
    frames_interaction_energy = np.expand_dims(misato_data["frames_interaction_energy"][:], 0)
    frames_interaction_energy = torch.tensor(frames_interaction_energy, dtype=torch.float32)
    
    MD_data.close()
    
    return {
        # Ligand encoding (for one-hot)
        'ligand_x': ligand_atom_index,  # Atomic indices (0-117) - ready for one-hot encoding
        'ligand_mass': ligand_atoms_mass,
        'ligand_trajectory_pos': ligand_trajectory_coordinates,  # [num_ligand_atoms, 100, 3]
        
        # Protein backbone encoding
        'protein_pos': protein_backbone_coordinates,  # [num_backbone_atoms, 3]
        'protein_backbone_residue': protein_backbone_residue,  # [num_residues]
        'mask_ca': mask_ca,  # Boolean mask for CA atoms
        'mask_c': mask_c,    # Boolean mask for C atoms
        'mask_n': mask_n,    # Boolean mask for N atoms
        
        # Energy
        'energy': frames_interaction_energy,  # [1, 100]
    }


def print_encoding_summary(encoded_data, pdb_id):
    """Print summary of encoded data"""
    print(f"\n{'='*60}")
    print(f"Encoding Summary for PDB ID: {pdb_id}")
    print(f"{'='*60}")
    
    print(f"\nLIGAND ENCODING (One-Hot):")
    print(f"  Number of ligand atoms (non-H): {encoded_data['ligand_x'].shape[0]}")
    print(f"  Atomic indices range: {encoded_data['ligand_x'].min().item()} - {encoded_data['ligand_x'].max().item()}")
    print(f"  First 10 atomic indices: {encoded_data['ligand_x'][:10].tolist()}")
    print(f"  Ligand trajectory shape: {encoded_data['ligand_trajectory_pos'].shape}")
    
    print(f"\nPROTEIN BACKBONE ENCODING:")
    print(f"  Number of backbone atoms: {encoded_data['protein_pos'].shape[0]}")
    print(f"  Number of residues: {encoded_data['protein_backbone_residue'].shape[0]}")
    print(f"  CA atoms: {encoded_data['mask_ca'].sum().item()}")
    print(f"  C atoms: {encoded_data['mask_c'].sum().item()}")
    print(f"  N atoms: {encoded_data['mask_n'].sum().item()}")
    print(f"  Backbone coordinates shape: {encoded_data['protein_pos'].shape}")
    
    print(f"\nENERGY:")
    print(f"  Energy shape: {encoded_data['energy'].shape}")
    print(f"  Energy range: {encoded_data['energy'].min().item():.4f} - {encoded_data['energy'].max().item():.4f}")
    
    # Show one-hot encoding example
    if encoded_data['ligand_x'].shape[0] > 0:
        ligand_one_hot = torch.nn.functional.one_hot(
            encoded_data['ligand_x'][:5], 
            num_classes=118
        ).float()
        print(f"\nONE-HOT ENCODING EXAMPLE (first 5 atoms):")
        print(f"  Shape: {ligand_one_hot.shape}")
        print(f"  Non-zero indices: {[torch.nonzero(row).squeeze().tolist() for row in ligand_one_hot]}")
    
    print(f"{'='*60}\n")


def process_all_entries(root_dir, mode, utils_dict):
    """Process all PDB entries in a split file"""
    # Load split file
    split_file_path = os.path.join(root_dir, "raw", f"{mode}.txt")
    if not os.path.exists(split_file_path):
        split_file_path = os.path.join(root_dir, "raw", f"{mode}_MD.txt")
    
    if not os.path.exists(split_file_path):
        raise FileNotFoundError(
            f"Split file not found: {split_file_path}\n"
            f"Expected: {os.path.join(root_dir, 'raw', f'{mode}.txt')} or {os.path.join(root_dir, 'raw', f'{mode}_MD.txt')}"
        )
    
    with open(split_file_path, "r") as f:
        pdb_id_list = [line.strip() for line in f.readlines() if line.strip()]
    
    # Load peptides filter
    utils_dir = os.path.join(
        os.path.dirname(__file__), 
        'NeuralMD', 'datasets', 'MISATO', 'utils'
    )
    peptides_file = os.path.join(utils_dir, "peptides.txt")
    peptides_idx_set = set()
    if os.path.exists(peptides_file):
        with open(peptides_file) as f:
            peptides_idx_set = {line.strip().upper() for line in f.readlines()}
    
    # Process each entry
    all_encoded_data = []
    successful = 0
    failed = 0
    
    print(f"Processing {len(pdb_id_list)} entries from {mode} split...")
    
    for i, pdb_id in enumerate(pdb_id_list):
        if pdb_id.upper() in peptides_idx_set:
            print(f"[{i+1}/{len(pdb_id_list)}] Skipping {pdb_id} (peptide)")
            continue
        
        try:
            encoded_data = encode_single_pdb_entry(root_dir, pdb_id, utils_dict)
            all_encoded_data.append(encoded_data)
            successful += 1
            
            if (i + 1) % 10 == 0:
                print(f"[{i+1}/{len(pdb_id_list)}] Processed {pdb_id}: "
                      f"{len(encoded_data['ligand_x'])} ligand atoms, "
                      f"{encoded_data['protein_backbone_residue'].shape[0]} residues")
        except Exception as e:
            failed += 1
            print(f"[{i+1}/{len(pdb_id_list)}] Error processing {pdb_id}: {e}")
            continue
    
    print(f"\nProcessing complete: {successful} successful, {failed} failed")
    return all_encoded_data


def main():
    parser = argparse.ArgumentParser(
        description='Encode MISATO data: ligand atom one-hot encoding and protein backbone encoding'
    )
    parser.add_argument(
        '--root_dir', 
        type=str, 
        required=True,
        help='Path to MISATO_100 directory (should contain raw/MD.hdf5 and raw/{mode}.txt)'
    )
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['train', 'test', 'val'],
        default='train',
        help='Dataset split mode: train, test, or val'
    )
    parser.add_argument(
        '--pdb_id', 
        type=str, 
        default=None,
        help='Single PDB ID to encode (e.g., "1ABC"). If not provided, use --process_all'
    )
    parser.add_argument(
        '--process_all', 
        action='store_true',
        help='Process all entries in the split file'
    )
    parser.add_argument(
        '--save_output', 
        type=str, 
        default=None,
        help='Optional: Save encoded data to .pt file'
    )
    
    args = parser.parse_args()
    
    # Validate root directory
    if not os.path.exists(args.root_dir):
        raise FileNotFoundError(f"Root directory not found: {args.root_dir}")
    
    # Load utility files
    print("Loading utility files...")
    utils_dict = load_utility_files()
    print("Utility files loaded successfully.")
    
    # Process data
    if args.process_all:
        # Process all entries
        all_data = process_all_entries(args.root_dir, args.mode, utils_dict)
        
        if args.save_output:
            output_path = args.save_output
            torch.save(all_data, output_path)
            print(f"\nSaved {len(all_data)} encoded entries to {output_path}")
    
    elif args.pdb_id:
        # Process single entry
        print(f"Encoding PDB ID: {args.pdb_id}")
        encoded_data = encode_single_pdb_entry(args.root_dir, args.pdb_id, utils_dict)
        print_encoding_summary(encoded_data, args.pdb_id)
        
        if args.save_output:
            torch.save(encoded_data, args.save_output)
            print(f"Saved encoded data to {args.save_output}")
    else:
        parser.error("Either --pdb_id or --process_all must be specified")


if __name__ == "__main__":
    main()
