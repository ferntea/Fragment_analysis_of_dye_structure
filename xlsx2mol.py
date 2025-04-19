from rdkit import Chem
import pandas as pd
import os

def excel_to_mol_files(excel_file):
    # Step 1: Read the Excel file into a DataFrame
    df = pd.read_excel(excel_file)

    # Step 2: Normalize column names to lowercase for case-insensitive matching
    df.columns = [col.strip().lower() for col in df.columns]

    # Step 3: Validate required columns ('name' and 'smiles')
    required_columns = {'name', 'smiles'}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Excel file must contain 'Name'/'name' and 'SMILES'/'smiles' columns. Missing: {required_columns - set(df.columns)}")

    # Step 4: Iterate through the rows and create molecules
    for index, row in df.iterrows():
        name = str(row['name']).strip()
        smiles = str(row['smiles']).strip()

        # Parse the SMILES string into an RDKit molecule object
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Warning: Could not parse SMILES '{smiles}' for Name '{name}'. Skipping...")
            continue

        # Add the title to the molecule
        mol.SetProp('_Name', name)

        # Add additional data as properties
        for col in df.columns:
            if col not in ['name', 'smiles']:
                value = row[col]
                if pd.notna(value):  # Skip NaN values
                    mol.SetProp(col.capitalize(), str(value))  # Capitalize property names for consistency

        # Save each molecule to a separate .mol file
        mol_file_name = f"{name}.mol"
        writer = Chem.rdmolfiles.MolToMolFile(mol, mol_file_name)
        print(f"Saved molecule to file: {mol_file_name}")

    print("All molecules have been saved as individual .mol files.")

# Example usage
excel_to_mol_files('data_for_sdf.xlsx')