from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit.Chem.rdmolfiles import SDWriter
import pandas as pd


def excel_to_sdf(excel_file, sdf_file):
    # Step 1: Read the Excel file into a DataFrame
    df = pd.read_excel(excel_file)

    # Step 2: Normalize column names to lowercase for case-insensitive matching
    df.columns = [col.strip().lower() for col in df.columns]

    # Step 3: Validate required columns ('name' and 'smiles')
    required_columns = {'name', 'smiles'}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Excel file must contain 'Name'/'name' and 'SMILES'/'smiles' columns. Missing: {required_columns - set(df.columns)}")

    # Step 4: Initialize the SDF writer
    writer = SDWriter(sdf_file)

    # Step 5: Iterate through the rows and create molecules
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

        # Write the molecule to the SDF file
        writer.write(mol)

    # Step 6: Close the SDF writer
    writer.close()
    print(f"SDF file saved successfully: {sdf_file}")


# Example usage
excel_to_sdf('data_for_sdf.xlsx', 'data_sdf_from_xlsx.sdf')