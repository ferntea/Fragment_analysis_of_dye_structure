import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import matplotlib.pyplot as plt
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
import numpy as np
import io


def excel_to_table_plot(excel_file, output_image, columns=5, fig_size=(7, 5), dpi=300):
    """
    Reads chemical names and SMILES from an Excel file, generates molecular structures,
    and arranges them in a table-like plot.

    Parameters:
        excel_file (str): Path to the input Excel file.
        output_image (str): Path to save the output image.
        columns (int): Number of columns in the table.
        fig_size (tuple): Size of the figure in inches (width, height).
        dpi (int): Resolution of the output image.
    """
    # Step 1: Read the Excel file into a DataFrame
    df = pd.read_excel(excel_file)

    # Step 2: Normalize column names to lowercase for case-insensitive matching
    df.columns = [col.strip().lower() for col in df.columns]

    # Step 3: Validate required columns ('name' and 'smiles')
    required_columns = {'name', 'smiles'}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Excel file must contain 'Name'/'name' and 'SMILES'/'smiles' columns. Missing: {required_columns - set(df.columns)}")

    # Step 4: Prepare data
    names = df['name'].tolist()
    smiles = df['smiles'].tolist()

    # Convert SMILES to RDKit molecules
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    valid_mols = [(mol, name) for mol, name in zip(mols, names) if mol is not None]

    if not valid_mols:
        raise ValueError("No valid molecules found in the input data.")

    # Step 5: Calculate rows and columns for the table
    num_compounds = len(valid_mols)
    rows = int(np.ceil(num_compounds / columns))

    # Step 6: Create a figure and axes
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    ax.axis('off')  # Turn off the axis

    # Step 7: Generate molecular images
    cell_width = fig_size[0] / columns
    cell_height = fig_size[1] / rows
    padding = 0.05  # Padding between cells
    name_padding = 0.02  # Padding for names

    for idx, (mol, name) in enumerate(valid_mols):
        row_idx = idx // columns
        col_idx = idx % columns

        # Calculate position for the current cell
        x_start = col_idx * cell_width
        y_start = fig_size[1] - (row_idx + 1) * cell_height

        # Get the bounding box of the molecule
        AllChem.Compute2DCoords(mol)
        conf = mol.GetConformer()
        x_coords = [conf.GetAtomPosition(i).x for i in range(conf.GetNumAtoms())]
        y_coords = [conf.GetAtomPosition(i).y for i in range(conf.GetNumAtoms())]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        mol_width = max_x - min_x
        mol_height = max_y - min_y

        # Calculate the aspect ratio of the molecule
        aspect_ratio = mol_width / mol_height

        # Determine the drawing dimensions while maintaining the aspect ratio
        max_draw_width = int(cell_width * dpi * (1 - 2 * padding))
        max_draw_height = int(cell_height * dpi * (1 - 2 * padding) - name_padding * dpi)

        if aspect_ratio > 1:  # Wider than tall
            draw_width = max_draw_width
            draw_height = int(draw_width / aspect_ratio)
        else:  # Taller than wide
            draw_height = max_draw_height
            draw_width = int(draw_height * aspect_ratio)

        # Ensure the drawing dimensions do not exceed the maximum allowed dimensions
        draw_width = min(draw_width, max_draw_width)
        draw_height = min(draw_height, max_draw_height)

        # Generate the molecule image with adjusted size
        drawer = rdMolDraw2D.MolDraw2DCairo(draw_width, draw_height)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        img = Image.open(io.BytesIO(drawer.GetDrawingText()))

        # Convert the image to a format suitable for matplotlib
        img_np = np.array(img)

        # Calculate the position to center the molecule image in the cell
        img_x_start = x_start + (cell_width - (draw_width / dpi)) / 2
        img_y_start = y_start + (cell_height - (draw_height / dpi) - name_padding) / 2

        # Add the molecule image to the plot
        ax.imshow(img_np, extent=[img_x_start, img_x_start + draw_width / dpi,
                                  img_y_start, img_y_start + draw_height / dpi])

        # Add the compound name below the molecule
        name_y_pos = y_start + name_padding / 2
        ax.text(x_start + cell_width / 2, name_y_pos, name,
                fontsize=8, ha='center', va='bottom', wrap=True)

    # Step 8: Adjust the limits of the plot to ensure all content is visible
    ax.set_xlim(0, fig_size[0])
    ax.set_ylim(0, fig_size[1])

    # Step 9: Save the output image
    plt.savefig(output_image, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Output image saved successfully: {output_image}")


# Example usage
excel_to_table_plot('data_for_sdf.xlsx', 'excel2table_plot.png',
                    fig_size=(7, 7.5), dpi = 600, columns=4)