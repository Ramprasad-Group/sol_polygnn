import pandas as pd
import torch
import sol_polygnn as sp
import sol_trainer as st

pe_smiles = "[*]CC[*]"  # the SMILES string for polyethylene

model_name = "solvent_class"
root_dir = f"./trained_models/{model_name}"

# Load class dictionary
class_labels = st.load.load_classlabels(root_dir, reverse=True)

# For convenience, let's define a function that makes predictions.
def make_prediction(data):
    """
    Return the mean and std. dev. of a model prediction.

    Args:
        data (pd.DataFrame): The input data for the prediction.
    """
    bond_config = sp.featurize.BondConfig(True, True, True)
    atom_config = sp.featurize.AtomConfig(
        True,
        True,
        True,
        True,
        True,
        True,
        combo_hybrid=False,  # if True, SP2/SP3 are combined into one feature
        aromatic=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # specify GPU

    # Load scalers
    scaler_dict = st.load2.load_scalers(root_dir)

    # Load selectors
    selectors = st.load2.load_selectors(root_dir)

    # Define a lambda function for smiles featurization.
    smiles_featurizer = lambda x: sp.featurize.get_minimum_graph_tensor(
        x,
        bond_config,
        atom_config,
        "monocycle",
    )

    # Load and evaluate ensemble.
    ensemble = st.load.load_ensemble(
        st.models.LinearEnsemble,
        root_dir,
        sp.models.polyGNN,
        device,
        submodel_kwargs_dict={
            "node_size": atom_config.n_features,
            "edge_size": bond_config.n_features,
            "selector_dim": 0,
            "graph_feats_dim": len(graph_feats),
            "node_feats_dim": 0,
            "output_dim": len(class_labels),
        },
        ensemble_init_kwargs={"monte_carlo": False},
    )

    y, y_mean_hat, y_std_hat, _selectors = st.infer.eval_ensemble(
        model=ensemble,
        root_dir=root_dir,
        dataframe=data,
        smiles_featurizer=smiles_featurizer,
        device=device,
    )
    return y_mean_hat, y_std_hat


# Let's make predictions using the "solvent_class" model.
properties = ["solvent_class"]
graph_feats = {
    "12-dichloroethane": 0,
    "acetic acid": 0,
    "acetone": 0,
    "acetonitrile": 0,
    "benzene": 0,
    "butanol": 0,
    "carbon disulfide": 0,
    "carbon tetrachloride": 0,
    "chlorobenzene": 0,
    "chloroform": 0,
    "cresol": 0,
    "cyclohexane": 0,
    "cyclohexanone": 0,
    "decalin": 0,
    "dichloroacetic acid": 0,
    "dichlorobenzene": 0,
    "dichloroethane": 0,
    "dichloromethane": 0,
    "diethyl ether": 0,
    "dioxane": 0,
    "dma": 0,
    "dmac": 0,
    "dmf": 0,
    "dmso": 0,
    "ethanol": 0,
    "ether": 0,
    "ethyl acetate": 0,
    "ethyl ether": 0,
    "ethylene chloride": 0,
    "formamide": 0,
    "formic acid": 0,
    "gamma-butyrolactone": 0,
    "glycerol": 0,
    "heptane": 0,
    "hexafluorobenzene": 0,
    "hexamethylphosphoramide": 0,
    "hexane": 0,
    "hydrochloric acid": 0,
    "methanesulfonic acid": 0,
    "methanol": 0,
    "methyl ethyl ketone": 0,
    "methylene chloride": 0,
    "nitric acid": 0,
    "nitrobenzene": 0,
    "nmp": 0,
    "o-chlorophenol": 0,
    "o-dichlorobenzene": 0,
    "p-chlorophenol": 0,
    "phenol": 0,
    "propanol": 0,
    "pyridine": 0,
    "sulfuric acid": 0,
    "tetrachloroethane": 0,
    "tetralin": 0,
    "thf": 0,
    "toluene": 0,
    "trichlorobenzene": 1,
    "trifluoroacetic acid": 0,
    "trifluoroethanol": 0,
    "water": 0,
    "xylene": 0,
}

print(f"Using model {model_name}.")
data = pd.DataFrame(
    {
        "smiles_string": [pe_smiles] * len(properties),
        "prop": properties,
        "graph_feats": [graph_feats],
    }
)
means, _ = make_prediction(data)
print()
print(
    f"trichlorobenzene is predicted to be a {class_labels[means[0]]} for polyethylene."
)
