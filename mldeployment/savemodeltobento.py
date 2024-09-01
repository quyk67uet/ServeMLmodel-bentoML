"""This module saves a Scikit-learn model to BentoML."""

from pathlib import Path
import joblib
import bentoml


def load_model_and_save_it_to_bento(model_file: Path) -> None:
    """Loads a scikit-learn model from disk and saves it to BentoML."""
    # Load the scikit-learn model from disk
    model = joblib.load(model_file)
    
    # Save the scikit-learn model to BentoML
    bento_model = bentoml.sklearn.save_model("sklearn_model", model)
    
    print(f"Bento model tag = {bento_model.tag}")


if __name__ == "__main__":
    load_model_and_save_it_to_bento(Path("mlp_model.joblib"))
