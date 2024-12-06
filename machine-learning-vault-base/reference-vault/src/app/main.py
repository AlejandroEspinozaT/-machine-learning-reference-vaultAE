import sys
from pathlib import Path
from capstone.mpg_regression import MpgRegressionCapstone
from capstone.tumor_classification import TumorClassificationCapstone
sys.path.append(str(Path(__file__).resolve().parent.parent))

DATA_PATH = "../capstone/data/breast+cancer+wisconsin+diagnostic/wdbc.data"

def main() -> None:
    mpg_capstone = MpgRegressionCapstone()
    mpg_capstone.run()

    tumor_capstone = TumorClassificationCapstone(DATA_PATH)
    tumor_capstone.run()


if __name__ == "__main__":
    main()
