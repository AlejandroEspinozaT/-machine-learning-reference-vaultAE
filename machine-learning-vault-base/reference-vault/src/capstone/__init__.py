from capstone import tumor_classification

DATA_PATH = "reference-vault/src/capstone/data/breast+cancer+wisconsin+diagnostic/wdbc.data"

if __name__ == "__main__":
    capstone = tumor_classification(DATA_PATH)
    capstone.run()
