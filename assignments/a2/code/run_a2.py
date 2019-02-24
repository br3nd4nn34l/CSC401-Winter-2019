from lm_train import lm_train

toy_data_path = "../data/Toy"

for lang in ["e", "f"]:
    lm_train(toy_data_path, lang, f"../outputs/{lang}_Toy_Model")