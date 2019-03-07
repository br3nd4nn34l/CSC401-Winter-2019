from lm_train import lm_train
from perplexity import preplexity

hans_train_path = "../data/Hansard/Training"

lang_models = {
    lang : lm_train(
        hans_train_path, lang,
        f"../outputs/{lang}_Hansard_Model"
    )
    for lang in "ef"
}

discussion = """
The lowest perplexities (~13) were achieved using the MLE estimate 
(no delta smoothing). In other words, the predictive power of 
the language model degrades when even minimal weighting is given towards 
unseen bigrams. This suggests that a vast majority of the test corpus 
can be predicted using only the grams from the training corpus.

English had an MLE perplexity of 13.75 whereas 
French had an MLE perplexity of 13.06. 
This rough equivalence was maintained regardless of delta,
suggesting that the language model is equally well suited 
for training on either language.

Overall, perplexity increased as the smoothing factor increased (see results below).
There was a small decrease between delta=0.0001 and 0.001 (43,39)->(39,36),
and an increase between delta=0.001 and 0.01 (39,36)->(41,41).
This suggests that a better smoothing factor exists 
(probably with perplexity around 36-40) somewhere in the 
interval of 0.0001 and 0.01, but this is unlikely to be better 
than the MLE perplexity (~13).
"""

with open("Task3.txt", "w") as task3:

    task3.write("======== DISCUSSION ========\n")
    task3.write(discussion)


    task3.write("======== RESULTS    ========\n")

    task3.write("Note: Delta=0 means no smoothing (MLE estimate)\n")
    task3.write("Language, Delta, Perplexity\n")

    for delta in [0, 0.0001, 0.001, 0.01, 0.1, 1]:

        if not delta:
            smoothing = False
        else:
            smoothing = True


        for (lang, model) in lang_models.items():
            perp = preplexity(model,
                              test_dir="../data/Hansard/Testing/",
                              language=lang,
                              smoothing=smoothing,
                              delta=delta)
            task3.write(f"{lang.upper()}, {delta}, {perp:1.4f}\n")