import a3_gmm

def run_experiment(seed, M=8, max_iter=20, num_speakers=32, allow_unseen_test=True):
    a3_gmm.main(
        M=M,
        k=5,
        epsilon=0,
        max_iter=max_iter,
        data_dir=a3_gmm.dataDir,
        num_speakers=num_speakers,
        silent_train=True,
        allow_unseen_test=allow_unseen_test,
        seed=seed
    )


if __name__ == '__main__':

    run_num_comps = False
    run_num_iters = False
    run_num_speak = True
    run_num_speak_no_unseen = False

    # Test effect of number of components
    if run_num_comps:
        print("Effect of Number of GMM Components (M) on Performance")
        for M in [8, 6, 4, 2, 1]:
            run_experiment(M=M,
                           max_iter=20,
                           seed=1)
        print("\n")

    # Test effect of number of training iterations
    if run_num_iters:
        print("Effect of Training Iterations (maxIter) on Performance")
        for max_iter in [0, 5, 10, 15, 20]:
            run_experiment(max_iter=max_iter,
                           M=8,
                           allow_unseen_test=False,
                           seed=0)
        print("\n")

    # Test effect of number of speakers
    if run_num_speak:
        print("Effect of Number of Speakers (S) on Performance (Unseen Data Allowed)")
        for num_speakers in [4, 8, 16, 24, 32]:
            run_experiment(401, num_speakers=num_speakers, max_iter=5)
        print("\n")

    if run_num_speak_no_unseen:
        print("Effect of Number of Speakers (S) on Performance (Unseen Data Disallowed)")
        for num_speakers in [32, 24, 16, 8, 4]:
            run_experiment(num_speakers=num_speakers,
                           allow_unseen_test=False,
                           M=8,
                           max_iter=3)
        print("\n")