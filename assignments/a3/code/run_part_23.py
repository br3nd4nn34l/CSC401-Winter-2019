import a3_gmm

if __name__ == '__main__':
    a3_gmm.main(
        M=8,
        k=5,  # number of top speakers to display, <= 0 if none
        epsilon=0,
        max_iter=20,
        data_dir=a3_gmm.dataDir,
        num_speakers=32,
        silent_train=True,
        allow_unseen_test=True
    )