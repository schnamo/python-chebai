import sys

from chem import experiments


def main():
    batch_size = int(sys.argv[2])
    experiment_id = sys.argv[1]
    ex = experiments.EXPERIMENTS[experiment_id](batch_size, *sys.argv[3:])
    ex.execute()


if __name__ == "__main__":
    main()
