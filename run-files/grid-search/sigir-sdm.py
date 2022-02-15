import copy
import json
import subprocess

if __name__ == "__main__":
    config = {"files": [
                            "/home/danlocke/phd-generated/dirichlet_prior/sigir-topic-topics-sigir-stop-unigram_dir_mu_1500.00.run",
                            "/home/danlocke/phd-generated/sdm_rerank/sigir-topic-topics-sigir-stop-sdm_rerank{0}-dir-mu-1500.00-weights-0.00-1.00-0.00-window-{1}.run",
                            "/home/danlocke/phd-generated/sdm_rerank/sigir-topic-topics-sigir-stop-sdm_rerank{0}-dir-mu-1500.00-weights-0.00-0.00-1.00-window-{1}.run",
            ],
            "bounds": [
                    [0, 1], [0, 1], [0, 1]
            ],
            "name": "", 
            "folds": [[1, 37, 39, 9, 45, 19, 51, 23, 25, 31],
                 [97, 3, 37, 39, 45, 19, 51, 23, 25, 31],
                 [1, 97, 3, 37, 9, 45, 19, 25, 31],
                 [1, 97, 3, 39, 9, 45, 19, 51, 23, 31],
                 [1, 97, 3, 37, 39, 9, 51, 23, 25]], 
            "test_folds": [[3, 97], [1, 9], [51, 23, 39], [25, 37], [45, 19, 31]] 
    }

    for i in range(1, 21):
        print('Doing: ', i)
        for j in ["", "-smooth"]:
            with open('run.json', 'w+') as f:
                data = copy.deepcopy(config)
                data["files"][1] = data["files"][1].format(j, i)
                data["files"][2] = data["files"][2].format(j, i)
                data["name"] = "sigir-sdm-{0}{1}-window".format(i, j)
                f.write(json.dumps(data))
        
            subprocess.run(["./gridsearch", "-c", "run.json", "-q", "/home/danlocke/go/src/github.com/dan-locke/phd-data/sigir/comb-sigir.txt", "-n=False"])
