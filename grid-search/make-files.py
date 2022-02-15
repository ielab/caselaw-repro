import copy
import json
import subprocess

if __name__ == "__main__":
    config = {"files": [
                            "/home/danlocke/phd-generated/preprocessing/dirichlet_prior/case-topics-filtered-phrasestop-unigram_dir_mu_1050.00.run",
                            "/home/danlocke/phd-generated/sdm_rerank/case-topics-filtered-phrasestop-sdm_rerank{0}-dir-mu-1050.00-weights-0.00-1.00-0.00-window-{1}.run",
                            "/home/danlocke/phd-generated/sdm_rerank/case-topics-filtered-phrasestop-sdm_rerank{0}-dir-mu-1050.00-weights-0.00-0.00-1.00-window-{1}.run",
            ],
            "bounds": [
                    [0, 1], [0, 1], [0, 1]
            ],
            "name": "", 
            "folds": [
                    [1, 2, 3, 4, 5, 7, 10, 11, 12, 13, 15, 17, 21, 22, 23, 24, 25, 27, 28, 29, 32, 34, 35, 37, 39, 41, 43, 49, 50, 54, 55, 57, 58, 59, 60, 61, 62, 64, 65, 67, 69, 73, 75, 76, 77, 78, 80, 81, 82, 83, 84, 85, 88, 90, 91, 93, 94, 95, 96, 97, 98, 99, 101, 102, 103, 105, 108, 109, 110, 111, 112, 113, 114, 115, 117, 118],
                    [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 15, 17, 19, 22, 23, 24, 25, 27, 28, 32, 33, 35, 44, 45, 46, 47, 49, 50, 53, 54, 55, 58, 59, 60, 61, 64, 65, 67, 69, 70, 71, 73, 74, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 93, 94, 95, 96, 97, 98, 103, 104, 105, 107, 108, 109, 110, 111, 113, 114, 115, 116, 117, 118],
                    [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 19, 21, 22, 24, 25, 27, 29, 33, 34, 35, 37, 39, 41, 43, 44, 45, 46, 47, 49, 53, 54, 55, 57, 59, 61, 62, 64, 65, 67, 70, 71, 74, 75, 76, 77, 78, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 94, 95, 96, 97, 98, 99, 101, 102, 104, 105, 107, 108, 109, 111, 112, 114, 116, 117, 118],
                    [1, 2, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 21, 23, 24, 27, 28, 29, 32, 33, 34, 35, 37, 39, 41, 43, 44, 45, 46, 47, 49, 50, 53, 54, 55, 57, 58, 60, 62, 64, 67, 69, 70, 71, 73, 74, 75, 76, 77, 78, 80, 81, 84, 86, 87, 88, 89, 90, 91, 93, 96, 99, 101, 102, 103, 104, 107, 108, 109, 110, 112, 113, 115, 116, 117],
                    [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 15, 17, 19, 21, 22, 23, 25, 28, 29, 32, 33, 34, 37, 39, 41, 43, 44, 45, 46, 47, 50, 53, 57, 58, 59, 60, 61, 62, 65, 69, 70, 71, 73, 74, 75, 76, 77, 78, 80, 82, 83, 85, 86, 87, 89, 93, 94, 95, 97, 98, 99, 101, 102, 103, 104, 105, 107, 110, 111, 112, 113, 114, 115, 116, 118]
            ],
            "test_folds": [
                [107, 6, 104, 46, 74, 19, 33, 116, 86, 8, 89, 70, 9, 44, 71, 53, 47, 87, 45],
                [34, 39, 101, 112, 99, 62, 43, 57, 102, 41, 29, 37, 11, 21, 78, 75, 76, 77, 7],
                [103, 15, 32, 60, 23, 13, 113, 93, 2, 80, 50, 17, 69, 1, 28, 73, 110, 58, 115],
                [105, 118, 12, 22, 65, 83, 97, 3, 59, 25, 94, 111, 98, 95, 82, 114, 61, 85, 4],
                [117, 96, 90, 88, 91, 109, 67, 24, 10, 81, 54, 49, 108, 84, 27, 64, 35, 55, 5]
            ]
    }

    for i in range(1, 21): 
        for j in ["", "-smooth"]:
            with open('run.json', 'w+') as f:
                data = copy.deepcopy(config)
                data["files"][1] = data["files"][1].format(j, i)
                data["files"][2] = data["files"][2].format(j, i)
                data["name"] = "sdm-{0}{1}-window".format(i, j)
                f.write(json.dumps(data))
        
            subprocess.run(["./gridsearch", "-c", "run.json", "-q", "/home/danlocke/go/src/github.com/dan-locke/phd-data/aus/filtered-qrels.txt", "-n=False"])
