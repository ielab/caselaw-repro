package main

import (
	"bytes"
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"github.com/dan-locke/phd/generic"
	"io/ioutil"
	"math"
	"os"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"

	trec_output "github.com/dan-locke/language-model-retrieval/trec-output"
)

const measures = 5

type QueryQrels struct {
	lookup map[string]int
	count float64
	ideal []float64
	idcg float64
}

func readQrels(path string) (map[string]QueryQrels, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	rel := map[string]QueryQrels{}
	scanner := bufio.NewScanner(f)
	var parts []string
	var ok bool
	var n int
	var existing QueryQrels
	for scanner.Scan() {
		parts = strings.Fields(scanner.Text())
		if _, ok = rel[parts[0]]; !ok {
			rel[parts[0]] = QueryQrels{
				lookup: map[string]int{},
				count:  0,
				ideal:  []float64{},
			}
		}
		n, _ = strconv.Atoi(parts[3])
		existing = rel[parts[0]]
		existing.lookup[parts[2]] = n
		if n > 0 {
			existing.count++
			existing.ideal = append(existing.ideal, float64(n))
		}
		rel[parts[0]] = existing
	}
	if scanner.Err() != nil {
		return nil, scanner.Err()
	}

	for k, v := range rel {
		sort.Slice(v.ideal, func(i, j int) bool {
			return v.ideal[i] > v.ideal[j]
		})

		for i := range v.ideal {
			v.idcg += (math.Pow(2, v.ideal[i])-1.0) / math.Log2(float64(i+2))
		}

		rel[k] = v
	}

	return rel, nil
}


type runScore struct {
	NDCG float64
	RBP float64
	RecipRank float64
	ExpectedRecipRank float64
	Recall20 float64
	Recall float64
	Unjudged20 float64
	NumRel float64
	NumRelRet float64
}

func scoreRanking(ids []string, rel QueryQrels, depth int, persistence float64) runScore {
	if rel.count == 0.0 {
		return runScore{}
	}

	l := len(ids)
	var i int
	var tmp int

	relVals := make([]int, 0, l)
	relRanks := make([]int, 0, l)
	rrFloat := make([]float64, 0, l)

	var pres bool
	assessed := 0
	maxLabel := 0

	for i = 0; i < l; i++ {
		tmp, pres = rel.lookup[ids[i]]
		if tmp > 0 {
			if tmp > maxLabel {
				maxLabel = tmp
			}
			relVals = append(relVals, tmp)
			relRanks = append(relRanks, i)
			rrFloat = append(rrFloat, float64(i))
		}
		if pres && i < 20 {
			assessed++
		}
	}

	foundRel := len(relVals)
	if foundRel == 0 {
		return runScore{}
	}

	recall20 := 0.0
	recallDepth := 0.0

	for i = 0; i < foundRel; i++ {
		if relRanks[i] < 20 {
			recall20++
		}
		if relRanks[i] < depth {
			recallDepth++
		} else {
			break
		}
	}

	unjudged20 := 20.0 - float64(assessed)
	recall20 /= rel.count
	recallDepth /= rel.count

	// calc RR
	recipRank := 1.0 / (rrFloat[0]+1.0)


	rbp := 0.0
	dcg := 0.0

	// needed for err
	err20 := 0.0
	ml := math.Exp2(float64(maxLabel))
	probRels := make([]float64, maxLabel+1)
	for i = 0; i <= maxLabel; i++ {
		probRels[i] = (math.Pow(2.0, float64(i)) - 1.0) / ml
	}
	p := 1.0

	var r_i, probRel float64
	for i = 0; i < foundRel; i++ {
		if relRanks[i] >= depth {
			break
		}
		r_i = float64(relVals[i])
		rbp += math.Pow(persistence, rrFloat[i])
		dcg += (math.Pow(2.0, r_i)-1.0) / math.Log2(rrFloat[i]+2.0)

		// steps for err
		if relRanks[i] > 20 {
			continue
		}
		probRel = probRels[relVals[i]]
		err20 +=  p  * (probRel/ (rrFloat[i]+1.0))
		p *= (1.0 - probRel)
	}
	rbp *= (1.0 - persistence)
	ndcg := dcg / rel.idcg
	return runScore{
		NDCG:              ndcg,
		RBP:               rbp,
		RecipRank:         recipRank,
		ExpectedRecipRank: err20,
		Recall20:          recall20,
		Recall:            recallDepth,
		Unjudged20:        unjudged20,
		NumRel: rel.count,
		NumRelRet: float64(foundRel),
	}
}

func average(results []runScore) runScore {
	averages := runScore{
		NDCG:              0,
		RBP:               0,
		RecipRank:         0,
		ExpectedRecipRank: 0,
		Recall20:          0,
		Recall:            0,
		Unjudged20:        0,
		NumRel:            0,
		NumRelRet:         0,
	}
	for i := range results {
		averages.NDCG += results[i].NDCG
		averages.RBP += results[i].RBP
		averages.RecipRank += results[i].RecipRank
		averages.ExpectedRecipRank += results[i].ExpectedRecipRank
		averages.Recall20 += results[i].Recall20
		averages.Recall += results[i].Recall
	}
	n := float64(len(results))
	averages.NDCG /= n
	averages.RBP /= n
	averages.RecipRank /= n
	averages.ExpectedRecipRank /= n
	averages.Recall20 /= n
	averages.Recall /= n
	return averages
}

func constructResFeatures(results []map[string][]trec_output.Result, numBounds int, normalise bool) map[string]struct{Ids []string; Scores [][]float64} {
	// qry to doc
	tmp := map[string]map[string][]float64{}
	var pres bool
	var vals []float64
	for i := range results {
		for k, v := range results[i] {
			_, pres = tmp[k]
			if !pres {
				tmp[k] = map[string][]float64{}
			}
			if normalise {
				min, max := math.Inf(1), math.Inf(-1)
				for _, r := range v {
					if r.Score > max {
						max = r.Score
					}
					if r.Score < min {
						min = r.Score
					}
				}
				diff := max-min

				for _, r := range v {
					vals = tmp[k][r.DocId]
					for len(vals) < i {
						vals = append(vals, 0)
					}
					vals = append(vals, (r.Score-min)/diff)
					tmp[k][r.DocId] = vals
				}
			} else {
				for _, r := range v {
					vals = tmp[k][r.DocId]
					for len(vals) < i {
						vals = append(vals, 0)
					}
					vals = append(vals, r.Score)
					tmp[k][r.DocId] = vals
				}

			}
		}
	}

	docs := map[string]struct{Ids []string; Scores [][]float64}{}
	for k, v := range tmp {
		ids := make([]string, 0, len(v))
		scores := make([][]float64, 0, len(v))
		for id, s := range v {
			ids = append(ids, id)
			for len(s) < numBounds {
				s = append(s, 0)
			}
			scores = append(scores, s)
		}
		docs[k] = struct {
			Ids    []string
			Scores [][]float64
		}{Ids: ids, Scores: scores}
	}

	return docs
}

func gridParams(bounds [][]float64) [][]float64 {
	filtered := [][]float64{}
	// construct grid
	n := len(bounds)
	var j int
	unfiltered := generic.AllCombinations(100, n, 0.01)
	var add bool
	for i := range unfiltered {
		add = true
		for j = 0; j < n; j++ {
			if !(unfiltered[i][j] > bounds[j][0] && unfiltered[i][j] < bounds[j][1]) {
				add = false
				break
			}
		}
		if add {
			filtered = append(filtered, unfiltered[i])
		}
	}
	return filtered
}

func writeResults(runName string, maxVals [][]float64, maxWeights [][][]float64, testFolds map[string]int, queryResults map[string]struct{Ids []string; Scores [][]float64}) {
	f, err := os.Create(runName+"-results.txt")
	if err != nil {
		panic(err)
	}
	var line string
	metrics := []string{"RBP", "NDCG", "RR", "ERR", "R@20"}
	line = strings.Join(metrics, " ")
	_, err = f.WriteString(line)
	if err != nil {
		panic(err)
	}
	line = ""
	for i := range maxVals {
		line += strings.Trim(fmt.Sprintf("%.4f", maxVals[i]), "[]")
		line += "\n"
	}
	for i := range maxWeights {
		line += fmt.Sprintf("%.2f", maxWeights[i])
		line += "\n"
	}
	_, err = f.WriteString(line)
	if err != nil {
		panic(err)
	}

	foldResults := make([]*bytes.Buffer, len(metrics))
	for i := range foldResults {
		foldResults[i] = &bytes.Buffer{}
	}
	for i := range maxWeights {
		for query, result := range queryResults {
			if v, ok := testFolds[query]; !ok || v != i {
				continue
			}
			for j := range maxWeights[i] {
				ids := make([]string, len(result.Ids))
				copy(ids, result.Ids)

				scored := combineScores(result.Scores, maxWeights[i][j])
				// sort results
				sort.Sort(sorter{ids, scored})
				results := make([]trec_output.Result, len(ids))
				for j := range ids {
					results[j] = trec_output.Result{query, ids[j], j, scored[j], "c"}
				}
				err = trec_output.ToOutputBuff(results, foldResults[j])
				if err != nil {
					panic(err)
				}
			}
		}
	}

	for i := range metrics {
		err = trec_output.SaveOutputFile(fmt.Sprintf("%s-%s-combine-max.run", runName, metrics[i]), foldResults[i])
		if err != nil {
			panic(err)
		}
	}
}

type sorter struct {
	ids []string
	scores []float64
}

func (s sorter) Len() int { return len(s.ids) }
func (s sorter) Less(i, j int) bool { return s.scores[i] > s.scores[j] }
func (s sorter) Swap(i, j int) {
	s.ids[i], s.ids[j] = s.ids[j], s.ids[i]
	s.scores[i], s.scores[j] = s.scores[j], s.scores[i]
}

func combineScores(scores [][]float64, weights []float64) []float64 {
	l := len(scores)
	n := len(weights)
	var i, j int
	comb := make([]float64, l)
	for i = 0; i < l; i++ {
		for j = 0; j < n; j++ {
			comb[i] += scores[i][j]*weights[j]
		}
	}
	return comb
}

func worker(fold int, wg *sync.WaitGroup, potential [][]float64, folds[][]string, results map[string]struct{Ids []string; Scores [][]float64}, qrels map[string]QueryQrels, maxVals *[][]float64, maxWeights *[][][]float64) {
	l := len(folds[fold])
	runScores := make([]runScore, l)
	var j int
	var ids []string
	var res struct{Ids []string; Scores [][]float64}

	for i := 0; i < len(potential); i++ {
		for j = 0; j < l; j++ {
			res = results[folds[fold][j]]
			ids = make([]string, len(res.Ids))
			copy(ids, res.Ids)
			scored := combineScores(res.Scores, potential[i])
			sort.Sort(sorter{ids, scored})
			runScores[j] = scoreRanking(ids, qrels[folds[fold][j]], 100, 0.8)
		}
		runScores[0] = average(runScores)
		if runScores[0].RBP > (*maxVals)[fold][0] {
			(*maxVals)[fold][0] = runScores[0].RBP
			copy((*maxWeights)[fold][0], potential[i])
		}
		if runScores[0].NDCG > (*maxVals)[fold][1] {
			(*maxVals)[fold][1] = runScores[0].NDCG
			copy((*maxWeights)[fold][1], potential[i])
		}
		if runScores[0].RecipRank > (*maxVals)[fold][2] {
			(*maxVals)[fold][2] = runScores[0].RecipRank
			copy((*maxWeights)[fold][2], potential[i])
		}
		if runScores[0].ExpectedRecipRank > (*maxVals)[fold][3] {
			(*maxVals)[fold][3] = runScores[0].ExpectedRecipRank
			copy((*maxWeights)[fold][3], potential[i])
		}
		if runScores[0].Recall20 > (*maxVals)[fold][4] {
			(*maxVals)[fold][4] = runScores[0].Recall20
			copy((*maxWeights)[fold][4], potential[i])
		}
	}
	wg.Done()
}

type config struct {
	Files []string `json:"files"`
	Bounds [][]float64 `json:"bounds"`
	Folds [][]int `json:"folds"`
	TestFolds [][]int `json:"test_folds"`
	strFolds [][]string
	testFoldLookup map[string]int
	Name string `json:"name"`
}

func readFoldFile(path string) ([][]string, [][]string, error) {
	_, err := ioutil.ReadFile(path)
	if err != nil {
		return nil, nil, err
	}
	
	return nil, nil, nil
}

func readConfigFile(path string) (config, error) {
	var c config
	buff, err := ioutil.ReadFile(path)
	if err != nil {
		return c, err
	}
	err = json.Unmarshal(buff, &c)
	if err != nil {
		return c, err
	}
	if len(c.Bounds) != len(c.Files) {
		return c, fmt.Errorf("Bounds and files do not match.\n")
	}
	for i := range c.Bounds {
		if len(c.Bounds[i]) != 2 {
			return c, fmt.Errorf("Incorrect bound with undetermined lower or upper for bound %d: %.2f.\n", i, c.Bounds[i])
		}
	}
	c.strFolds = make([][]string, len(c.Folds))
	c.testFoldLookup = map[string]int{}
	for i := range c.strFolds {
		c.strFolds[i] = make([]string, len(c.Folds[i]))
		for j := range c.strFolds[i] {
			c.strFolds[i][j] = strconv.Itoa(c.Folds[i][j])
		}
		for j := range c.TestFolds[i] {
			c.testFoldLookup[strconv.Itoa(c.TestFolds[i][j])] = i
		}
	}
	return c, err
}

func main() {

	qrelFile := flag.String("q", "", "Path to qrel file.")
	routines := flag.Int("r", runtime.NumCPU()-1, "Number of routines.")
	configFile := flag.String("c", "", "Path to run config file.")
	normalise := flag.Bool("n", true, "Normalise result files.")
	flag.Parse()

	qrels, err := readQrels(*qrelFile)
	if err != nil {
		panic(err)
	}

	c, err := readConfigFile(*configFile)
	if err != nil {
		panic(err)
	}

	resFiles := make([]map[string][]trec_output.Result, len(c.Files))
	for i := range c.Files {
		res, err := trec_output.ReadResultFile(c.Files[i])
		if err != nil {
			panic(err)
		}
		resFiles[i] = res
	}

	results := constructResFeatures(resFiles, len(c.Bounds), *normalise)

	potential := gridParams(c.Bounds)

	var wg sync.WaitGroup
	// for each fold for each measure, etc ...
	maxVals := make([][]float64, len(c.Folds))
	maxInds := make([][][]float64, len(c.Folds))
	for i := range c.Folds {
		maxVals[i] = make([]float64, measures)
		maxInds[i] = make([][]float64, measures)
		for j := range maxInds[i] {
			maxInds[i][j] = make([]float64, len(c.Bounds))
		}
	}

	if *routines > len(c.Folds) {
		*routines = len(c.Folds)
	}
	for i := 0; i < *routines; i++ {
		wg.Add(1)
		go worker(i, &wg, potential, c.strFolds, results, qrels, &maxVals, &maxInds)
	}
	wg.Wait()

	writeResults(c.Name, maxVals, maxInds, c.testFoldLookup, results)
}



