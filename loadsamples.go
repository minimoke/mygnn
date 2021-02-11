package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"math"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"time"
)

type samplecfg struct {
	DATASETNAME  string
	FNAME        string
	CSVHEADER    bool
	IDCOL        bool
	LABELCOL     int
	INPUTS       int
	HIDDEN       int
	OUTPUTS      int
	LEARNINGRATE float64
	EPOCHS       int
	LABEL        map[string][]float64
}

var sconfig samplecfg

func loadcfg() {

	if len(os.Args) < 2 {
		log.Println("Usage: mygnn [JSON config file]")
		log.Fatal("No config file on command line")
	}

	content, err := ioutil.ReadFile(os.Args[1])
	if err != nil {
		log.Fatal(err)
	}

	err = json.Unmarshal(content, &sconfig)
	if err != nil {
		fmt.Println("error:", err)
	}
}

func loadsamples(fname string) [][]float64 {

	var samples [][]float64
	var samplesN [][]float64
	var mins, maxs []float64

	csvFile, _ := os.Open(fname)
	reader := csv.NewReader(csvFile)

	if sconfig.CSVHEADER {
		_, err := reader.Read()
		if err == io.EOF {
			os.Exit(1)
		}
	}

	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}

		if err != nil {
			log.Fatal(err)
		}

		var templine []float64

		for i := range record {
			if sconfig.IDCOL && i == 0 {
				continue
			}

			if i == sconfig.LABELCOL {
				templine = append(templine, sconfig.LABEL[record[sconfig.LABELCOL]]...)
				continue
			}

			if s, err := strconv.ParseFloat(record[i], 64); err != nil {
				log.Printf("Error Converting %v to float64", record[i])
			} else {
				templine = append(templine, s)
			}
		}

		samples = append(samples, templine)
	}

	// Get min/max for each Column
	for i := range samples[0] {
		colmin, colmax := colstats(coltoslice(i, samples))
		mins = append(mins, colmin)
		maxs = append(maxs, colmax)
	}

	// Normalise Data
	for _, r := range samples {
		var cols []float64
		for ci, cv := range r {
			if ci < sconfig.LABELCOL {
				cols = append(cols, normalise(cv, mins[ci], maxs[ci]))
			} else {
				cols = append(cols, cv)
			}
		}
		samplesN = append(samplesN, cols)
	}

	return samplesN
}

func colstats(vals []float64) (min, max float64) {

	sort.Float64s(vals)

	minval := vals[0]
	maxval := vals[len(vals)-1]

	return minval, maxval
}

func coltoslice(column int, samples [][]float64) []float64 {

	var colslice []float64

	for _, r := range samples {
		colslice = append(colslice, r[column])

	}

	return colslice
}

func compslices(output, label []float64) int {

	if len(output) != len(label) {
		log.Println("Slices not the sames length")
		return 0
	}

	for i, v := range output {
		if v != label[i] {
			return 0
		}
	}
	return 1
}

func samplesplit(split float64, samples [][]float64) (trainingset, testset [][]float64) {

	var train [][]float64
	var test [][]float64

	trainsamples := math.RoundToEven(float64(len(samples)) * (split / 100))

	randSource := rand.NewSource(time.Now().UnixNano())
	randGen := rand.New(randSource)
	ilist := randGen.Perm(len(samples))

	for i := 0; i < int(trainsamples); i++ {
		train = append(train, samples[ilist[i]])
	}

	for i := int(trainsamples); i < len(samples); i++ {
		test = append(test, samples[ilist[i]])
	}

	return train, test

}

func normalise(x, min, max float64) float64 {
	return (x - min) / (max - min)
}
