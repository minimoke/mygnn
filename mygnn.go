package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"
)

type network struct {
	input        int
	hidden       int
	output       int
	ihweights    *mat.Dense
	howeights    *mat.Dense
	learningRate float64
}

var showdebug = false

func main() {

	var right int

	round := func(_, _ int, v float64) float64 { return math.Round(v) }

	// Load Network and Sample Config
	loadcfg()
	log.Println("Loaded Sample Config:", sconfig)

	// Input,Hidden,Output,Learning Rate
	nn := newnetwork(sconfig.INPUTS, sconfig.HIDDEN, sconfig.OUTPUTS, sconfig.LEARNINGRATE)

	fmt.Println("Training Epochs :", sconfig.EPOCHS)
	fmt.Println("0 - - - - - - - - 100")

	randSource := rand.NewSource(time.Now().UnixNano())
	randGen := rand.New(randSource)

	trainingset, testset := samplesplit(80, loadsamples(sconfig.FNAME))

	for e := 0; e < sconfig.EPOCHS; e++ {
		if rem := math.Mod(float64(e), float64(sconfig.EPOCHS)/10); rem == 0 {
			fmt.Printf("* ")

		}
		si := randGen.Intn(len(trainingset))

		nn.train(trainingset[si][:sconfig.INPUTS], trainingset[si][sconfig.INPUTS:])
	}

	for t := range testset {
		if showdebug {
			fmt.Println()
			fmt.Println("x :", testset[t][:sconfig.INPUTS])
			fmt.Println("y :", testset[t][sconfig.INPUTS:])
		}
		pout := nn.predict(testset[t][:sconfig.INPUTS])

		pout.Apply(round, pout)

		if showdebug {
			fmt.Println("o :", mat.Col(nil, 0, pout))
		}

		right += compslices(testset[t][sconfig.INPUTS:], mat.Col(nil, 0, pout))
	}

	fmt.Printf("\nSamples: %v, Right: %v, Accuracy: %v%%\n", len(testset), right, (float32(right) / float32(len(testset)) * 100))

}

func newnetwork(input, hidden, output int, learningRate float64) network {

	setrand := func(_, _ int, v float64) float64 { return rand.NormFloat64() }
	// hidden Layer Weights
	// from layer = Input
	// to layer = Hidden
	// columns = from layer
	// rows = to layer
	ihweights := mat.NewDense(hidden, input, nil)
	ihweights.Apply(setrand, ihweights)
	matPrint(ihweights, "Hidden Layer Weights", showdebug)

	// output layer weights
	howeights := mat.NewDense(output, hidden, nil)
	howeights.Apply(setrand, howeights)
	matPrint(howeights, "Output Layer Weights", showdebug)

	return network{
		input,
		hidden,
		output,
		ihweights,
		howeights,
		learningRate,
	}

}

func (nn *network) train(sample []float64, label []float64) {

	setsig := func(_, _ int, v float64) float64 { return sigmoid(v) }
	setsigprime := func(_, _ int, v float64) float64 { return sigmoidprime(v) }

	// Sample
	x := mat.NewDense(nn.input, 1, sample)
	matPrint(x, "Input Layer - x", showdebug)

	// Labels
	y := mat.NewDense(nn.output, 1, label)
	matPrint(y, "Labels - y", showdebug)

	// Output Layer
	o := mat.NewDense(nn.output, 1, nil)

	// Output of Hidden Layer
	oh := mat.NewDense(nn.hidden, 1, nil)

	// [Hidden Layer Weights] dot [Inputs]
	oh.Mul(nn.ihweights, x)
	matPrint(oh, "Hidden Layer Outputs", showdebug)

	// Apply Sigmoid to Hidden Layer Output
	oh.Apply(setsig, oh)
	matPrint(oh, "Hidden Layer Sigmoid", showdebug)

	// [Hidden Layer Output] dot [Output Layer Weights]
	o.Mul(nn.howeights, oh)
	matPrint(oh, "Output Layer Outouts", showdebug)

	// Apply Sigmoid to Output
	o.Apply(setsig, o)
	matPrint(o, "Output Layer Sigmoid", showdebug)

	// Find Output error
	oerr := mat.NewDense(len(label), 1, nil)
	oerr.Sub(y, o)
	matPrint(oerr, "Output Error", showdebug)

	// Find hidden error
	r, _ := nn.howeights.T().Dims()
	_, c := oerr.Dims()
	herr := mat.NewDense(r, c, nil)
	herr.Mul(nn.howeights.T(), oerr)
	matPrint(herr, "Hidden Layer Error", showdebug)

	// START ADJUST HIDDEN-OUTPUT WEIGHTS

	// add(net.outputWeights, scale(net.learningRate, dot(multiply(outputErrors, sigmoidPrime(finalOutputs)), hiddenOutputs.T())))

	//sigmoidPrime(finalOutputs)
	o.Apply(setsigprime, o)
	matPrint(o, "Output Layer SigmoidPrime", showdebug)

	//multiply(outputErrors, sigmoidPrime(finalOutputs)
	oerr.MulElem(oerr, o)
	matPrint(oerr, "Output Layer Errors x SigmoidPrime", showdebug)

	//dot(multiply(outputErrors, sigmoidPrime(finalOutputs)), hiddenOutputs.T())
	r, _ = oerr.Dims()
	_, c = oh.T().Dims()
	oedhot := mat.NewDense(r, c, nil)
	oedhot.Mul(oerr, oh.T())
	matPrint(oedhot, "Output Layer Errors x SigmoidPrime) . Hidden Layer Output", showdebug)

	//scale(net.learningRate, dot(multiply(outputErrors, sigmoidPrime(finalOutputs)), hiddenOutputs.T())))
	oedhot.Scale(nn.learningRate, oedhot)
	matPrint(oedhot, "oedhot scaled by learningRate", showdebug)

	// Adjust Hidden to Output Weights
	nn.howeights.Add(nn.howeights, oedhot)
	matPrint(nn.howeights, "Adjusted Hidden to Output Weights", showdebug)

	// END ADJUST HIDDEN-OUTPUT WEIGHTS

	// START ADJUST INPUT-HIDDEN WEIGHTS

	// add(net.hiddenWeights,scale(net.learningRate,dot(multiply(hiddenErrors, sigmoidPrime(hiddenOutputs)),inputs.T())))

	//sigmoidPrime(hiddenOutputs)
	oh.Apply(setsigprime, oh)
	matPrint(oh, "Hidden Layer SigmoidPrime", showdebug)

	//multiply(hiddenErrors, sigmoidPrime(hiddenOutputs)
	herr.MulElem(herr, oh)
	matPrint(herr, "Hidden Layer Errors x SigmoidPrime", showdebug)

	//dot(multiply(hiddenErrors, sigmoidPrime(hiddenOutputs)), Inputs.T())
	r, _ = herr.Dims()
	_, c = x.T().Dims()
	hedit := mat.NewDense(r, c, nil)
	hedit.Mul(herr, x.T())
	matPrint(hedit, "Hidden Layer Errors x SigmoidPrime) . Iutput", showdebug)

	//scale(net.learningRate, dot(multiply(hiddenErrors, sigmoidPrime(hiddenOutputs)), Inputs.T())))
	hedit.Scale(nn.learningRate, hedit)
	matPrint(hedit, "hedit scaled by learningRate", showdebug)

	// Adjust Input to Hidden Weights
	nn.ihweights.Add(nn.ihweights, hedit)
	matPrint(nn.ihweights, "Adjusted Input ot Hidden Weights", showdebug)

	// END ADJUST INPUT-HIDDEN WEIGHTS

}

func (nn *network) predict(sample []float64) *mat.Dense {

	setsig := func(_, _ int, v float64) float64 { return sigmoid(v) }

	// Sample
	x := mat.NewDense(nn.input, 1, sample)
	matPrint(x, "Input Layer - x", showdebug)

	// Output Layer
	o := mat.NewDense(nn.output, 1, nil)

	// Output of Hidden Layer
	oh := mat.NewDense(nn.hidden, 1, nil)

	// [Hidden Layer Weights] dot [Inputs]
	oh.Mul(nn.ihweights, x)
	matPrint(oh, "Hidden Layer Outputs", showdebug)

	// Apply Sigmoid to Hidden Layer Output
	oh.Apply(setsig, oh)
	matPrint(oh, "Hidden Layer Sigmoid", showdebug)

	// [Hidden Layer Output] dot [Output Layer Weights]
	o.Mul(nn.howeights, oh)
	matPrint(oh, "Output Layer Outouts", showdebug)

	// Apply Sigmoid to Output
	o.Apply(setsig, o)
	matPrint(o, "Output Layer Sigmoid", showdebug)

	return o
}

func matPrint(X mat.Matrix, message string, display bool) {
	if display {
		fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
		fmt.Println(message)
		fmt.Printf("%v\n", fa)
	}
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func sigmoidprime(x float64) float64 {
	return x * (1.0 - x)
}
