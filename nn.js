//only 3 layer network
class NeuralNetwork {
	constructor(inputNodes, hiddenNodes, outputNodes) {
		this.inputNodes = inputNodes;
		this.hiddenNodes = hiddenNodes;
		this.outputNodes = outputNodes;
		this.activation = sigmoid;
		this.activationDerivative = dSigmoid;

		//weights between input layer and hidden layer
		this.weights_ih = new Matrix(this.hiddenNodes, this.inputNodes);
		this.weights_ih.randomize();
		//weights between hidden layer and output layer
		this.weights_ho = new Matrix(this.outputNodes, this.hiddenNodes);
		this.weights_ho.randomize();

		//bias of hidden layer
		this.bias_h = Matrix.fromArray(Array.from({ length: this.hiddenNodes }, () => 1));
		//bias of output layer
		this.bias_o = Matrix.fromArray(Array.from({ length: this.outputNodes }, () => 1));

		this.learningRate = 0.1;
	}

	// https://www.youtube.com/watch?v=MPmLWsHzPlU&list=PLRqwX-V7Uu6Y7MdSCaIfsxc561QI0U0Tb&index=13
	feedforward(inputArray) {
		let inputs = Matrix.fromArray(inputArray);

		// GENERATING HIDDEN LAYER OUTPUTS
		// H = sig(W * I + B)
		// H -> hidden layer
		// W -> matrix of weights between inputs layer and hidden layer
		// B -> bias vector
		// sig -> sigmoid (activation function)
		//convert input array into matrix
		//calculate weighted sum of inputs
		let hidden = Matrix.multiply(this.weights_ih, inputs);
		//add bias to prevent limit case 0
		hidden.add(this.bias_h);
		//activation function
		hidden.map(this.activation);

		// GENERATING OUTPUT LAYER OUTPUTS
		// O = sig(W * H + B)
		// o -> output layer
		// W -> matrix of weights between hidden layer and output layer
		// B -> bias vector
		// sig -> sigmoid (activation function)
		let outputs = Matrix.multiply(this.weights_ho, hidden);
		outputs.add(this.bias_o);
		outputs.map(this.activation);

		return outputs.toArray();
	}

	// https://www.youtube.com/watch?v=r2-P1Fi1g60&list=PLRqwX-V7Uu6Y7MdSCaIfsxc561QI0U0Tb&index=15
	//stocastic training -> calculate the error and adjust the weights for every single set of inputs (opposite of batch training)
	train(inputArray, targetsArray) {
		//BACKPROPAGATION of the error using this equations (ΔW = change to apply to the weights, g = activation function (here its derivative g' is used), error = error calculated in the next layer (if i'm in hidden i take error of output for example), lr = learning rate, input = the input of the previous layer, i = previous layer,j = next layer, * = scalar/hadamard product, • = vectorial product)
		// F1: ΔWij = [lr * error_j * g'(x_j)] • transposed(input_i) -> deltas for weights
		//					↑gradient↑
		// F2: ΔBj = [lr * error_j * g'(x_j)] -> deltas for biases

		//rerun the feedforward algorythm
		let inputs = Matrix.fromArray(inputArray);
		let hidden = Matrix.multiply(this.weights_ih, inputs); //this is H
		hidden.add(this.bias_h);
		//save non-activated matrixes to apply derivative next
		let nonActivatedHidden = hidden.clone();
		hidden.map(this.activation);
		let outputs = Matrix.multiply(this.weights_ho, hidden); //this is O
		outputs.add(this.bias_o);
		//save non-activated matrixes to apply derivative next
		let nonActivatedOutputs = outputs.clone();
		outputs.map(this.activation);

		//calculate the error of the output layer -> error = targets - outputs
		let targets = Matrix.fromArray(targetsArray);
		let output_errors = Matrix.subtract(targets, outputs); //this is Eo

		//calculate the error of the hidden layer
		//backpropagation -> the error of the hidden layer is a portion of the total error for each weight -> trough some maths we get this formula (the weights matrix has to be transposed to match rows and cols in multiplications)
		//E[i] = transposed(W[i][i+1])*E[i+1]
		//applying the formula to hidden layer:
		//Eh = transposed(Who) • Eo
		let hidden_errors = Matrix.multiply(Matrix.transpose(this.weights_ho), output_errors); //this is Eh

		//calculate deltas for hidden layer
		//F1 formula applyed to the output layer:
		//ΔWho = lr * 		Eo  		*		(O*(1-O))			•		transposed(H)
		//		      ↑output error↑	 ↑derivative of sigmoid↑		  ↑input of output layer↑
		// calculating ΔWho = lr * Eo * (O*(1-O)) • transposed(H)
		// calculate (O*(1-O))
		let gradients_ho = Matrix.map(nonActivatedOutputs, this.activationDerivative);
		// calculate Eo * (O*(1-O))
		gradients_ho.scalarMultiply(output_errors);
		// calculate lr * Eo * (O*(1-O)) = gradient
		gradients_ho.scalarMultiply(this.learningRate);
		// calculate transposed(H)
		let hidden_t = Matrix.transpose(hidden);
		let weight_ho_deltas = Matrix.multiply(gradients_ho, hidden_t);
		//apply deltas to weights
		this.weights_ho.add(weight_ho_deltas);

		// calculate deltas for input layer
		// ΔWih = lr * Eh * (H*(1-H)) • transposed(I)
		let gradients_ih = Matrix.map(nonActivatedHidden, this.activationDerivative);
		gradients_ih.scalarMultiply(hidden_errors);
		gradients_ih.scalarMultiply(this.learningRate);
		let inputs_t = Matrix.transpose(inputs);
		let weight_ih_deltas = Matrix.multiply(gradients_ih, inputs_t);
		//apply deltas to weights
		this.weights_ih.add(weight_ih_deltas);

		// calculate deltas for the biases of hidden layer using F2 formula
		// ΔBo = lr * Eo * (O*(1-O)) -> I already calculated this! just apply the gradients to the biases then
		this.bias_o.add(gradients_ho);
		//same for imput layer bias
		this.bias_h.add(gradients_ih);
	}

	setLearningRate(lr) {
		this.learningRate = lr;
	}

	setActivationFunction(activation, derivative) {
		this.activation = activation;
		this.activationDerivative = derivative;
	}
}

// sigmoid function used as activation function
const sigmoid = (x) => 1 / (1 + Math.exp(-x));

// first derivative of sigmoid function
const dSigmoid = (x) => sigmoid(x) * (1 - sigmoid(x));
