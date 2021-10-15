let brain;
let color = { r: 0, b: 0, g: 0 };
let brainGuess;
let isTrained = false;
let gen = 0;
function setup() {
	createCanvas(800, 800);

	pickColor();

	//3 inputs (r,g,b)
	//3 hidden like the inputs
	//2 output (0:probability of black, 1:probability of white)
	brain = new NeuralNetwork(3, 10, 3);
	brain.setLearningRate(0.05);
}

function draw() {
	// frameRate(10);
	background(0);

	noStroke();
	textAlign(CENTER, CENTER);

	if (!isTrained) {
		train();
	} else {
		process();
	}
}

function mousePressed() {
	if (isTrained) {
		pickColor();
		brainGuess = colorGuesser(color);
		noLoop();
		redraw();
	}
}

function keyPressed() {
	if (!isTrained) {
		isTrained = checkIsTrained(true);
		console.log('isTrained', isTrained);
	}
}

const process = () => {
	fill(color.r, color.g, color.b);
	rect(0, 200, width, height - 400);
	fill(255);
	textSize(20);
	text(`Trained in ${gen} generations`, 400, 10);
	textSize(40);
	text('Press the mouse button to take a guess', 400, 100);

	if (brainGuess) {
		let c = deNormalizeColor(brainGuess);
		textSize(60);
		fill(c.r, c.g, c.b);
		text('Text with guessed color', 400, 300);
		let comp = getComplimentary(color);
		fill(comp.r, comp.g, comp.b);
		text('Text with complimentary color', 400, 500);
		textSize(30);
		text(`I think the right text color is rgb(${c.r},${c.g},${c.b})`, 400, 700);
		// if (brainGuess === 'black') {
		// 	fill(0);
		// 	ellipse(250, 490, 60);
		// } else {
		// 	fill(255);
		// 	ellipse(550, 490, 60);
		// }
	}
};

const train = () => {
	background(color.r * 255, color.g * 255, color.b * 255);
	textSize(60);
	text('training the brain...', 400, 390);
	textSize(20);
	text(gen, 400, 10);

	for (let i = 0; i < 10000; i++) {
		gen++;
		pickColor();
		brain.train(getInputArray(color), getTargetArray(colorPredictor(color)));
	}
	isTrained = checkIsTrained();
};

// no ML
const colorPredictor = (c) => getComplimentary(c);
// const colorPredictor = ({ r, g, b }) => (r + g + b > 300 ? 'black' : 'white');

const colorGuesser = (c) => {
	let outputs = brain.feedforward(getInputArray(c));
	return { r: outputs[0], g: outputs[1], b: outputs[2] };
};

// test data emulator
const checkIsTrained = (log) => {
	let c = getRandomColor();
	let guess = colorGuesser(c);
	let target = colorPredictor(c);
	target = normalizeColor(target);

	Object.keys(guess).forEach((key) => (guess[key] = Number(guess[key].toFixed(2))));
	Object.keys(target).forEach((key) => (target[key] = Number(target[key].toFixed(2))));

	log && console.log(guess, target);
	if (!ObjectSimilar(guess, target, 2, log)) {
		return false;
	}
	return true;
};

//normalize inputs to range 0-1
const getInputArray = ({ r, g, b }) => [r / 255, g / 255, b / 255];

const getTargetArray = ({ r, g, b }) => [
	map(r, 0, 255, 0, 1),
	map(g, 0, 255, 0, 1),
	map(b, 0, 255, 0, 1),
];

const getRandomColor = () => ({ r: random(255), g: random(255), b: random(255) });

const pickColor = () => {
	color = getRandomColor();
};

/* getComplimentary : Converts hex value to HSL, shifts
 * hue by 180 degrees and then converts hex, giving complimentary color
 * as a hex value
 * @param  [String] hex : hex value
 * @return [String] : complimentary color as hex value
 */
function getComplimentary({ r, g, b }) {
	// Convert hex to rgb
	// Credit to Denis http://stackoverflow.com/a/36253499/4939630
	// var rgb = 'rgb(' + (hex = hex.replace('#', '')).match(new RegExp('(.{' + hex.length/3 + '})', 'g')).map(function(l) { return parseInt(hex.length%2 ? l+l : l, 16); }).join(',') + ')';

	// // Get array of RGB values
	// rgb = rgb.replace(/[^\d,]/g, '').split(',');

	// var r = rgb[0], g = rgb[1], b = rgb[2];

	// Convert RGB to HSL
	// Adapted from answer by 0x000f http://stackoverflow.com/a/34946092/4939630
	r /= 255.0;
	g /= 255.0;
	b /= 255.0;
	var max = Math.max(r, g, b);
	var min = Math.min(r, g, b);
	var h,
		s,
		l = (max + min) / 2.0;

	if (max == min) {
		h = s = 0; //achromatic
	} else {
		var d = max - min;
		s = l > 0.5 ? d / (2.0 - max - min) : d / (max + min);

		if (max == r && g >= b) {
			h = (1.0472 * (g - b)) / d;
		} else if (max == r && g < b) {
			h = (1.0472 * (g - b)) / d + 6.2832;
		} else if (max == g) {
			h = (1.0472 * (b - r)) / d + 2.0944;
		} else if (max == b) {
			h = (1.0472 * (r - g)) / d + 4.1888;
		}
	}

	h = (h / 6.2832) * 360.0 + 0;

	// Shift hue to opposite side of wheel and convert to [0-1] value
	h += 180;
	if (h > 360) {
		h -= 360;
	}
	h /= 360;

	// Convert h s and l values into r g and b values
	// Adapted from answer by Mohsen http://stackoverflow.com/a/9493060/4939630
	if (s === 0) {
		r = g = b = l; // achromatic
	} else {
		var hue2rgb = function hue2rgb(p, q, t) {
			if (t < 0) t += 1;
			if (t > 1) t -= 1;
			if (t < 1 / 6) return p + (q - p) * 6 * t;
			if (t < 1 / 2) return q;
			if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
			return p;
		};

		var q = l < 0.5 ? l * (1 + s) : l + s - l * s;
		var p = 2 * l - q;

		r = hue2rgb(p, q, h + 1 / 3);
		g = hue2rgb(p, q, h);
		b = hue2rgb(p, q, h - 1 / 3);
	}

	r = Math.round(r * 255);
	g = Math.round(g * 255);
	b = Math.round(b * 255);

	return { r, g, b };
	// // Convert r b and g values to hex
	// rgb = b | (g << 8) | (r << 16);
	// return '#' + (0x1000000 | rgb).toString(16).substring(1);
}

const arrayEquals = (array1, array2) =>
	array1.length === array2.length &&
	array1.every(function (value, index) {
		return value === array2[index];
	});
const ObjectEquals = (obj1, obj2) => arrayEquals(Object.values(obj1), Object.values(obj2));

const normalizeColor = ({ r, g, b }) => ({
	r: map(r, 0, 255, 0, 1),
	g: map(g, 0, 255, 0, 1),
	b: map(b, 0, 255, 0, 1),
});

const normalizeArray = ([r, g, b]) => [
	map(r, 0, 255, 0, 1),
	map(g, 0, 255, 0, 1),
	map(b, 0, 255, 0, 1),
];

const deNormalizeColor = ({ r, g, b }) => ({
	r: parseInt(Math.round(map(r, 0, 1, 0, 255))),
	g: parseInt(Math.round(map(g, 0, 1, 0, 255))),
	b: parseInt(Math.round(map(b, 0, 1, 0, 255))),
});

const ObjectSimilar = (obj1, obj2, error, log) => {
	error = error ?? 1; //%
	//max:v = 100 : x -> x = 100*v/max

	let arr1 = Object.values(obj1);
	let arr2 = Object.values(obj2);
	let max = Math.max(...[...arr1, ...arr2]);

	arr1 = arr1.map((v) => (100 * v) / max);
	arr2 = arr2.map((v) => (100 * v) / max);

	let diff = arr1.map((v, i) => Math.abs(v - arr2[i]));
	let media = diff.reduce((acc, v) => acc + v, 0) / diff.length;
	if (log) {
		console.log(diff, media);
	}
	if (media <= error) {
		return true;
	}
	return false;
};
