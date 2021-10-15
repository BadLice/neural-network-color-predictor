class Matrix {
	constructor(rows, cols) {
		this.data = Array.from({ length: rows }, () => Array.from({ length: cols }, () => 0));
		this.rows = rows;
		this.cols = cols;
	}

	static fromArray(arr) {
		let m = new Matrix(arr.length, 1);
		m.map((_, i) => arr[i]);
		return m;
	}

	toArray() {
		let arr = [];
		this.data.forEach((row) => row.forEach((value) => arr.push(value)));
		return arr;
	}

	static multiply(m1, m2) {
		// matrix multiplication
		//this is A, n is B, im doing A x B
		if (m1.cols !== m2.rows) {
			throw new Error(
				"Error in Matrix.multiply function: columns of A and rows of B don't match"
			);
		}

		let result = new Matrix(m1.rows, m2.cols);
		result.data.forEach((r, i) =>
			r.forEach((_, j) => {
				let row = m1.getRow(i);
				let col = m2.getCol(j);
				result.data[i][j] = row.reduce(
					(acc, value, index) => acc + value * col[index],
					0
				);
			})
		);
		return result;
	}

	//scalar multiplication
	scalarMultiply = (n) => {
		if (n instanceof Matrix) {
			this.data = this.data.map((row, i) => row.map((value, j) => value * n.data[i][j]));
		} else {
			this.data = this.data.map((row) => row.map((value) => value * n));
		}
		this.updateSize();
	};

	add = (n) => {
		if (n instanceof Matrix) {
			//matrix addition
			if (n.cols !== this.cols || n.rows !== this.rows) {
				throw new Error("Error in Matrix.add function: sizes don't match");
			}
			this.data = this.data.map((row, i) => row.map((value, j) => value + n.data[i][j]));
		} else {
			//scalar addition
			this.data = this.data.map((row) => row.map((value) => value + n));
		}
		this.updateSize();
	};

	static subtract(a, b) {
		let result = new Matrix(a.rows, a.cols);
		a.data.forEach((row, i) =>
			row.forEach((value, j) => (result.data[i][j] = value - b.data[i][j]))
		);
		return result;
	}

	static transpose = (m) => {
		let result = new Matrix(m.cols, m.rows);
		m.data.forEach((row, i) =>
			row.forEach((value, j) => {
				result.data[j][i] = value;
			})
		);
		return result;
	};

	transpose = () => {
		let result = new Matrix(this.cols, this.rows);
		this.data.forEach((row, i) =>
			row.forEach((value, j) => {
				result.data[j][i] = value;
			})
		);
		this.data = result.data;
		this.updateSize();
	};

	map(callback) {
		this.data = this.data.map((row, i) => row.map((value, j) => callback(value, i, j)));
	}

	static map(m, callback) {
		let result = new Matrix(m.rows, m.cols);
		result.data = result.data.map((row, i) =>
			row.map((_, j) => callback(m.data[i][j], i, j))
		);
		return result;
	}

	clone() {
		let result = new Matrix(this.rows, this.cols);
		result.data.forEach((row, i) =>
			row.forEach((_, j) => (result.data[i][j] = this.data[i][j]))
		);
		// console.log(result.data);
		return result;
	}

	randomize = () => {
		this.data = this.data.map((row) => row.map(() => Math.random() * 2 - 1));
		this.updateSize();
	};

	updateSize = () => {
		this.rows = this.data.length;
		this.cols = this.data[0].length;
	};

	getRow = (index) => this.data[index];

	getCol = (index) => {
		let res = [];
		for (let i = 0; i < this.rows; i++) {
			res = [...res, this.data[i][index]];
		}
		return res;
	};

	print() {
		console.table(this.data);
	}
}
