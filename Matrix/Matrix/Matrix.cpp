#include<iostream>
#include <iomanip>
#include<string>
#include<random>
#include<vector>
#include<array>
#include<functional>
#include<ctime>

#define MATRIX_TYPE double
#define ABS(val) (((val)<0)?(-(val)):(val))

using namespace std;

typedef vector<MATRIX_TYPE> linear_t;

class Matrix {
private:
	vector<linear_t> matrix;
	int row;
	int col;
	bool transpose;
private:
	void copy(Matrix const &ref);
	MATRIX_TYPE partDot(Matrix &ref1, Matrix &ref2, int x, int y);
public:
	Matrix();
	Matrix(const int row, const int col);
	Matrix(Matrix const &ref);
	~Matrix();
	Matrix& operator=(Matrix const &ref);
	MATRIX_TYPE& operator()(const int x, const int y);
	MATRIX_TYPE operator()(const int x, const int y) const;
	int getRow() const;
	int getCol() const;
	bool getTranspose() const;
	Matrix& resize(const int row, const int col);
	Matrix& assign(const MATRIX_TYPE n);
	Matrix& applyFunc(function<MATRIX_TYPE(MATRIX_TYPE)> func);
	Matrix& applyFunc(function<MATRIX_TYPE(MATRIX_TYPE, MATRIX_TYPE)> func, Matrix &ref);
	Matrix& dot(Matrix &ref1, Matrix &ref2);
	Matrix& add(Matrix &ref);
	Matrix& min(Matrix &ref);
	Matrix& mul(Matrix &ref);
	Matrix& div(Matrix &ref);
	Matrix& T();
	Matrix& zeros();
	Matrix& random(int start, int end);
	Matrix& randomReal(double start, double end);
	friend ostream& operator<<(ostream& os, Matrix& mat);
};
void Matrix::copy(Matrix const &ref) {
	int row = ref.getRow();
	int col = ref.getCol();
	transpose = ref.getTranspose();
	resize(row, col);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			(*this)(j, i) = ref(j, i);
		}
	}
}
MATRIX_TYPE Matrix::partDot(Matrix &ref1, Matrix &ref2, int x, int y) {
	MATRIX_TYPE sum = 0;
	for (int i = 0; i < ref1.getCol(); i++) {
		sum += ref1(i, y) * ref2(x, i);
	}
	return sum;
}
Matrix::Matrix() : transpose(false) {
	resize(0, 0);
}
Matrix::Matrix(const int row, const int col) : transpose(false) {
	resize(row, col);
}
Matrix::Matrix(Matrix const &ref) {
	copy(ref);
}
Matrix::~Matrix() {
}
Matrix& Matrix::operator=(Matrix const &ref) {
	copy(ref);
	return *this;
}
MATRIX_TYPE& Matrix::operator()(const int x, const int y) {
	bool tp = this->getTranspose();
	return this->matrix[(tp) ? x : y][(tp) ? y : x];
}
MATRIX_TYPE Matrix::operator()(const int x, const int y) const {
	bool tp = this->getTranspose();
	const MATRIX_TYPE n = this->matrix[(tp) ? x : y][(tp) ? y : x];
	return n;
}
int Matrix::getRow() const {
	return (this->getTranspose()) ? this->col : this->row;
}
int Matrix::getCol() const {
	return (this->getTranspose()) ? this->row : this->col;
}
bool Matrix::getTranspose() const {
	return this->transpose;
}
Matrix& Matrix::resize(const int row, const int col) {
	if (this->getRow() != row)
		this->matrix.resize(row);
	if (this->getCol() != col)
		for (auto &mat : this->matrix)
			mat.resize(col);
	this->row = row;
	this->col = col;
	return *this;
}
Matrix& Matrix::assign(const MATRIX_TYPE n) {
	return applyFunc([n](MATRIX_TYPE d) {return n;});
}
Matrix& Matrix::applyFunc(function<MATRIX_TYPE(MATRIX_TYPE)> func) {
	for (auto &v : this->matrix) {
		for (auto &n : v) {
			n = func(n);
		}
	}
	return *this;
}
Matrix& Matrix::applyFunc(function<MATRIX_TYPE(MATRIX_TYPE, MATRIX_TYPE)> func, Matrix &ref) {
	try {
		if (this->getRow() != ref.getRow() || this->getCol() != ref.getCol()) {
			throw - 1;
		}
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				(*this)(j, i) = func((*this)(j, i), ref(j, i));
			}
		}
	}
	catch (int err) {
		throw err;
		return *this;
	}
	return *this;
}
Matrix& Matrix::dot(Matrix &ref1, Matrix &ref2) {
	resize(ref1.getRow(), ref2.getCol());
	for (int i = 0; i < this->getRow(); i++) {
		for (int j = 0; j < this->getCol(); j++) {
			(*this)(j, i) = partDot(ref1, ref2, j, i);
		}
	}
	return *this;
}
Matrix& Matrix::add(Matrix &ref) {
	try {
		applyFunc([](MATRIX_TYPE n, MATRIX_TYPE n2) {return n + n2;}, ref);
	}
	catch (int err) {
		cout << err << endl;
	}
	return *this;
}
Matrix& Matrix::min(Matrix &ref) {
	try {
		applyFunc([](MATRIX_TYPE n, MATRIX_TYPE n2) {return n - n2;}, ref);
	}
	catch (int err) {
		cout << err << endl;
	}
	return *this;
}
Matrix& Matrix::mul(Matrix &ref) {
	try {
		applyFunc([](MATRIX_TYPE n, MATRIX_TYPE n2) {return n * n2;}, ref);
	}
	catch (int err) {
		cout << err << endl;
	}
	return *this;
}
Matrix& Matrix::div(Matrix &ref) {
	try {
		applyFunc([](MATRIX_TYPE n, MATRIX_TYPE n2) {return n / n2;}, ref);
	}
	catch (int err) {
		cout << err << endl;
	}
	return *this;
}
Matrix& Matrix::T() {
	this->transpose ^= 1;
	return *this;
}
Matrix& Matrix::zeros() {
	for (auto &v : this->matrix) {
		for (auto &n : v) {
			n = 0;
		}
	}
	return *this;
}
Matrix& Matrix::random(int start, int end) {
	random_device rd;
	mt19937_64 gen(rd());
	uniform_int_distribution<int> dis(start, end);

	for (auto &v : this->matrix) {
		for (auto &n : v) {
			n = dis(gen);
		}
	}
	return *this;
}
Matrix& Matrix::randomReal(double start, double end) {
	random_device rd;
	mt19937_64 gen(rd());
	uniform_real_distribution<double> dis(start, end);
	for (auto &v : this->matrix) {
		for (auto &n : v) {
			n = dis(gen);
		}
	}
	return *this;
}
ostream& operator<<(ostream& os, Matrix& mat)
{
	os << "[Matrix : Row=" << mat.getRow() << " Col=" << mat.getCol() << "]" << endl;
	os.precision(3);
	for (int i = 0; i < mat.getRow(); i++) {
		os << "[ ";
		for (int j = 0; j < mat.getCol(); j++) {
			os << setw(7) << left << mat(j, i) << " ";
		}
		os << "]" << endl;
	}
	return os;
}
/*
int main() {
	Matrix input(1, 3);
	Matrix hidden_z(1, 3);
	Matrix hidden_a(1, 3);
	Matrix output_z(1, 1);
	Matrix output_a(1, 1);

	Matrix weight1(3, 3);
	Matrix weight2(3, 1);

	weight1.randomReal(-1, 1);
	weight2.randomReal(-1, 1);

	input(0, 0) = 1;
	input(1, 0) = 1;
	input(2, 0) = 1;

	auto relu = [](MATRIX_TYPE d) {
		return (d < 0) ? 0 : d;
	};
	auto sigmoid = [](MATRIX_TYPE x) {
		return 0.5*(x / (1 + ABS(x))) + 0.5;
	};

	int t = clock();

	int sdf = 10000;
	while (sdf--)
	{
		//cout << "Input Layer" << endl;
		//cout << input << endl;

		hidden_z = hidden_a.dot(input, weight1);
		//cout << "Hidden Layer" << endl;
		hidden_a.applyFunc(sigmoid);
		//cout << hidden_z << endl << hidden_a << endl;

		output_z = output_a.dot(hidden_a, weight2);
		//cout << "Output Layer" << endl;
		output_a.applyFunc(sigmoid);

	}
	cout << output_z << endl << output_a << endl;

	cout << "Time : " << clock() - t << endl;
}
*/