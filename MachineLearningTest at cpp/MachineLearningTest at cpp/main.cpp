#include<iostream>
#include"..\..\Matrix\Matrix\Matrix.cpp"

#define NN_TYPE MATRIX_TYPE
#define NN_LABLE vector<vector<NN_TYPE>>

using namespace std;

class NeuralNetwork {
private:
	vector<Matrix> netNode;
	vector<Matrix> outNode;
	vector<Matrix> weight;
	vector<int> network;
	NN_LABLE  xLable;
	NN_LABLE  yLable;
	int index;
	vector<function<MATRIX_TYPE(MATRIX_TYPE)>> actFunc;
	vector<function<MATRIX_TYPE(MATRIX_TYPE)>> actDFunc;
	int epoch;	// default = 100
	double eta;	// default = 0.01
	
private:
	NeuralNetwork& forward();
	NeuralNetwork& backPropagation();
	int getIndex();
	NeuralNetwork& setInputLayer(vector<NN_TYPE> vec);
public:
	NN_TYPE sumOfError(function<NN_TYPE(NN_TYPE, NN_TYPE)> lossFunc);
	NN_TYPE _directSOE(function<NN_TYPE(Matrix&, int)> lf);
public:
	NeuralNetwork(const vector<int> net, const vector<function<MATRIX_TYPE(MATRIX_TYPE)>> func, const vector<function<MATRIX_TYPE(MATRIX_TYPE)>> dFunc);
	NeuralNetwork& input(const NN_LABLE  xLableData, const NN_LABLE yLableData);
	NeuralNetwork& run();
	vector<int> getNetwork() const;
	void showNetNode(int n);
	void showOutNode(int n);
	void showWeight(int n);
	void showResult();
	friend ostream& operator<<(ostream& os, NeuralNetwork& nn);
};
NeuralNetwork& NeuralNetwork::forward() {
	if (weight.size() != actFunc.size()) {
		cout << "Incorrect number of activation functions." << endl;
		return *this;
	}
	for (int i = 0; i < network.size()-1; i++) {
		outNode[i+1] = netNode[i].dot(outNode[i], weight[i]);
		outNode[i+1].applyFunc(actFunc[i]);
	}
	return *this;
}
NeuralNetwork& NeuralNetwork::backPropagation() {
	return *this;
}
int NeuralNetwork::getIndex() {
	index++;
	if (index >= xLable.size())
		index = 0;
	return index;
}
NeuralNetwork& NeuralNetwork::setInputLayer(vector<NN_TYPE> vec) {
	outNode[0] = vec;
	return *this;
}
NN_TYPE NeuralNetwork::sumOfError(function<NN_TYPE(NN_TYPE, NN_TYPE)> lossFunc) {
	static Matrix tempMat;
	static Matrix yLableMat;
	return tempMat(outNode.back()).applyFunc(lossFunc, yLableMat(yLable[index])).getAllSum();
}
NN_TYPE NeuralNetwork::_directSOE(function<NN_TYPE(Matrix&, int)> lossFunc) {
	return lossFunc(outNode.back(), index);
}
NeuralNetwork::NeuralNetwork(const vector<int> net, const vector<function<MATRIX_TYPE(MATRIX_TYPE)>> func, const vector<function<MATRIX_TYPE(MATRIX_TYPE)>> dFunc)
	: epoch(100), eta(0.01), index(-1) {
	network.assign(net.begin(), net.end());
	netNode.resize(net.size() - 1);
	outNode.resize(net.size());
	weight.resize(net.size() - 1);
	actFunc.assign(func.begin(), func.end());
	actDFunc.assign(dFunc.begin(), dFunc.end());
	for (int i = 0; i < netNode.size(); i++) {
		netNode[i].resize(1, network[i+1]);
	}
	for (int i = 0; i < outNode.size(); i++) {
		outNode[i].resize(1, network[i]);
	}
	for (int i = 0; i < weight.size(); i++) {
		weight[i].resize(network[i], network[i + 1]).randomReal(-1, 1);
	}
}
NeuralNetwork& NeuralNetwork::input(const NN_LABLE  xLableData, const NN_LABLE yLableData) {
	this->xLable.assign(xLableData.begin(), xLableData.end());
	this->yLable.assign(yLableData.begin(), yLableData.end());
	return *this;
}
NeuralNetwork& NeuralNetwork::run() {
	epoch = 4;
	for (int i = 0; i < epoch; i++) {
		setInputLayer(xLable[getIndex()]);
		forward();
	}
	return *this;
}
vector<int> NeuralNetwork::getNetwork() const {
	return network;
}
void NeuralNetwork::showNetNode(int n) {
	cout << netNode[n] << endl;
}
void NeuralNetwork::showOutNode(int n) {
	cout << outNode[n] << endl;
}
void NeuralNetwork::showWeight(int n) {
	cout << weight[n] << endl;
}
void NeuralNetwork::showResult() {
	cout << outNode.back() << endl;
}
ostream& operator<<(ostream& os, NeuralNetwork& nn) {
	os << "Neural Network :: [ ";
	for (auto &layer : nn.getNetwork()) {
		os << layer << " ";
	}
	os << "]" << endl;
	for (int i = 0; i < nn.getNetwork().size()-1; i++) {
		nn.showOutNode(i);
		nn.showWeight(i);
		nn.showNetNode(i);
	}
	os << "Result Output" << endl;
	nn.showResult();
	return os;
}

int main()
{
	auto sigmoid = [](MATRIX_TYPE x) {
		return 0.5*(x / (1 + ABS(x))) + 0.5;
	};

	NN_LABLE  xLable = { { 0,0 },{ 0,1 },{ 1,0 },{ 1,1 } };
	//NN_LABLE  yLable = { { 0 },{ 1 },{ 1 },{ 1 } };
	NN_LABLE yLable(4);
	yLable[0].resize(1000);
	yLable[1].resize(1000);
	yLable[2].resize(1000);
	yLable[3].resize(1000);

	auto d_sse = [&yLable](Matrix &mat, int index) {
		NN_TYPE sum = 0;
		for (int i = 0; i < yLable[index].size(); i++) {
			sum += 0.5*(mat(i, 0) - yLable[index][i])*(mat(i, 0) - yLable[index][i]);
		}
		return sum;
	};

	auto sse = [](MATRIX_TYPE n, MATRIX_TYPE n2) {
		return 0.5*(n - n2)*(n - n2);
	};

	NeuralNetwork nn({ 2,3,3,1000 }, { sigmoid, sigmoid, sigmoid }, { sigmoid, sigmoid, sigmoid });
	nn.input(xLable, yLable);
	nn.run();
	//cout << nn << endl << endl;
	
	int s = clock();
	int r = 1000;
	double sss = 0.0;


	sss = 0.0;
	s = clock();
	for (int i = 0; i < r; i++)
		sss += nn._directSOE(d_sse);
	cout << "T : " << clock() - s << endl;
	cout << sss << endl;


	sss = 0.0;
	s = clock();
	for (int i = 0; i < r; i++)
		sss += nn.sumOfError(sse);
	cout << "T : " << clock() - s << endl;
	cout << sss << endl;
}