#include<iostream>
#include"..\..\Matrix\Matrix\Matrix.cpp"

#define NN_TYPE double

using namespace std;

class NeuralNetwork {
private:
	vector<Matrix> netNode;
	vector<Matrix> outNode;
	vector<Matrix> weight;
	vector<int> network;
	vector<vector<NN_TYPE>> xLable;
	vector<vector<NN_TYPE>> yLable;
	int index;
	vector<function<MATRIX_TYPE(MATRIX_TYPE)>> actFunc;
	int epoch;	// default = 100
	double eta;	// default = 0.01
	
private:
	NeuralNetwork& forward();
	NeuralNetwork& backPropagation();
public:
	NN_TYPE sumOfError(function<NN_TYPE(Matrix&, int)> lossFunc);
public:
	NeuralNetwork(const vector<int> net, const vector<function<MATRIX_TYPE(MATRIX_TYPE)>> func);
	NeuralNetwork& input(const vector<MATRIX_TYPE> data);
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

}
NN_TYPE NeuralNetwork::sumOfError(function<NN_TYPE(Matrix&, int)> lossFunc) {
	return lossFunc(outNode.back(), index);
}
NeuralNetwork::NeuralNetwork(const vector<int> net, const vector<function<MATRIX_TYPE(MATRIX_TYPE)>> func) : epoch(100), eta(0.01), index(0) {
	network.assign(net.begin(), net.end());
	netNode.resize(net.size() - 1);
	outNode.resize(net.size());
	weight.resize(net.size() - 1);
	for (auto f : func) {
		actFunc.push_back(f);
	}
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
NeuralNetwork& NeuralNetwork::input(const vector<MATRIX_TYPE> data) {
	if (data.size() != network[0]) {
		cout << "The size of the input layer does not match." << endl;
		return *this;
	}
	for (int i = 0; i < network[0]; i++) {
		outNode[0](i, 0) = data[i];
	}
	return *this;
}
NeuralNetwork& NeuralNetwork::run() {
	epoch = 1;
	for (int i = 0; i < epoch; i++) {
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

	vector<vector<NN_TYPE>> xLable = { { 0,0 },{ 0,1 },{ 1,0 },{ 1,1 } };
	vector<vector<NN_TYPE>> yLable = { { 0 },{ 1 },{ 1 },{ 1 } };

	auto ss = [&yLable](Matrix &mat, int index) {
		NN_TYPE sum = 0;
		for (int i = 0; i < yLable[index].size(); i++) {
			sum += 0.5*(mat(i, 0) - yLable[index][i])*(mat(i, 0) - yLable[index][i]);
		}
		return sum;
	};

	NeuralNetwork nn({ 2,3,3,1 }, { sigmoid, sigmoid, sigmoid });
	nn.input({ 1,1 });
	nn.run();
	cout << nn << endl << endl;
	cout << nn.sumOfError(ss) << endl;
}