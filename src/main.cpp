
#include <iostream>
#include <lapacke.h>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <vector>
#include <array>
#include <memory>
#include <fstream>
#include <chrono>

void fatalError(bool condition, const std::string& message);

double magnitude(int p[3]);
int smallest(int a, int b);
int calculateReflections(int x1, int x2, int cubeSize);

double JFunctionSin(double distance, double k);
double JFunctionExp(double distance, double gamma);
double JFunctionStep(double distance, double d_cutoff);

using Clock = std::chrono::_V2::system_clock::time_point;



class PointCube {
public:
	PointCube() = default;

    PointCube(size_t size)
        : size_(size), data(size * size * size, false) {}  // initialize all to false

	PointCube(const PointCube& other)
		:size_(other.size_), data(other.data) {}
	
    std::vector<bool>::reference at(size_t i, size_t j, size_t k) {
        return data[i * size_ * size_ + j * size_ + k];
    }

	bool at(size_t i, size_t j, size_t k) const {
        return data[i * size_ * size_ + j * size_ + k];
    }

	void reset(size_t size)
	{
		size_ = size;
		data.assign(size * size * size, false);
	}

	size_t size()
	{
		return size_;
	}

private:
    size_t size_;
    std::vector<bool> data;
};

class PositionList {
	public:
	PositionList() = default;

	PositionList(size_t size)
		:size_(size), data(size * 3, 0) {}

	PositionList(const PositionList& other)
		:size_(other.size_), data(other.data) {}

	double& at(size_t i, size_t component) {
        return data[3 * i + component];
    }

	double at(size_t i, size_t component) const {
        return data[3 * i + component];
    }

	size_t size() const
	{
		return size_;
	}

	void reset(size_t size)
	{
		size_ = size;
		data.assign(size * 3, 0);
	}

	private:
	size_t size_;
	std::vector<double>data;

};

class ParticleMatrix {
public:
    ParticleMatrix() = default;

    ParticleMatrix(size_t pnum)
        : particleNumber_(pnum), data(new double[pnum * pnum]()) {}

    ~ParticleMatrix() {
        delete[] data;
    }

	ParticleMatrix(const ParticleMatrix& other)
        : particleNumber_(other.particleNumber_),
          data(new double[other.particleNumber_ * other.particleNumber_]) 
{
        std::copy(other.data, other.data + particleNumber_ * particleNumber_, data);
    }

    double& at(size_t i, size_t j) {
        return data[i * particleNumber_ + j];
    }

    double at(size_t i, size_t j) const {
        return data[i * particleNumber_ + j];
    }

    double* raw_data() {
        return data;
    }

    size_t size() const {
        return particleNumber_;
    }

	void reset(size_t pnum) {
		particleNumber_ = pnum;
		delete[] data;
		data = new double[pnum * pnum]();
	}

	ParticleMatrix transpose() const {
		ParticleMatrix transposed(particleNumber_);
		for (size_t i = 0; i < particleNumber_; ++i) {
			for (size_t j = 0; j < particleNumber_; ++j) {
				transposed.at(j, i) = at(i, j);
			}
		}
		return transposed;
	}

private:
    size_t particleNumber_;
    double* data = nullptr;
};

class Simulation{
private:
	enum class ControlVariable{
		CUTOFF_VALUE,
		J_PARAMETER,
		IMPURITY_PERCENT,
		CUBE_SIZE
	};

	enum ModificationMode{
	LINEAR, EXPONENTIAL
	};

	uint16_t m_cubeSize;
	double m_impurityPercent;
	double m_jParameter;
	double m_cutoffValue;
	uint16_t m_particleNum;
	bool m_requiresRecalculation;
	ModificationMode m_modificationMode;
	ControlVariable m_controlVariable;
	double m_controlVarIncrement;
	size_t m_maxIterations;

	PointCube m_material;
	PositionList m_positions;
	ParticleMatrix m_distances;
	ParticleMatrix m_jMatrix, m_hMatrix;
	std::vector<double> m_eigenvalues;

	int m_info;

	std::vector<double> m_eigenvalueCutoff;
	double m_curieTemperature;

	std::ofstream m_csvCurieTemperatures;  
	std::ofstream m_csvEigenValues;

	double(*m_jFunction)(double, double);

	bool m_logEigenvalues, m_logCurieTemperature, m_logConsole, m_logIterationDuration;

	// member functions

	void populate()
	{
		for (int i =0; i < m_positions.size(); i++)
		{
			int x = rand() % m_material.size();
			int y = rand() % m_material.size();
			int z = rand() % m_material.size();

			if (m_material.at(x,y,z) == false)
			{
				m_material.at(x,y,z) = true;

				m_positions.at(i,0) = x;
				m_positions.at(i,1) = y;
				m_positions.at(i,2) = z;
				
			}
			else
			{
				i--;
			}
		}
	}
	void calculateDistances()
	{
		size_t n = m_positions.size();
		for (size_t i = 0; i < n; i++)
		{
			for (size_t j = 0; j < n; j++)
			{
				if (i == j)
					this->m_distances.at(i, j) = 0;
				else
				{
					int effective_relative_position[3];

					effective_relative_position[0] = calculateReflections(m_positions.at(i, 0), m_positions.at(j, 0),m_material.size());
					effective_relative_position[1] = calculateReflections(m_positions.at(i, 1), m_positions.at(j, 1), m_material.size());
					effective_relative_position[2] = calculateReflections(m_positions.at(i, 2), m_positions.at(j, 2), m_material.size());

					m_distances.at(i, j) = magnitude(effective_relative_position);
				}
			}
		}
	}
	void calculateJ()
	{

		for (size_t i=0;i < m_jMatrix.size() ;i++)
		{
			for(size_t j = 0 ; j < m_jMatrix.size();j++)
			{
				if (i==j)
					m_jMatrix.at(i,i)=0;
				else
					m_jMatrix.at(i,j) = m_jFunction(m_distances.at(i,j), m_jParameter);
			}
		}
	}
	void calculateH()
	{
		for (size_t i=0;i < m_jMatrix.size() ;i++)
		{
			double sum_i =0;
			for(size_t n = 0 ; n < m_jMatrix.size();n++)
			{
				sum_i+=m_jMatrix.at(i,n);
			}
			
			for(size_t j = 0 ; j < m_jMatrix.size();j++)
			{
				m_hMatrix.at(i,j) = (i==j)?sum_i:0-m_jMatrix.at(i,j);
			}

		}
	}
	void eigenvalueCutoff()
	{
		m_eigenvalueCutoff.clear();
		for (int i = 0; i < m_eigenvalues.size(); i++)
		{
			if(m_eigenvalues[i] > m_cutoffValue)
				m_eigenvalueCutoff.push_back(m_eigenvalueCutoff[i]);
			else 
				m_eigenvalueCutoff.push_back(m_cutoffValue);
		}
	}
	void reverseEigenvalues()
	{
		for (size_t i = 0 ; i < m_eigenvalueCutoff.size(); i++)
		{
			if (m_eigenvalueCutoff[i] != 0)
				m_eigenvalueCutoff[i] = 1 / m_eigenvalueCutoff[i];
			else{
				m_eigenvalueCutoff[i] = std::numeric_limits<double>::max();
			}
		}
	}
	void calculateCurieTemperature()
	{
		double coefficient = 1;
		double result = 0;

		eigenvalueCutoff();
		reverseEigenvalues();

		for (double term : m_eigenvalueCutoff)
			result += term;

		m_curieTemperature = coefficient * m_eigenvalueCutoff.size() / result;
	}
	void pipeline(int index)
	{
		if (m_requiresRecalculation || index == 0)
		{
			if(m_logConsole)
				std::cout << "Calculating...";

			m_particleNum = static_cast<uint16_t>(pow(m_cubeSize, 3) * m_impurityPercent);
			m_material.reset(m_cubeSize);
			m_positions.reset(m_particleNum);
			m_distances.reset(m_particleNum);
			m_jMatrix.reset(m_particleNum);
			m_hMatrix.reset(m_particleNum);
			m_eigenvalues.assign(m_particleNum,0.0);

			populate();
			calculateDistances();
			calculateJ();
			calculateH();

			m_info = LAPACKE_dsyev(LAPACK_COL_MAJOR, 'V', 'U', m_particleNum, m_hMatrix.raw_data(), m_particleNum, m_eigenvalues.data());
			fatalError(m_info > 0, "The algorithm failed to compute eigenvalues");

			if(m_logConsole)
				std::cout << "\r";
		}
		calculateCurieTemperature();
	}
	
	void modifyControlVariable()
	{
		switch(m_controlVariable)
		{
			case ControlVariable::CUTOFF_VALUE:
				m_cutoffValue  *= m_controlVarIncrement;
				break;
			case ControlVariable::J_PARAMETER:
				m_jParameter += m_controlVarIncrement;
				break;
			case ControlVariable::IMPURITY_PERCENT:
				m_impurityPercent += m_controlVarIncrement;
				break;
			case ControlVariable::CUBE_SIZE:
				m_cubeSize += static_cast<uint16_t>(m_controlVarIncrement);
				break;
			default:
				fatalError(true, "Invalid control variable");
		}
	}
	double getControlVariableValue() const
	{
		switch(m_controlVariable)
		{
			case ControlVariable::CUTOFF_VALUE:
				return m_cutoffValue;
			case ControlVariable::J_PARAMETER:
				return m_jParameter;
			case ControlVariable::IMPURITY_PERCENT:
				return m_impurityPercent;
			case ControlVariable::CUBE_SIZE:
				return static_cast<double>(m_cubeSize);
			default:
				fatalError(true, "Invalid control variable");
		}
	}

	void openFiles()
	{
		if(m_logCurieTemperature){
			m_csvCurieTemperatures.open("curie_temperarures.csv");
			fatalError(!m_csvCurieTemperatures.is_open(), "Failed to open eigenvalues.csv for writing.");

		}

		if(m_logEigenvalues){
			m_csvEigenValues.open("eigenvalues.csv");
			fatalError(!m_csvEigenValues.is_open(), "Failed to open eigenvalues.csv for writing.");
		}

	}
	

public:

	Simulation()
	{

	}

	int initialize()
	{
		m_cubeSize = 10;
		m_impurityPercent = 0.5;
		m_jParameter = 1.1;
		m_cutoffValue = 1;
		m_particleNum = 0;
		m_requiresRecalculation = false;
		m_controlVariable = ControlVariable::CUTOFF_VALUE;
		m_controlVarIncrement = 0.9;
		m_maxIterations = 100;
		m_jFunction = JFunctionSin;

		m_logConsole=true;
		m_logCurieTemperature =true;
		m_logEigenvalues = true;
		m_logIterationDuration = true;

		return 0;
	}

	void mainLoop()
	{
		openFiles();
		
		for (int i = 0; i < m_maxIterations && getControlVariableValue() > 0; i++)
		{
			Clock start = std::chrono::high_resolution_clock::now();

			pipeline(i);

			Clock end = std::chrono::high_resolution_clock::now();
			double duration = std::chrono::duration<double>(end - start).count();

			fatalError(!m_csvCurieTemperatures.is_open(), "The process was interrupted");

			m_csvCurieTemperatures << i + 1 << "," << getControlVariableValue() << "," << m_curieTemperature << "," << duration << "\n";
			m_csvCurieTemperatures.flush();
			if(m_logConsole){
				std::cout << i + 1 << ". " << getControlVariableValue() << " -> T= " << m_curieTemperature;
				if(m_logIterationDuration)
					std::cout << ", (" << duration << "s)\n";
				else
					std::cout << "\n";
			}

			modifyControlVariable();
		}

		m_csvCurieTemperatures.close();
	}

};

void fatalError(bool condition, const std::string& message) {
	if (condition) {
		std::cerr << message << "\n";
		exit(EXIT_FAILURE);
	}
}


double magnitude(int p[3])
{
    return sqrt(pow(p[0],2)+pow(p[1],2)+pow(p[2],2));
}

int smallest(int a, int b)
{
    return a < b ? a : b;
}

int calculateReflections(int x1, int x2, int cubeSize)
{
    int diff = smallest(abs(x1 - x2), abs(x1-x2-cubeSize));
    diff = smallest(diff, abs(x2-x1-cubeSize));
    return diff;
    
}



double JFunctionSin(double distance, double k)
{
    return sin(k * distance) / distance * distance;
}

double JFunctionExp(double distance, double gamma)
{
    return distance = exp(-gamma * distance);
}

double JFunctionStep(double distance, double d_cutoff)
{
    return distance > d_cutoff ? 0 : 1;
}





int main()
{
	Simulation simulation;

	simulation.initialize();
	simulation.mainLoop();

	system("pause");
	return 0;
}
