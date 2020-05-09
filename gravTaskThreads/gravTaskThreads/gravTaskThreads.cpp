#include <iostream>
#include <cstdio>
#include <fstream>
#include <vector>
#include <chrono>
#include <ctime>
#include <thread>

#define GravConst 6.674e-11

struct Point {
	float x, y, z;
	float vx = 0, vy = 0, vz = 0;
	float ax = 0, ay = 0, az = 0;
	float m;
};

static const int THREADS_NUMBER = std::thread::hardware_concurrency();

constexpr float sqr(float x) { return x * x; }




//std::vector<float> forcesCalc(std::vector<Point>& points, int i) {
//	std::vector<float> forces = { 0,0,0 };
//	for (int j = 0; j < points.size(); j++) {
//		if (i == j) continue;
//
//		float dx = points[j].x - points[i].x;
//		float dy = points[j].y - points[i].y;
//		float dz = points[j].z - points[i].z;
//
//		float dist = sqrt(dx * dx + dy * dy + dz * dz);
//
//		float F = (GravConst * points[i].m * points[j].m) / sqr(dist);
//		forces[0] += F * dx / dist;
//		forces[1] += F * dy / dist;
//		forces[2] += F * dz / dist;
//	}
//	return forces;
//}
//
//void nextStep(std::vector<Point>& points, std::vector<Point>& resPoints, int& dt, int& i) {
//	//for (int i = 0; i < points.size(); i++) {
//	std::vector<float> forcesArr = forcesCalc (points, i);
//		resPoints[i].x = points[i].x + points[i].vx * dt + (points[i].ax * sqr(dt)) / 2;
//		resPoints[i].y = points[i].y + points[i].vy * dt + (points[i].ay * sqr(dt)) / 2;
//		resPoints[i].z = points[i].z + points[i].vz * dt + (points[i].az * sqr(dt)) / 2;
//
//		resPoints[i].m = points[i].m;
//
//		resPoints[i].vx = points[i].vx + points[i].ax * dt;
//		resPoints[i].vy = points[i].vy + points[i].ay * dt;
//		resPoints[i].vz = points[i].vz + points[i].az * dt;
//
//		resPoints[i].ax = forcesArr[0] / points[i].m;
//		resPoints[i].ay = forcesArr[1] / points[i].m;
//		resPoints[i].az = forcesArr[2] / points[i].m;
//}


void forcesCalc_threading(std::vector<Point>& points, std::vector<Point>& resPoints, const int thread_number, int& dt) {
	const int n_operations = points.size() / THREADS_NUMBER;
	const int rest_operations = points.size() % THREADS_NUMBER;

	int start_op, end_op;

	if (thread_number == 0) {
		start_op = n_operations * thread_number;
		end_op = (n_operations * (thread_number + 1)) + rest_operations;
	}
	else {
		start_op = n_operations * thread_number + rest_operations;
		end_op = (n_operations * (thread_number + 1)) + rest_operations;
	}

	for (int op = start_op; op < end_op; ++op) {
		float fx = 0, fy = 0, fz = 0;
		for (int j = 0; j < points.size(); j++) {
			if (j == op) continue;
			float dx = points[j].x - points[op].x;
			float dy = points[j].y - points[op].y;
			float dz = points[j].z - points[op].z;

			float dist = sqrt(dx * dx + dy * dy + dz * dz);

			float F = (GravConst * points[op].m * points[j].m) / sqr(dist);
			fx += F * dx / dist;
			fy += F * dy / dist;
			fz += F * dz / dist;
		}
		int i = op;
		resPoints[i].x = points[i].x + points[i].vx * dt + (points[i].ax * sqr(dt)) / 2;
		resPoints[i].y = points[i].y + points[i].vy * dt + (points[i].ay * sqr(dt)) / 2;
		resPoints[i].z = points[i].z + points[i].vz * dt + (points[i].az * sqr(dt)) / 2;

		resPoints[i].m = points[i].m;

		resPoints[i].vx = points[i].vx + points[i].ax * dt;
		resPoints[i].vy = points[i].vy + points[i].ay * dt;
		resPoints[i].vz = points[i].vz + points[i].az * dt;

		resPoints[i].ax = fx / points[i].m;
		resPoints[i].ay = fy / points[i].m;
		resPoints[i].az = fz / points[i].m;
	}
}


void readPointsData(const std::string& name, std::vector<Point>& data, int& size) {
	std::ifstream infile(name);


	Point point;

	for (int i = 0; i < size; i++) {

		infile >> point.x >> point.y >> point.z >> point.m;
		data.push_back(point);
	}
}


void readData(const std::string& name, int& size, int& iterations, int& dt) {

	std::ifstream in(name);
	in >> size >> iterations >> dt;
}

void writeFile(std::ofstream& outfile, std::vector<Point>& data) {

	for (int i = 0; i < data.size(); i++) {
		outfile << data[i].x << ' ' << data[i].y << ' ' << data[i].z << ' ' << data[i].m << "\t\t";
	}
	outfile << "\n";

}



//Генератор случайных точек
//void genPoints() {
//	srand(time(NULL));
//	std::ofstream out("inputDataPoint.txt", std::ios::trunc);
//	for (int i = 0; i < 100; i++) {
//		out << 0.001 * (rand() % 2001 - 1000) << " " <<  0.001 * (rand() % 2001 - 1000) << " " << 0.001 * (rand() % 2001 - 1000) << " " << 0.01 * (rand() % 101) << std::endl;
//	}
//	out.close();
//}

int main()
{
	std::ofstream outfile("output.txt", std::ios::trunc);
	int size, iterations, dt;

	readData("Data.txt", size, iterations, dt);

	std::vector<std::thread> threadVec;

	


	std::vector<Point> dataFirst;
	dataFirst.reserve(size);
	//считываем данные

	readPointsData("inputDataPoint.txt", dataFirst, size);

	//промежуточный вектор

	std::vector<Point> dataSecond(size);

	for (int i = 0; i < THREADS_NUMBER; ++i) {
	/*	std::thread th(forcesCalc_threading, dataFirst, dataSecond, i, dt);*/
		threadVec.push_back(std::thread(forcesCalc_threading, std::ref(dataFirst), std::ref(dataSecond), i, std::ref(dt)));
	}
	//тесты
	dataFirst[1].vy = -0.0002;

	//счётчик времени
	auto now = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < iterations; i++)
	{
		//if (i % 2 == 0) {
		//	/*nextStep(dataFirst, dataSecond, dt);*/
		//	writeFile(outfile, dataSecond);
		//}
		//else {
		//	/*nextStep(dataSecond, dataFirst, dt);*/
		//	writeFile(outfile, dataFirst);
		//}
		if (i > 0) {
			std::vector<Point> tmp = dataFirst;
			dataFirst = dataSecond;
			dataSecond = tmp;
		}
		for (int i = 0; i < THREADS_NUMBER; ++i) {
			if (threadVec[i].joinable())
				threadVec[i].join();
		}
		writeFile(outfile, dataFirst);
	}


	//расчёт времени
	auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - now);
	std::cout << "Time : " << elapsed.count() << "s.\n";

	outfile.close();
	/*genPoints();*/


	return 0;
}
