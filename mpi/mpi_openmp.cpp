#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <cstdio>
#include <fstream>
#include <vector>
#include <ctime>
#include <cmath>
#include <time.h>

#define EPS 1e-6
#define GravConst 6.674e-11

MPI_Datatype MPI_POINT;

struct Point
{
	float x,y,z;
	float vx, vy, vz;
	float ax, ay, az;
	float m;
};

inline float sqr(float x) { return x * x; }

const int MasterProcessor = 0;

std::vector<float> forcesCalc(std::vector<Point>& points, int& i, int& size){
	std::vector<float> forces(3);
	for (int j = 0; j < size; ++j)
	{
		if (i == j) continue;

		float dx = points[j].x - points[i].x;
		float dy = points[j].y - points[i].y;
		float dz = points[j].z - points[i].z;

		float dist = sqrt(sqr(dx) + sqr(dy) + sqr(dz));

		if (dist < EPS) dist = EPS;

		float F = (GravConst * points[i].m * points[j].m) / sqr(dist);
		forces[0] = F * dx / dist;
		forces[1] = F * dy / dist;
		forces[2] = F * dz / dist;
	}
	return forces;
}

void nextStep(std::vector<Point>& points, std::vector<Point>& resPoints, int& i, const int j, const int dt, int& size){
	std::vector<float> forcesArr = forcesCalc(points, i, size);

	resPoints[j].x = points[i].x + points[i].vx * dt + (points[i].ax * sqr(dt))/2;
	resPoints[j].y = points[i].y + points[i].vy * dt + (points[i].ay * sqr(dt))/2;
	resPoints[j].z = points[i].z + points[i].vz * dt + (points[i].az * sqr(dt))/2;

	resPoints[j].m = points[i].m;

	resPoints[j].vx = points[i].vx + points[i].ax * dt;
	resPoints[j].vy = points[i].vy + points[i].ay * dt;
	resPoints[j].vz = points[i].vz + points[i].az * dt;

	resPoints[j].ax = forcesArr[0] / points[i].m;
	resPoints[j].ay = forcesArr[1] / points[i].m;
	resPoints[j].az = forcesArr[2] / points[i].m;
}

void forcesCalc_mpi_openmp(std::vector<Point>& points, std::vector<Point>& resPoints, const int partSize, const int rank, const int dt, int& size, const int rankMP){

	// std::cerr << "Thread number : " << thread_number << std::endl;
	// std::cerr << "Start_op : " << start_op << std::endl;
	// std::cerr << "End_op : " << end_op << std::endl;

	const int n_operations = partSize / 4;
	const int rest_operations = partSize % 4;

	int start_op, end_op;

	if (rankMP == 0){
		start_op = n_operations * rankMP;
		end_op = (n_operations * (rankMP + 1)) + rest_operations;
	}
	else{
		start_op = n_operations * rankMP + rest_operations;
		end_op = (n_operations * (rankMP + 1)) + rest_operations;
	}

	for(int op = start_op; op < end_op; ++op){
		int numberPoint = ((rank - 1) * partSize) + op;
		nextStep(points, resPoints, numberPoint, op, dt , size);
	}


	// for (int i = 0; i < partSize; i++){
	// 	int numberPoint = ((rank - 1) * partSize) + i;
	// 	nextStep(points, resPoints, numberPoint , i, dt, size);
	// }

	// for (int i = 0 ; i < partSize; i++){
	// 	std::cerr << resPoints[i].y << std::endl;
	// }
	// MPI_Send(resPoints.data() + pindex, partSize , MPI_POINT, 0 , 0 ,MPI_COMM_WORLD);
}

void readData (std::ifstream& in, int &size, int& iterations, int& dt){
	//std::ifstream in(name);
	in >> size >> iterations >> dt;
}

void readPointsData(std::ifstream& in, std::vector<Point>& data, int& size){
	//std::ifstream in(name);

	Point point;
	for (int i = 0; i < size; ++i)
	{
		in >> point.x >> point.y >> point.z >> point.m;
		point.vx = point.vy = point.vz = point.ax = point.ay = point.az = 0;
		data.push_back(point);
	}
}

void writeFile (std::ofstream& outfile, std::vector<Point> data){
	for (int i = 0; i < data.size(); ++i)
	{
		outfile << i << " : " <<data[i].x <<  ' ' <<data[i].y << ' ' << data[i].z << ' ' << data[i].m << "\t\t";
	}
	outfile << '\n';
}

void worker(int rank){
	MPI_Status status;
	std::vector<Point> points;
	int size , partSize, dt;
	while(1){
		MPI_Recv(&dt, 1 , MPI_INT, 0 , 0 , MPI_COMM_WORLD, &status);
		if (dt == -1 ) break;
		MPI_Recv(&size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
		points.reserve(size);
		MPI_Recv(&partSize, 1, MPI_INT, 0 , 0 , MPI_COMM_WORLD, &status);
		std::vector<Point> resPoints(partSize);
		MPI_Recv(points.data(), size, MPI_POINT, 0 , 0 , MPI_COMM_WORLD, &status);
		#pragma omp parallel num_threads(4)
		{
			//std::cerr << "rank OMP : " << omp_get_thread_num() << " size OMP: " << omp_get_num_threads() << std::endl;
			int rankMP = omp_get_thread_num(); 
			forcesCalc_mpi_openmp(points, resPoints, partSize, rank, dt, size, rankMP);
		}
		MPI_Send(resPoints.data(), partSize, MPI_POINT, 0, 0 , MPI_COMM_WORLD);
	}
}

void master(int worker_count){
	int size, iterations, dt;
	std::ifstream in("Data.txt");
	readData(in, size, iterations, dt);
	std::ofstream outfile ("output.txt", std::ios::trunc);
	// std::cerr << size << " " << iterations << " " << dt << std::endl;
	std::vector<Point> dataFirst;	
	dataFirst.reserve(size);

	std::ifstream inPoint("inputDataPoint.txt");
	readPointsData(inPoint, dataFirst, size);

	//dataFirst[1].vy = -0.0002;

	int killProc = -1;

	if (size < worker_count){
		for (int i = size + 1; i <= worker_count; i++){
			MPI_Send(&killProc, 1, MPI_INT, i, 0 , MPI_COMM_WORLD);
			std::cerr << "Kill number : " << i << "worker" << std::endl;
		}
		worker_count = size;
	}

	int partSize = size / worker_count;
	for (int it = 0 ; it < iterations ; it++){
			for (int i = 1; i <= worker_count; i++){
				MPI_Send(&dt, 1 , MPI_INT, i , 0 , MPI_COMM_WORLD);
				MPI_Send(&size, 1 , MPI_INT, i , 0 , MPI_COMM_WORLD);
				MPI_Send(&partSize, 1 , MPI_INT, i , 0 , MPI_COMM_WORLD);
				MPI_Send(dataFirst.data(), size, MPI_POINT, i , 0 , MPI_COMM_WORLD);
			}

			MPI_Status status;

			for (int i = 1; i <= worker_count; i++){
				std::vector<Point> resPoints(partSize);
				MPI_Recv(resPoints.data(), partSize, MPI_POINT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
				int idProc = status.MPI_SOURCE;
				for(int j = 0; j < partSize; j++){
					dataFirst[(partSize * (idProc - 1)) + j] = resPoints[j];
				}
			}
			writeFile(outfile, dataFirst);
	}
	for(int i = 1; i <= worker_count; i++){
		MPI_Send(&killProc, 1, MPI_INT, i, 0 , MPI_COMM_WORLD);
	}
	outfile.close();
}



int main (int argc, char* argv[]){
	//auto now = std::chrono::high_resolution_clock::now();
	clock_t start = clock();
	int rank, THREADS_COUNT;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &THREADS_COUNT);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	MPI_Type_contiguous(10, MPI_FLOAT, &MPI_POINT);
	MPI_Type_commit(&MPI_POINT);

	if (rank == MasterProcessor){
		master(THREADS_COUNT - 1);
	}
	else{
		worker(rank);
	}

	MPI_Type_free(&MPI_POINT);
	MPI_Finalize();

	clock_t end = clock();
	double elapsed = (double)(end - start)/ CLOCKS_PER_SEC;

	//auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - now);
	std::cerr << "Time :" << elapsed << "ms.\n";
	return 0;
}
