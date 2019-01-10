all: ppFinal


ppFinal : main.o cnnModel.o parser.o
	g++ -o ppFinal main.o cnnModel.o parser.o -fopenmp

.cpp.o:
	g++ -c $< -fopenmp


clean:
	rm -rf ppFinal *.o
