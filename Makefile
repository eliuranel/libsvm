CXX ?= g++
CFLAGS = -Wall -Wconversion -fPIC
SHVER = 2
OS = $(shell uname)

all: svm-train svm-predict svm-scale svm-gowerscale

lib: svm.o
	if [ "$(OS)" = "Darwin" ]; then \
		SHARED_LIB_FLAG="-dynamiclib -Wl,-install_name,libsvm.so.$(SHVER)"; \
	else \
		SHARED_LIB_FLAG="-shared -Wl,-soname,libsvm.so.$(SHVER)"; \
	fi; \
	$(CXX) $${SHARED_LIB_FLAG} svm.o -o libsvm.so.$(SHVER)

svm-predict: svm-predict.c svm.o
	$(CXX) $(CFLAGS) -g svm-predict.c svm.o -o svm-predict -lm
svm-train: svm-train.c svm.o
	$(CXX) $(CFLAGS) -g svm-train.c svm.o -o svm-train -lm
svm-scale: svm-scale.c
	$(CXX) $(CFLAGS) -g svm-scale.c -o svm-scale
svm-gowerscale: svm-gowerscale.c svm.o
	$(CXX) $(CFLAGS) -g svm-gowerscale.c svm.o -o svm-gowerscale -lm
svm.o: svm.cpp svm.h
	$(CXX) $(CFLAGS) -g -c svm.cpp
clean:
	rm -f *~ svm.o svm-train svm-predict svm-scale svm-gowerscale libsvm.so.$(SHVER)
