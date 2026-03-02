include Make_linux.inc
#include Make_msys2.inc
#include Make_osx.inc

CXXFLAGS_BASE = -std=c++17 -Iinclude

ifeq ($(DEBUG),yes)
CXXFLAGS2 ?= $(CXXFLAGS_BASE) -g -O0 -Wall -fbounds-check -pedantic -D_GLIBCXX_DEBUG
else
CXXFLAGS2 ?= $(CXXFLAGS_BASE) -O2 -march=native -Wall
endif

SRC = $(wildcard src/*.cpp)
OBJ = $(patsubst src/%.cpp,build/%.o,$(SRC))
TARGET = ant_simu.exe

.PHONY: all clean help

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS2) $^ -o $@ $(LIB)

build/%.o: src/%.cpp | build
	$(CXX) $(CXXFLAGS2) -c $< -o $@

build:
	@mkdir -p build

clean:
	@rm -rf build/*.o $(TARGET) *~

help:
	@echo "Available targets:"
	@echo "    all            : compile the vectorized executable"
	@echo "    clean          : remove objects and binary"
	@echo "Add DEBUG=yes to compile in debug"
	@echo "Configuration:"
	@echo "    CXX      : $(CXX)"
	@echo "    CXXFLAGS : $(CXXFLAGS2)"
