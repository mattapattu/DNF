//-----------------------------------------------
// Version 2.0    22/08/18 
//-----------------------------------------------
// fichier interaçage C++ -> python
//
// utilisation avec :  _ "pixy.cpp" version 2.0
//	 	       _ "pixy.h"   version 2.0

// fonction pixy_get_frame() : capture d'images
// 	en c++ :    utilisation d'un memcpy inutil
// 	en python : appel de la fonction  pixy_get_frame(64000)

%module pixy

%{
#define SWIG_FILE_WITH_INIT
#include "pixy.h"
%}

%include "stdint.i"
%include "carrays.i"
%include "numpy.i"

%init%{
import_array();
%}

//%apply (unsigned char** ARGOUTVIEW_ARRAY1, int* DIM1) {(unsigned char** current_frame, int* nbPixels)} // avec le struct

%apply (unsigned char* ARGOUT_ARRAY1, int DIM1) {(unsigned char* current_frame, int nbPixels)}           // uniquement la fonction Pixy_get_frame()

%include "/home/leat/pixy/src/host/libpixyusb/include/pixy.h" 

%array_class(struct Block, BlockArray);

int  pixy_init();
void pixy_close();
void pixy_error(int error_code);
int  pixy_blocks_are_new();
int pixy_get_blocks(uint16_t max_blocks, BlockArray *blocks);
int  pixy_rcs_set_position(uint8_t channel, uint16_t position);
int pixy_get_frame(unsigned char * current_frame, int nbPixels);

struct Block
{
  uint16_t type;
  uint16_t signature;
  uint16_t x;
  uint16_t y;
  uint16_t width;
  uint16_t height;
  int16_t  angle;
};
