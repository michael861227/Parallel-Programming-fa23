#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //
  __pp_vec_float x, result;
  __pp_vec_int exp, count;
  __pp_vec_int one = _pp_vset_int(1);
  __pp_vec_int zero = _pp_vset_int(0);
  __pp_vec_float clamped = _pp_vset_float(9.999999f);
  __pp_mask maskAll, expMask, counterPositiveMask, clampedMask;
  
  // For N % VECTOR_WIDTH != 0
  for (int i = N;i < N + VECTOR_WIDTH;i++) {
    values[i] = 0.0f;
    exponents[i] = 1;
  }


  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    // Initial Mask
    maskAll = _pp_init_ones();
    expMask = _pp_init_ones(0);
    clampedMask = _pp_init_ones(0);
    count = _pp_vset_int(0);

    // load values and exponents
    // x = values[i : i + VECTOR_WIDTH]
    // exp = exponents[i: i + VECTOR_WIDTH]
    _pp_vload_float(x, values + i, maskAll);        
    _pp_vload_int(exp, exponents + i, maskAll);     

    // if exp == 0, output = 1
    _pp_veq_int(expMask, exp, zero, maskAll);
    _pp_vset_float(result, 1.f, expMask);

    // else
    expMask = _pp_mask_not(expMask);
    _pp_vmove_float(result, x, expMask);                       // result = x
    _pp_vsub_int(count, exp, one, expMask);                    // count = exp - 1
    _pp_vgt_int(counterPositiveMask, count, zero, maskAll);    // counterPositiveMask = count > 0
    
    while (_pp_cntbits(counterPositiveMask)) {
      _pp_vmult_float(result, result, x, counterPositiveMask); // result = result * x
      _pp_vsub_int(count, count, one, counterPositiveMask);    // count = count - 1
      _pp_vgt_int(counterPositiveMask, count, zero, maskAll);  // counterPositiveMask = count > 0 
    }
    
    _pp_vgt_float(clampedMask, result, clamped, maskAll);      // result > clamped
    _pp_vmove_float(result, clamped, clampedMask);        // result = clamped

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{

  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //

  __pp_vec_float sum = _pp_vset_float(0);
  __pp_mask mask = _pp_init_ones();
  int shift = VECTOR_WIDTH;

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    __pp_vec_float vec;

    _pp_vload_float(vec, values + i, mask);
    _pp_vadd_float(sum, sum, vec, mask);
  }

  while (shift != 1) {
    _pp_hadd_float(sum, sum);
    _pp_interleave_float(sum, sum);
    shift >>= 1;
  }

  return sum.value[0];
}