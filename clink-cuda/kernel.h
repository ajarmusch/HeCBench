// a is a multiple of WGS for simplicity
#define N 8192
#define WGS 256
#define SAMPLE_TEST_LEN 20000

__device__
float sigmoid(float x)
{
  return 1.f / (1.f + expf(-x));
}

__global__ void
lstm_inference(
  const float*__restrict__ d_x, 
  const float*__restrict__ d_inW, 
  const float*__restrict__ d_intW, 
  const float*__restrict__ d_intB, 
  const float*__restrict__ d_outW, 
  const float*__restrict__ d_outB, 
        float*__restrict__ d_y)
{

  int t,i,j;
  int gid = blockDim.x * blockIdx.x + threadIdx.x;

  float h_state[5] = {0,0,0,0,0};
  float c_state[5] = {0,0,0,0,0};
  float i_state[5] = {0,0,0,0,0};
  float f_state[5] = {0,0,0,0,0};
  float o_state[5] = {0,0,0,0,0};
  float g_state[5] = {0,0,0,0,0};

  for (t = 0; t < SAMPLE_TEST_LEN; ++t) {
    for (j = 0; j < 5; ++j) {
      i_state[j] = d_inW[j] * d_x[gid * SAMPLE_TEST_LEN + t];
      for (i = 0; i < 5; ++i)
        i_state[j] += h_state[i] * d_intW[j*5+i];
      i_state[j] += d_intB[j];
      i_state[j] = sigmoid(i_state[j]);
    }

    for (j = 0; j < 5; ++j) {
      f_state[j] = d_inW[5+j] * d_x[gid * SAMPLE_TEST_LEN + t];
      for (i = 0; i < 5; ++i)
        f_state[j] += h_state[i] * d_intW[25+j*5+i];
      f_state[j] += d_intB[5+j];
      f_state[j] = sigmoid(f_state[j]);
    }

    for (j = 0; j < 5; ++j) {
      o_state[j] = d_inW[10+j] * d_x[gid * SAMPLE_TEST_LEN + t];
      for (i = 0; i < 5; ++i)
        o_state[j] += h_state[i] * d_intW[50+j*5+i];
      o_state[j] += d_intB[10+j];
      o_state[j] = sigmoid(o_state[j]);
    }

    for (j = 0; j < 5; ++j) {
      g_state[j] = d_inW[15+j] * d_x[gid * SAMPLE_TEST_LEN + t];
      for (i = 0; i < 5; ++i)
        g_state[j] += h_state[i] * d_intW[75+j*5+i];
      g_state[j] += d_intB[15+j];
      g_state[j] = tanh(g_state[j]);
    }

    for (j = 0; j < 5; ++j) {
      c_state[j] = c_state[j] * f_state[j] + g_state[j] * i_state[j];
      h_state[j] = tanh(c_state[j]) * o_state[j];
    }

    d_y[gid * SAMPLE_TEST_LEN + t] = d_outB[0];
    for (j = 0; j < 5; ++j)
      d_y[gid * SAMPLE_TEST_LEN + t] += h_state[j] * d_outW[j];
  }
}
