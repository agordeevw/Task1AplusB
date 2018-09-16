__kernel void aplusb(
  __global const float* as,
  __global const float* bs,
  __global float* cs,
  unsigned int size)
{
  const size_t n = get_global_id(0);
  if (n < size) {
    cs[n] = as[n] + bs[n];
  }
}
