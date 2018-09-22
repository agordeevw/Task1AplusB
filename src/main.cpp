#include <CL/cl.h>
#include <libclew/ocl_init.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>

#include <vector>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <fstream>
#include <cassert>
#include <climits>
#include <algorithm>


template <typename T>
std::string to_string(T value)
{
    std::ostringstream ss;
    ss << value;
    return ss.str();
}

void reportError(cl_int err, const std::string &filename, int line)
{
    if (CL_SUCCESS == err)
        return;

    // Таблица с кодами ошибок:
    // libs/clew/CL/cl.h:103
    // P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
    std::string message = "OpenCL error code " + to_string(err) + " encountered at " + filename + ":" + to_string(line);
    throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)

template <class T>
class OCLObject {
public:
  OCLObject() : object(nullptr) {}

  ~OCLObject() {
    if (object)
      release(object);
  }

  T get() { return object; }

  void reset(T other) {
    this->~OCLObject();
    object = other;
  }

private:
  static void release(cl_context object) {
    OCL_SAFE_CALL(clReleaseContext(object));
  }
  static void release(cl_command_queue object) {
    OCL_SAFE_CALL(clReleaseCommandQueue(object));
  }
  static void release(cl_mem object) {
    OCL_SAFE_CALL(clReleaseMemObject(object));
  }
  static void release(cl_program object) {
    OCL_SAFE_CALL(clReleaseProgram(object));
  }
  static void release(cl_kernel object) {
    OCL_SAFE_CALL(clReleaseKernel(object));
  }

  T object;
};

class OCLTaskApp {
public:
  OCLTaskApp() {
    if (!ocl_init())
      throw std::runtime_error("Can't init OpenCL driver!");
  }

  void choose_device() {
    device = choose_best_device_of_type(CL_DEVICE_TYPE_GPU);
    if (!device)
      device = choose_best_device_of_type(CL_DEVICE_TYPE_CPU);
    if (!device)
      throw std::runtime_error("No OCL device available");

    std::size_t device_name_size;
    OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &device_name_size));
    std::vector<char> device_name(device_name_size);
    OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, device_name_size, device_name.data(), nullptr));
    std::cout << "Chosen device:\t" << device_name.data() << std::endl;
  }

  void create_context() {
    cl_int errcode;
    context.reset(clCreateContext(nullptr, 1, &device, nullptr, nullptr, &errcode));
    OCL_SAFE_CALL(errcode);
  }

  void create_command_queue() {
    cl_int errcode;
    command_queue.reset(clCreateCommandQueue(context.get(), device, 0, &errcode));
    OCL_SAFE_CALL(errcode);
  }

  void create_buffers_for_input(float* as, float* bs, unsigned int n) {
    cl_mem_flags flag = CL_MEM_READ_ONLY;
    cl_int errcode;
    as_gpu.reset(clCreateBuffer(context.get(), flag, n * sizeof(float), nullptr, &errcode));
    OCL_SAFE_CALL(errcode);
    bs_gpu.reset(clCreateBuffer(context.get(), flag, n * sizeof(float), nullptr, &errcode));
    OCL_SAFE_CALL(errcode);

    cl_event buffers_written[2];
    OCL_SAFE_CALL(clEnqueueWriteBuffer(command_queue.get(), as_gpu.get(), CL_FALSE, 0,
      n * sizeof(float), as, 0, nullptr, &buffers_written[0]));
    OCL_SAFE_CALL(clEnqueueWriteBuffer(command_queue.get(), bs_gpu.get(), CL_FALSE, 0,
      n * sizeof(float), bs, 0, nullptr, &buffers_written[1]));
    OCL_SAFE_CALL(clWaitForEvents(2, buffers_written));
  }

  void create_buffer_for_output(unsigned int n) {
    cl_mem_flags flag = CL_MEM_WRITE_ONLY;
    cl_int errcode;
    cs_gpu.reset(clCreateBuffer(context.get(), flag, n * sizeof(float), nullptr, &errcode));
    OCL_SAFE_CALL(errcode);
  }

  void create_program(const std::string& kernel_source) {
    const char* source = kernel_source.data();
    const std::size_t length = kernel_source.size();
    cl_int errcode;
    program.reset(clCreateProgramWithSource(context.get(), 1, &source, &length, &errcode));
    OCL_SAFE_CALL(errcode);
  }

  void try_compile_program() {
    const cl_int errcode = clBuildProgram(program.get(), 1, &device, nullptr, nullptr, nullptr);
    if (errcode == CL_SUCCESS || errcode == CL_BUILD_PROGRAM_FAILURE) {
      std::size_t log_size;
      OCL_SAFE_CALL(clGetProgramBuildInfo(program.get(), device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size));
      std::vector<char> log(log_size);
      OCL_SAFE_CALL(clGetProgramBuildInfo(program.get(), device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr));
      std::cout << "Program compilation log:\n" << log.data() << std::endl;

      if (errcode == CL_SUCCESS)
        std::cout << "Program compilation succeeded\n" << std::endl;
      if (errcode == CL_BUILD_PROGRAM_FAILURE)
        throw std::runtime_error("Program compilation failed");
    }
    OCL_SAFE_CALL(errcode);
  }

  void create_kernel() {
    cl_int errcode;
    kernel.reset(clCreateKernel(program.get(), "aplusb", &errcode));
    OCL_SAFE_CALL(errcode);
  }

  void set_kernel_args(unsigned int size) {
    OCL_SAFE_CALL(clSetKernelArg(kernel.get(), 0, sizeof(cl_mem), &as_gpu));
    OCL_SAFE_CALL(clSetKernelArg(kernel.get(), 1, sizeof(cl_mem), &bs_gpu));
    OCL_SAFE_CALL(clSetKernelArg(kernel.get(), 2, sizeof(cl_mem), &cs_gpu));
    OCL_SAFE_CALL(clSetKernelArg(kernel.get(), 3, sizeof(unsigned int), &size));
  }

  void execute_kernel(std::size_t global_work_size) {
    cl_event kernel_finished_event;
    OCL_SAFE_CALL(clEnqueueNDRangeKernel(command_queue.get(), kernel.get(), 1, 
      nullptr, &global_work_size, nullptr, 0, nullptr, &kernel_finished_event));
    OCL_SAFE_CALL(clWaitForEvents(1, &kernel_finished_event));
  }

  void transfer_output_buffer_to_host(float* cs, unsigned int n) {
    OCL_SAFE_CALL(clEnqueueReadBuffer(command_queue.get(), cs_gpu.get(), 
      CL_TRUE, 0, sizeof(float) * n, cs, 0, nullptr, nullptr));
  }

private:
  class ScoredDevice {
  public:
    ScoredDevice(cl_device_id device)
      : device(device), score(score_device(device)) {}

    cl_device_id get_device() const { return device; }
    int get_score() const { return score; }

    bool operator<(const ScoredDevice& other) const {
      return score < other.score;
    }

  private:
    static int score_device(cl_device_id device) {
      cl_bool is_available;
      OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_AVAILABLE, sizeof(is_available), &is_available, nullptr));
      if (!is_available)
        return INT_MIN;

      cl_ulong global_memory_size;
      OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_memory_size), &global_memory_size, nullptr));
      int global_memory_size_in_mb = static_cast<int>(global_memory_size / 1024 / 1024);

      return global_memory_size_in_mb;
    }

    cl_device_id device;
    int score;
  };

  static cl_device_id choose_best_device_of_type(cl_device_type type) {
    auto devices = find_all_devices_of_type(type);
    std::vector<ScoredDevice> scored_devices;
    for (auto device : devices)
      scored_devices.push_back(ScoredDevice(device));

    auto best_device_it = std::max_element(scored_devices.begin(), scored_devices.end());
    constexpr int minimal_acceptable_score = 0;
    return (best_device_it != scored_devices.end() 
      && best_device_it->get_score() >= minimal_acceptable_score)
      ? best_device_it->get_device()
      : nullptr;
  }

  static std::vector<cl_device_id> find_all_devices_of_type(cl_device_type type) {
    cl_uint num_platforms;
    OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &num_platforms));
    std::vector<cl_platform_id> platforms(num_platforms);
    OCL_SAFE_CALL(clGetPlatformIDs(num_platforms, platforms.data(), nullptr));

    std::vector<cl_device_id> devices;
    for (auto& platform : platforms) {
      cl_uint num_devices;
      const cl_int errcode = clGetDeviceIDs(platform, type, 0, nullptr, &num_devices);
      if (errcode == CL_SUCCESS) {
        const auto old_size = devices.size();
        devices.resize(devices.size() + num_devices);
        OCL_SAFE_CALL(clGetDeviceIDs(platform, type, num_devices, &devices[old_size], nullptr));
      } else if (errcode != CL_DEVICE_NOT_FOUND) {
        OCL_SAFE_CALL(errcode);
      }
    }

    return devices;
  }

  cl_device_id device = nullptr;
  OCLObject<cl_context> context;
  OCLObject<cl_command_queue> command_queue;
  OCLObject<cl_mem> as_gpu;
  OCLObject<cl_mem> bs_gpu;
  OCLObject<cl_mem> cs_gpu;
  OCLObject<cl_program> program;
  OCLObject<cl_kernel> kernel;
};

int main()
try
{
    OCLTaskApp app;

    // TODO 1 По аналогии с заданием Example0EnumDevices узнайте какие есть устройства, и выберите из них какое-нибудь
    // (если есть хоть одна видеокарта - выберите ее, если нету - выбирайте процессор)
    app.choose_device();

    // TODO 2 Создайте контекст с выбранным устройством
    // См. документацию https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL Runtime -> Contexts -> clCreateContext
    // Не забывайте проверять все возвращаемые коды на успешность (обратите внимание что в данном случае метод возвращает
    // код по переданному аргументом errcode_ret указателю)
    // И хорошо бы сразу добавить в конце clReleaseContext (да, не очень RAII, но это лишь пример)
    app.create_context();

    // TODO 3 Создайте очередь выполняемых команд в рамках выбранного контекста и устройства
    // См. документацию https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL Runtime -> Runtime APIs -> Command Queues -> clCreateCommandQueue
    // Убедитесь что в соответствии с документацией вы создали in-order очередь задач
    // И хорошо бы сразу добавить в конце clReleaseQueue
    app.create_command_queue();

    unsigned int n = 100 * 1000 * 1000;
    // Создаем два массива псевдослучайных данных для сложения и массив для будущего хранения результата
    std::vector<float> as(n, 0);
    std::vector<float> bs(n, 0);
    std::vector<float> cs(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = r.nextf();
        bs[i] = r.nextf();
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    // TODO 4 Создайте три буфера в памяти устройства (в случае видеокарты - в видеопамяти - VRAM) - для двух суммируемых массивов as и bs (они read-only) и для массива с результатом cs (он write-only)
    // См. Buffer Objects -> clCreateBuffer
    // Размер в байтах соответственно можно вычислить через sizeof(float)=4 и тот факт что чисел в каждом массиве - n штук
    // Данные в as и bs можно прогрузить этим же методом скопировав данные из host_ptr=as.data() (и не забыв про битовый флаг на это указывающий)
    // или же через метод Buffer Objects -> clEnqueueWriteBuffer
    // И хорошо бы сразу добавить в конце clReleaseMemObject (аналогично все дальнейшие ресурсы вроде OpenCL под-программы, кернела и т.п. тоже нужно освобождать)
    app.create_buffers_for_input(as.data(), bs.data(), n);
    app.create_buffer_for_output(n);

    // TODO 6 Выполните TODO 5 (реализуйте кернел в src/cl/aplusb.cl)
    // затем убедитесь что выходит загрузить его с диска (убедитесь что Working directory выставлена правильно - см. описание задания)
    // напечатав исходники в консоль (if проверяет что удалось считать хоть что-то)
    std::string kernel_sources;
    {
        std::ifstream file("src/cl/aplusb.cl");
        kernel_sources = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        if (kernel_sources.size() == 0) {
            throw std::runtime_error("Empty source file! May be you forgot to configure working directory properly?");
        }
        std::cout
            << "Kernel sources:\n"
            "-------------------------\n"
            << kernel_sources
            << "-------------------------\n"
            << std::endl;
    }

    // TODO 7 Создайте OpenCL-подпрограмму с исходниками кернела
    // см. Runtime APIs -> Program Objects -> clCreateProgramWithSource
    // у string есть метод c_str(), но обратите внимание что передать вам нужно указатель на указатель
    app.create_program(kernel_sources);

    // TODO 8 Теперь скомпилируйте программу и напечатайте в консоль лог компиляции
    // см. clBuildProgram
    // А так же напечатайте лог компиляции (он будет очень полезен, если в кернеле есть синтаксические ошибки - т.е. когда clBuildProgram вернет CL_BUILD_PROGRAM_FAILURE)
    // см. clGetProgramBuildInfo
    //    size_t log_size = 0;
    //    std::vector<char> log(log_size, 0);
    //    if (log_size > 1) {
    //        std::cout << "Log:" << std::endl;
    //        std::cout << log.data() << std::endl;
    //    }
    app.try_compile_program();

    // TODO 9 Создайте OpenCL-kernel в созданной подпрограмме (в одной подпрограмме может быть несколько кернелов, но в данном случае кернел один)
    // см. подходящую функцию в Runtime APIs -> Program Objects -> Kernel Objects
    app.create_kernel();

    // TODO 10 Выставите все аргументы в кернеле через clSetKernelArg (as_gpu, bs_gpu, cs_gpu и число значений, убедитесь что тип количества элементов такой же в кернеле)
    {
        // unsigned int i = 0;
        // clSetKernelArg(kernel, i++, ..., ...));
        // clSetKernelArg(kernel, i++, ..., ...));
        // clSetKernelArg(kernel, i++, ..., ...));
        // clSetKernelArg(kernel, i++, ..., ...));
    }
    app.set_kernel_args(n);

    // TODO 11 Выше увеличьте n с 1000*1000 до 100*1000*1000 (чтобы дальнейшие замеры были ближе к реальности)
    // (уже сделано)

    // TODO 12 Запустите выполнения кернела:
    // - С одномерной рабочей группой размера 128
    // - В одномерном рабочем пространстве размера roundedUpN, где roundedUpN - наименьшее число кратное 128 и при этом не меньшее n
    // - см. clEnqueueNDRangeKernel
    // - Обратите внимание что чтобы дождаться окончания вычислений (чтобы знать когда можно смотреть результаты в cs_gpu) нужно:
    //   - Сохранить событие "кернел запущен" (см. аргумент "cl_event *event")
    //   - Дождаться завершения полунного события - см. в документации подходящий метод среди Event Objects
    {
        size_t workGroupSize = 128;
        size_t global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
        timer t; // Это вспомогательный секундомер, он замеряет время своего создания и позволяет усреднять время нескольких замеров
        for (unsigned int i = 0; i < 20; ++i) {
            // clEnqueueNDRangeKernel...
            // clWaitForEvents...
            app.execute_kernel(global_work_size);

            t.nextLap(); // При вызове nextLap секундомер запоминает текущий замер (текущий круг) и начинает замерять время следующего круга
        }
        // Среднее время круга (вычисления кернела) на самом деле считаются не по всем замерам, а лишь с 20%-перцентайля по 80%-перцентайль (как и стандартное отклониение)
        // подробнее об этом - см. timer.lapsFiltered
        // P.S. чтобы в CLion быстро перейти к символу (функции/классу/много чему еще) достаточно нажать Ctrl+Shift+Alt+N -> lapsFiltered -> Enter
        std::cout << "Kernel average time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;

        // TODO 13 Рассчитайте достигнутые гигафлопcы:
        // - Всего элементов в массивах по n штук
        // - Всего выполняется операций: операция a+b выполняется n раз
        // - Флопс - это число операций с плавающей точкой в секунду
        // - В гигафлопсе 10^9 флопсов
        // - Среднее время выполнения кернела равно t.lapAvg() секунд
        const std::size_t total_ops = n;
        const double ops_per_second = static_cast<double>(n) / t.lapAvg();
        const double gflops = ops_per_second / (1000.0 * 1000.0 * 1000.0);
        std::cout << "GFlops: " << gflops << std::endl;

        // TODO 14 Рассчитайте используемую пропускную способность обращений к видеопамяти (в гигабайтах в секунду)
        // - Всего элементов в массивах по n штук
        // - Размер каждого элемента sizeof(float)=4 байта
        // - Обращений к видеопамяти т.о. 2*n*sizeof(float) байт на чтение и 1*n*sizeof(float) байт на запись, т.е. итого 3*n*sizeof(float) байт
        // - В гигабайте 1024*1024*1024 байт
        // - Среднее время выполнения кернела равно t.lapAvg() секунд
        const std::size_t total_vram_memory_accesses = 3 * n * sizeof(float);
        const double vram_bandwidth = static_cast<double>(total_vram_memory_accesses) / t.lapAvg();
        const double vram_bandwidth_in_gbytes = vram_bandwidth / static_cast<double>(1024 * 1024 * 1024);
        std::cout << "VRAM bandwidth: " << vram_bandwidth_in_gbytes << " GB/s" << std::endl;
    }

    // TODO 15 Скачайте результаты вычислений из видеопамяти (VRAM) в оперативную память (RAM) - из cs_gpu в cs (и рассчитайте скорость трансфера данных в гигабайтах в секунду)
    {
        timer t;
        for (unsigned int i = 0; i < 20; ++i) {
            // clEnqueueReadBuffer...
            app.transfer_output_buffer_to_host(cs.data(), n);

            t.nextLap();
        }
        std::cout << "Result data transfer time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;

        const std::size_t total_transfer_data_size = n * sizeof(float);
        const double transfer_rate = static_cast<double>(total_transfer_data_size) / t.lapAvg();
        const double transfer_rate_in_gbytes = transfer_rate / (1000.0 * 1000.0 * 1000.0);
        std::cout << "VRAM -> RAM bandwidth: " << transfer_rate_in_gbytes << " GB/s" << std::endl;
    }

    // TODO 16 Сверьте результаты вычислений со сложением чисел на процессоре (и убедитесь, что если в кернеле сделать намеренную ошибку, то эта проверка поймает ошибку)
    for (unsigned int i = 0; i < n; ++i) {
        if (cs[i] != as[i] + bs[i]) {
            throw std::runtime_error("CPU and GPU results differ!");
        }
    }
    std::cout << "CPU and GPU results are identical\n";

    return EXIT_SUCCESS;
}
catch (const std::runtime_error& e) {
    std::cout << "ERROR: " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::bad_alloc&) {
    std::cout << "ERROR: not enough memory, close other applications and try again" << std::endl;
    return EXIT_FAILURE;
}