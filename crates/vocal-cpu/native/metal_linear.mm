#include "internal.h"
#include "projection_rows.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

namespace vocal::cpu {
struct MetalLinearContext {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  id<MTLCommandQueue> queue = [device newCommandQueue];
  id<MTLBuffer> input;
  id<MTLBuffer> output;
};

extern "C" bool vocal_cpu_metal_available() {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    return device && MPSSupportsMTLDevice(device) && [device newCommandQueue] != nil;
  }
}

static MPSMatrixDescriptor *matrix_descriptor(NSUInteger rows,
                                              NSUInteger columns) {
  const auto stride = [MPSMatrixDescriptor rowBytesFromColumns:columns
                                                    dataType:MPSDataTypeFloat32];
  return [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                               columns:columns
                                              rowBytes:stride
                                              dataType:MPSDataTypeFloat32];
}

static id<MTLBuffer> reserve(id<MTLDevice> device, id<MTLBuffer> buffer,
                             MPSMatrixDescriptor *descriptor) {
  require(descriptor.rowBytes && descriptor.rows <= device.maxBufferLength / descriptor.rowBytes,
          "Metal matrix exceeds the device buffer length");
  const auto bytes = descriptor.rows * descriptor.rowBytes;
  if (buffer.length < bytes)
    buffer = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  require(buffer != nil, "Metal matrix buffer allocation failed");
  return buffer;
}

static void upload(id<MTLBuffer> buffer, MPSMatrixDescriptor *descriptor,
                   const void *input) {
  upload_projection_rows(buffer.contents, descriptor.rowBytes, input,
                         descriptor.rows, descriptor.columns);
}

class MetalLinear final : public Kernel {
  const size_t id;
  MPSMatrixDescriptor *input_layout;
  MPSMatrixDescriptor *output_layout;
  MPSMatrix *weights;
  MPSMatrixMultiplication *operation;

public:
  MetalLinear(Graph &g, size_t id) : id(id) {
    auto &context = *g.metal_linear;
    const auto &node = g.nodes[id];
    const auto &input = g.tensors[node.inputs[0]];
    const auto &weight = g.tensors[node.inputs[1]];
    const auto rows = input.elements() / input.shape.back();
    const auto columns = g.tensors[id].shape.back();
    const auto width = input.shape.back();
    input_layout = matrix_descriptor(rows, width);
    output_layout = matrix_descriptor(rows, columns);
    auto layout = matrix_descriptor(weight.shape[0], weight.shape[1]);
    auto buffer = reserve(context.device, nil, layout);
    upload(buffer, layout, weight.data());
    weights = [[MPSMatrix alloc] initWithBuffer:buffer descriptor:layout];
    operation = [[MPSMatrixMultiplication alloc] initWithDevice:context.device
                                               transposeLeft:NO transposeRight:node.attrs[0] != 0
                                               resultRows:rows resultColumns:columns
                                               interiorColumns:width alpha:1.0 beta:0.0];
    require(weights && operation, "Metal projection preparation failed");
    packed_inputs = {1};
    packed_bytes = buffer.length;
    implementation = "metal:mps-matrix-fp32";
  }

  void run(Graph &g) override {
    @autoreleasepool {
      auto &context = *g.metal_linear;
      const auto &node = g.nodes[id];
      g.stream.wait();
      context.input = reserve(context.device, context.input, input_layout);
      context.output = reserve(context.device, context.output, output_layout);
      upload(context.input, input_layout, g.tensors[node.inputs[0]].data());
      auto input = [[MPSMatrix alloc] initWithBuffer:context.input descriptor:input_layout];
      auto output = [[MPSMatrix alloc] initWithBuffer:context.output descriptor:output_layout];
      auto command = [context.queue commandBuffer];
      require(input && output && command, "Metal projection encoding failed");
      [operation encodeToCommandBuffer:command leftMatrix:input rightMatrix:weights resultMatrix:output];
      [command commit];
      [command waitUntilCompleted];
      if (command.status != MTLCommandBufferStatusCompleted) {
        const char *message = command.error.localizedDescription.UTF8String;
        throw std::runtime_error(message ? message : "Metal projection execution failed");
      }
      const auto *bias = node.inputs.size() == 3
          ? static_cast<const float *>(g.tensors[node.inputs[2]].data()) : nullptr;
      download_projection_rows(static_cast<float *>(g.tensors[id].data()),
                               context.output.contents, output_layout.rowBytes,
                               output_layout.rows, output_layout.columns, bias);
    }
  }
};

std::unique_ptr<Kernel> prepare_metal_linear(Graph &g, size_t id) {
  const auto &node = g.nodes[id];
  const auto &input = g.tensors[node.inputs[0]];
  const auto &weight = g.tensors[node.inputs[1]];
  const auto &output = g.tensors[id];
  if (node.folded || node.residual.storage || node.activation != ActivationPostOp::None ||
      !g.nodes[node.inputs[1]].folded || input.dtype != 1 || weight.dtype != 1 ||
      output.dtype != 1 || !input.contiguous() || !weight.contiguous() ||
      !output.contiguous() || weight.shape.size() != 2 ||
      input.shape.size() < 2 || output.shape.size() < 2 ||
      input.shape.back() <= 0 || output.shape.back() <= 0)
    return nullptr;
  if (node.inputs.size() == 3) {
    const auto &bias = g.tensors[node.inputs[2]];
    if (bias.dtype != 1 || !bias.contiguous() || bias.elements() != output.shape.back())
      return nullptr;
  }
  const bool transpose = node.attrs[0] != 0;
  if (weight.shape[transpose ? 1 : 0] != input.shape.back() ||
      weight.shape[transpose ? 0 : 1] != output.shape.back() ||
      input.elements() / input.shape.back() != output.elements() / output.shape.back())
    return nullptr;
  @autoreleasepool {
    if (!g.metal_linear) {
      g.metal_linear = std::make_shared<MetalLinearContext>();
      require(g.metal_linear->queue != nil, "Metal command queue unavailable");
    }
    return std::make_unique<MetalLinear>(g, id);
  }
}
} // namespace vocal::cpu
